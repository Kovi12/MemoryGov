import json
import re
import requests


def extract_text_from_response(data):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for k in ["answer", "text", "generated_text", "response", "content", "output", "result"]:
            if k in data and data[k]:
                return str(data[k])
        return json.dumps(data, ensure_ascii=False)
    return str(data)


def extract_json_object(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    a = text.find("{")
    b = text.rfind("}")
    if a >= 0 and b > a:
        return json.loads(text[a:b + 1])

    raise ValueError("No JSON object found in decomposer output")


def normalize_events(obj, fallback_text):
    events = obj.get("events", []) if isinstance(obj, dict) else []
    out = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        typ = str(ev.get("type", "")).strip().lower()
        txt = str(ev.get("text", "")).strip()

        if typ not in {"memory_event", "question", "other"}:
            typ = "other"

        if txt:
            out.append({"type": typ, "text": txt})

    if not out:
        out = [{"type": "question" if "?" in fallback_text else "memory_event", "text": fallback_text.strip()}]

    return out


def needs_decomposition(text):
    t = str(text or "").lower()
    markers = [
        " also ", " and ", " but ", ";", "\n",
        "remember", "forget", "do not use", "don't use", "no longer",
        "for this session", "do not save", "api key", "password", "secret",
        "what ", "which ", "where ", "when ", "how ", "?"
    ]
    hits = sum(1 for m in markers if m in f" {t} ")
    return len(t.split()) > 18 or hits >= 2


def call_generate(answer_url, user_message, max_new_tokens=512):
    base = str(answer_url or "").rstrip("/")
    if not base:
        raise RuntimeError("answer_url is empty")

    payload = {
        "user_message": user_message,
        "prompt": user_message,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
    }

    r = requests.post(base + "/generate", json=payload, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"/generate HTTP {r.status_code}: {r.text[:500]}")

    try:
        return extract_text_from_response(r.json())
    except Exception:
        return r.text


def decompose_turn(user_turn, answer_url, mode="auto"):
    user_turn = str(user_turn or "").strip()
    if not user_turn:
        return []

    if mode == "off":
        return [{"type": "question" if "?" in user_turn else "memory_event", "text": user_turn}]

    if mode == "auto" and not needs_decomposition(user_turn):
        return [{"type": "question" if "?" in user_turn else "memory_event", "text": user_turn}]

    prompt = f"""
You are a segmentation engine for a memory-governance system.

Your ONLY task is to copy the user's turn into atomic spans.

Return strict JSON only:
{{"events":[{{"type":"memory_event|question|other","text":"EXACT COPIED SPAN"}}]}}

Critical rules:
1. Do not paraphrase.
2. Do not summarize.
3. Do not change paths, emails, tokens, names, numbers, or wording.
4. The "text" field must be copied verbatim from the user turn, except you may remove leading connectors such as "also," or "and".
5. Split into atomic spans when the user gives multiple intentions.
6. A span is memory_event if it asks to remember, forget, update, correct, replace, not use old information, store a secret, or says something should not be saved.
7. A span is question if it asks for information now.
8. A span is other only if it is neither memory-affecting nor a question.
9. Do not output lifecycle labels like WRITE, UPDATE, FORGET, QUARANTINE.
10. Do not answer the user.

Examples:

User turn:
Remember that I prefer concise changelogs, and what is my active evaluation path?
Output:
{{"events":[{{"type":"memory_event","text":"Remember that I prefer concise changelogs"}},{{"type":"question","text":"what is my active evaluation path?"}}]}}

User turn:
Do not use the old evaluation path anymore; the active evaluation path is processed/eval_v2. Also, what do you remember about my evaluation path?
Output:
{{"events":[{{"type":"memory_event","text":"Do not use the old evaluation path anymore; the active evaluation path is processed/eval_v2."}},{{"type":"question","text":"what do you remember about my evaluation path?"}}]}}

User turn:
Remember that my API key is EXAMPLE_REDACTED_KEY, and what do you remember about my evaluation path?
Output:
{{"events":[{{"type":"memory_event","text":"Remember that my API key is EXAMPLE_REDACTED_KEY"}},{{"type":"question","text":"what do you remember about my evaluation path?"}}]}}

User turn:
{user_turn}
Output:
""".strip()

    try:
        raw = call_generate(answer_url, prompt)
        obj = extract_json_object(raw)
        return normalize_events(obj, user_turn)
    except Exception as exc:
        return [
            {
                "type": "memory_event" if "?" not in user_turn else "question",
                "text": user_turn,
                "decomposer_error": str(exc),
            }
        ]
