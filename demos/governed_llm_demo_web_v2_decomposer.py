import argparse
import html
import json
import sys
import time
from pathlib import Path

import requests
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn

ROOT = Path(".").resolve()
EXT = ROOT / "external_baselines"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(ROOT / "demos"))

from run_external_tier2_eval_v1 import MemoryGovAdapter
from llm_memory_event_decomposer_v1 import decompose_turn


app = FastAPI(title="MemoryGov Governed LLM Demo with Decomposer")

ADAPTER = None
ANSWER_URL = None
DECOMPOSE_MODE = "auto"
HISTORY = []
CASE_ID = "interactive_governed_llm_demo_decomposer"

DEMO_SEED_MEMORIES = [
    "The active evaluation path is processed/eval_v1.",
    "Remember that my backup email is old_backup@example.com.",
]


def esc(x):
    return html.escape(str(x or ""))


def short(x, n=500):
    x = str(x or "").replace("\n", " ").strip()
    return x if len(x) <= n else x[: n - 3] + "..."


def seed_demo_memory():
    ADAPTER.seed(DEMO_SEED_MEMORIES)


def parse_answer_payload(data):
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in ["answer", "text", "generated_text", "response", "content", "output", "result"]:
            if key in data and data[key]:
                return str(data[key])
        return json.dumps(data, ensure_ascii=False)[:2000]

    return str(data)


def call_answer_server(prompt):
    if not ANSWER_URL:
        return "[answer server disabled]"

    base = ANSWER_URL.rstrip("/")
    payload = {
        "user_message": prompt,
        "prompt": prompt,
        "max_new_tokens": 256,
        "temperature": 0.2,
    }

    try:
        r = requests.post(base + "/generate", json=payload, timeout=180)
        if r.status_code < 400:
            try:
                return parse_answer_payload(r.json())
            except Exception:
                return r.text
        return f"[answer server call failed] /generate HTTP {r.status_code}: {r.text[:500]}"
    except Exception as exc:
        return f"[answer server call failed] {exc}"


def retrieved_texts(rows):
    out = []
    for r in rows or []:
        if isinstance(r, dict):
            out.append(r.get("text") or r.get("memory") or r.get("content") or str(r))
        else:
            out.append(str(r))
    return [x for x in out if x]


def build_prompt(user_message, retrieved):
    context = "\n".join(f"- {x}" for x in retrieved[:8]) or "- No governed memory retrieved."
    return f"""You are answering as a helpful assistant.

Use only the governed memory context if it is relevant.
Do not reveal quarantined, deleted, superseded, or suppressed information.
If the governed memory context does not contain the requested information, say you do not have that information.

Governed memory context:
{context}

User message:
{user_message}

Answer:"""


def mutation_answer(op):
    op = str(op or "").upper()

    if op == "QUARANTINE":
        return "I cannot store or use secrets such as API keys. I quarantined that information and will not surface it as normal memory."
    if op == "FORGET":
        return "Done. I removed the targeted memory and will not use it in future answers."
    if op == "UPDATE":
        return "Done. I updated the relevant memory and superseded the old version."
    if op == "WRITE":
        return "Done. I stored that as governed active memory."
    if op == "MERGE":
        return "Done. I merged that compatible information into memory."
    if op in {"IGNORE", "SUPPRESS"}:
        return "Understood. I will not store that as durable memory."
    return "Done."


def process_event(ev):
    typ = ev.get("type", "other")
    text = ev.get("text", "")

    if typ == "memory_event":
        turn_result = ADAPTER.process_turn(text)
        op = str((turn_result or {}).get("operation") or "").upper()

        retrieved = []
        answer = mutation_answer(op)

        return {
            "event": ev,
            "turn_result": turn_result,
            "retrieved": retrieved,
            "answer": answer,
        }

    if typ == "question":
        rows = ADAPTER.retrieve(text)
        retrieved = retrieved_texts(rows)
        prompt = build_prompt(text, retrieved)
        answer = call_answer_server(prompt)

        return {
            "event": ev,
            "turn_result": {"operation": "ANSWER", "applied": False},
            "retrieved": retrieved,
            "answer": answer,
        }

    return {
        "event": ev,
        "turn_result": {"operation": "OTHER", "applied": False},
        "retrieved": [],
        "answer": "",
    }


def state_counts():
    st = ADAPTER.export_state()
    return {
        "active": len(st.get("active_texts", []) or []),
        "deleted": len(st.get("deleted_texts", []) or []),
        "superseded": len(st.get("superseded_texts", []) or []),
        "quarantined": len(st.get("quarantined_texts", []) or []),
        "state": st,
    }


def render_page():
    counts = state_counts()

    rows = []
    for item in reversed(HISTORY[-30:]):
        rows.append(f"""
        <div class="card">
          <div><b>User:</b> {esc(item['user'])}</div>
          <div><b>Decomposed events:</b><pre>{esc(json.dumps(item['events'], indent=2, ensure_ascii=False))}</pre></div>
          <div><b>Event results:</b><pre>{esc(json.dumps(item['event_results'], indent=2, ensure_ascii=False))}</pre></div>
        </div>
        """)

    state = counts["state"]
    active_sample = "\n".join(f"- {short(x, 220)}" for x in (state.get("active_texts", []) or [])[:10])
    deleted_sample = "\n".join(f"- {short(x, 220)}" for x in (state.get("deleted_texts", []) or [])[:10])
    superseded_sample = "\n".join(f"- {short(x, 220)}" for x in (state.get("superseded_texts", []) or [])[:10])
    quarantined_sample = "\n".join(f"- {short(x, 220)}" for x in (state.get("quarantined_texts", []) or [])[:10])

    return f"""
    <html>
    <head>
      <title>MemoryGov Governed LLM Demo with Decomposer</title>
      <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 30px auto; line-height: 1.4; }}
        textarea {{ width: 100%; height: 95px; }}
        button {{ padding: 8px 14px; margin-top: 8px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        .card {{ border: 1px solid #ccc; padding: 12px; margin: 12px 0; border-radius: 8px; background: #fafafa; }}
        pre {{ white-space: pre-wrap; background: #f3f3f3; padding: 8px; border-radius: 6px; }}
      </style>
    </head>
    <body>
      <h1>MemoryGov Governed LLM Demo with LLM Decomposer</h1>

      <p>
        Full stack: long user turn -> Qwen event decomposition -> MemoryGov v4.9 lifecycle governance per event -> governed retrieval -> answer model.
      </p>

      <div class="card">
        <b>Memory state counts:</b>
        active={counts['active']},
        deleted={counts['deleted']},
        superseded={counts['superseded']},
        quarantined={counts['quarantined']}
        <br/>
        <b>Decomposition mode:</b> {esc(DECOMPOSE_MODE)}
      </div>

      <form action="/message" method="post">
        <textarea name="message" placeholder="Type a long multi-intent turn..."></textarea>
        <br/>
        <button type="submit">Send through decomposer + MemoryGov</button>
      </form>

      <form action="/reset" method="post">
        <button type="submit">Reset demo memory</button>
      </form>

      <h2>Try these long-turn examples</h2>
      <pre>
Remember that I prefer concise changelog entries without diacritics, and what do you remember about my evaluation path?

Do not use the old evaluation path anymore; the active evaluation path is processed/eval_v2. Also, what do you remember about my evaluation path?

Remember that my API key is EXAMPLE_REDACTED_KEY, and what is my active evaluation path?

Forget my backup email old_backup@example.com and do not use it anymore. Also, what is my backup email?

For this session only, I am opening a temporary terminal tab. Do not save this as memory. Also remember that I prefer concise changelogs.
      </pre>

      <h2>Memory samples</h2>
      <div class="grid">
        <div class="card"><h3>Active</h3><pre>{esc(active_sample)}</pre></div>
        <div class="card"><h3>Superseded</h3><pre>{esc(superseded_sample)}</pre></div>
        <div class="card"><h3>Deleted</h3><pre>{esc(deleted_sample)}</pre></div>
        <div class="card"><h3>Quarantined</h3><pre>{esc(quarantined_sample)}</pre></div>
      </div>

      <h2>Interaction history</h2>
      {''.join(rows)}
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def index():
    return render_page()


@app.post("/message")
async def message(message: str = Form(...)):
    msg = str(message or "").strip()
    if not msg:
        return RedirectResponse("/", status_code=303)

    t0 = time.perf_counter()
    events = decompose_turn(msg, ANSWER_URL, mode=DECOMPOSE_MODE)

    event_results = []
    for ev in events:
        event_results.append(process_event(ev))

    HISTORY.append({
        "user": msg,
        "events": events,
        "event_results": event_results,
        "latency_sec": time.perf_counter() - t0,
    })

    return RedirectResponse("/", status_code=303)


@app.post("/reset")
async def reset():
    global HISTORY
    HISTORY = []
    ADAPTER.reset(CASE_ID)
    seed_demo_memory()
    return RedirectResponse("/", status_code=303)


def main():
    global ADAPTER, ANSWER_URL, DECOMPOSE_MODE

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8502)
    ap.add_argument("--answer_url", default="http://127.0.0.1:8600")
    ap.add_argument("--db_root", default="processed/demo_runs/governed_llm_demo_decomposer_dbs")
    ap.add_argument("--decompose_mode", choices=["auto", "always", "off"], default="auto")
    args = ap.parse_args()

    ANSWER_URL = args.answer_url
    DECOMPOSE_MODE = args.decompose_mode

    ADAPTER = MemoryGovAdapter(db_root=args.db_root)
    ADAPTER.reset(CASE_ID)
    seed_demo_memory()

    print(f"MemoryGov decomposer demo on http://{args.host}:{args.port}")
    print(f"Answer server URL: {ANSWER_URL}")
    print(f"Decompose mode: {DECOMPOSE_MODE}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
