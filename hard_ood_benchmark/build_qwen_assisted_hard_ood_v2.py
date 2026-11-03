import json
import random
import re
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(31)

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

SEEDS_PATH = Path("processed/memorygovbench_operation_hard_ood_v1/operation_hard_ood_200.jsonl")
OUT_DIR = Path("processed/memorygovbench_operation_hard_ood_v2_qwen")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "operation_hard_ood_v2_qwen.jsonl"

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def make_prompt(row):
    return f"""You rewrite benchmark cases for a memory-governance operation classifier.

The gold label is fixed and must remain correct, but you must not mention the label:
{row["label"]}

Original case:
{row["text"]}

Rewrite the case to sound more natural and less templated.

Rules:
- Preserve the same intended operation.
- Do not include any of these words: WRITE, IGNORE, UPDATE, MERGE, FORGET, QUARANTINE.
- Avoid obvious trigger phrases when possible.
- Keep all facts required to infer the operation.
- You may add one harmless distractor sentence.
- Do not invent secrets except by preserving synthetic secrets already present.
- Return only valid JSON:
{{"rewritten_text":"..."}}
"""


def parse_json_text(s):
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    v = obj.get("rewritten_text")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def bad_rewrite(text):
    upper = text.upper()
    if any(lab in upper for lab in LABELS):
        return True
    if len(text) < 40:
        return True
    if len(text) > 4000:
        return True
    return False


def main():
    seeds = read_jsonl(SEEDS_PATH)
    print("seed rows:", len(seeds), dict(Counter(r["label"] for r in seeds)))

    print("loading:", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    first_device = next(model.parameters()).device

    outputs = []

    for i, row in enumerate(seeds, 1):
        prompt = make_prompt(row)
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rewritten = parse_json_text(gen)

        if rewritten is None or bad_rewrite(rewritten):
            rewritten = row["text"]
            status = "fallback_original"
        else:
            status = "qwen_rewrite"

        new = dict(row)
        new["source_case_id"] = row.get("case_id")
        new["case_id"] = f"hard_ood_v2_qwen_{len(outputs)+1:04d}"
        new["text"] = clean(rewritten)
        new["generation_status"] = status
        new["generator_model"] = MODEL_NAME

        outputs.append(new)

        if i <= 10 or i % 25 == 0:
            print(i, row["label"], status, clean(rewritten)[:220], flush=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(outputs),
        "label_counts": dict(Counter(r["label"] for r in outputs)),
        "generation_status_counts": dict(Counter(r["generation_status"] for r in outputs)),
        "source": str(SEEDS_PATH),
        "path": str(OUT_PATH),
        "generator_model": MODEL_NAME,
        "important_note": "Qwen rewrites text only. Gold labels are inherited from deterministic source cases.",
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
