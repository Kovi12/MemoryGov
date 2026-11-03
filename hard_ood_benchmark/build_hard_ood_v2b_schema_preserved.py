import json
from pathlib import Path
from collections import Counter

SRC = Path("processed/memorygovbench_operation_hard_ood_v2_qwen/operation_hard_ood_v2_qwen.jsonl")
OUT_DIR = Path("processed/memorygovbench_operation_hard_ood_v2b_qwen_schema")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "operation_hard_ood_v2b_qwen_schema.jsonl"
SUMMARY = OUT_DIR / "summary.json"

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def clean(x):
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"none", "null", "n/a"}:
        return ""
    return x

def build_text(row):
    existing = clean(row.get("existing_memory"))
    user_msg = clean(row.get("user_message"))
    candidate = clean(row.get("candidate_memory"))

    blocks = []

    if existing:
        blocks.append(f"Existing memory:\n{existing}")

    if user_msg:
        blocks.append(f"User message:\n{user_msg}")

    if candidate:
        blocks.append(f"Candidate memory:\n{candidate}")

    # If a row somehow has no structured fields, preserve original text.
    if not blocks:
        return clean(row.get("text"))

    return "\n\n".join(blocks)

rows = read_jsonl(SRC)
out_rows = []

for i, r in enumerate(rows):
    nr = dict(r)
    nr["case_id"] = f"hard_ood_v2b_qwen_schema_{i:04d}"
    nr["source_case_id"] = r.get("case_id")
    nr["text"] = build_text(r)
    nr["schema_preserved"] = True
    nr["note"] = (
        "Schema-preserved repair of qwen v2. Field contents are taken from structured "
        "row fields; block headers are reconstructed deterministically."
    )
    out_rows.append(nr)

with OUT.open("w", encoding="utf-8") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

summary = {
    "rows": len(out_rows),
    "label_counts": dict(Counter(r["label"] for r in out_rows)),
    "generation_status_counts": dict(Counter(r.get("generation_status", "NA") for r in out_rows)),
    "source": str(SRC),
    "path": str(OUT),
    "important_note": "v2b preserves benchmark schema by reconstructing fixed block headers from structured fields.",
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
