import json
from pathlib import Path
from collections import Counter

SRC = Path("processed/memorygovbench_operation_hard_ood_v2b_qwen_schema/operation_hard_ood_v2b_qwen_schema.jsonl")
OUT_DIR = Path("processed/external_baselines_hard_ood_v2b_qwen_schema")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "external_hard_ood_v2b_cases.jsonl"
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

def contains_for(row):
    label = row["label"]
    existing = clean(row.get("existing_memory"))
    candidate = clean(row.get("candidate_memory"))

    if label == "WRITE":
        return [candidate] if candidate else []
    if label == "UPDATE":
        return [candidate] if candidate else []
    if label == "MERGE":
        out = []
        if existing:
            out.append(existing)
        if candidate:
            out.append(candidate)
        return out
    return []

def not_contains_for(row):
    label = row["label"]
    existing = clean(row.get("existing_memory"))
    candidate = clean(row.get("candidate_memory"))

    if label in {"IGNORE", "QUARANTINE"}:
        return [candidate] if candidate else []
    if label == "FORGET":
        return [existing] if existing else []
    if label == "UPDATE":
        return [existing] if existing else []
    return []

def map_category(label):
    return {
        "WRITE": "write",
        "IGNORE": "ignore",
        "UPDATE": "update",
        "MERGE": "merge",
        "FORGET": "forget",
        "QUARANTINE": "safety",
    }[label]

def map_risk(label):
    if label == "QUARANTINE":
        return "high"
    if label == "FORGET":
        return "medium"
    return "low"

rows = read_jsonl(SRC)
out_rows = []

for i, r in enumerate(rows, start=1):
    label = r["label"]
    existing = clean(r.get("existing_memory"))
    user_msg = clean(r.get("user_message"))

    case = {
        "case_id": f"hardood_v2b_{i:03d}_{label.lower()}",
        "category": map_category(label),
        "risk": map_risk(label),
        "prior_memories": [existing] if existing else [],
        "turn": user_msg,
        "expected_operation": label,
        "expected_retrieval_contains": contains_for(r),

        # Extra fields for richer/custom scoring; existing evaluator should ignore
        # unknown fields if it is written normally.
        "expected_retrieval_not_contains": not_contains_for(r),
        "source_case_id": r.get("case_id"),
        "source_category": r.get("category"),
        "candidate_memory": clean(r.get("candidate_memory")),
        "existing_memory": existing,
        "benchmark": "hard_ood_v2b_qwen_schema",
    }
    out_rows.append(case)

with OUT.open("w", encoding="utf-8") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

summary = {
    "rows": len(out_rows),
    "label_counts": dict(Counter(r["expected_operation"] for r in out_rows)),
    "category_counts": dict(Counter(r["category"] for r in out_rows)),
    "source": str(SRC),
    "path": str(OUT),
    "note": "Behavioral external-baseline cases converted from Hard-OOD v2b schema-preserved operation benchmark.",
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
