import json
import re
from pathlib import Path
from collections import Counter, defaultdict


TRAIN_DIR = Path("processed/memorygovbench_operation_v1_3_2_no_confirmation_semantic_model_ready_noleak")
STRESS_PATH = Path("processed/external_baselines_tier2_v2_250/external_tier2_cases_v4_5_no_confirmation.jsonl")
OUT_DIR = Path("processed/external_baseline_results/tier2_v1/stress225_complexity_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s):
    return [t for t in norm(s).split() if len(t) >= 3]


def shingles(tok, n=5):
    return set(tuple(tok[i:i+n]) for i in range(max(0, len(tok)-n+1)))


def row_text_train(r):
    return str(r.get("text") or r.get("input") or r.get("prompt") or "")


def row_text_case(r):
    parts = []
    for key in ["case_id", "category", "query"]:
        if key in r:
            parts.append(str(r[key]))
    for turn in r.get("turns", []):
        if isinstance(turn, dict):
            parts.append(str(turn.get("user_message", "")))
            parts.append(str(turn.get("message", "")))
            parts.append(str(turn.get("candidate", "")))
    for key in ["prior_memories", "expected_active_contains", "expected_retrieval_contains"]:
        v = r.get(key)
        if isinstance(v, list):
            parts.extend(map(str, v))
        elif v:
            parts.append(str(v))
    return "\n".join(parts)


train_rows = []
for p in TRAIN_DIR.glob("*.jsonl"):
    train_rows.extend(read_jsonl(p))

stress_rows = read_jsonl(STRESS_PATH)

train_texts = [row_text_train(r) for r in train_rows]
stress_texts = [row_text_case(r) for r in stress_rows]

train_shingles = set()
for txt in train_texts:
    train_shingles |= shingles(tokens(txt), 5)

trigger_words = {
    "write": ["remember", "store", "save", "keep"],
    "ignore": ["temporary", "today", "currently", "right now", "this session", "scratch"],
    "update": ["no longer", "instead", "changed", "switched", "now use", "not use"],
    "merge": ["also", "include", "includes", "additionally", "in addition"],
    "forget": ["forget", "delete", "remove", "erase"],
    "safety": ["password", "token", "api key", "passport", "bank", "secret", "private key"],
}

summary = {
    "train_rows": len(train_rows),
    "stress_cases": len(stress_rows),
    "category_counts": Counter(r.get("category") for r in stress_rows),
    "case_overlap": [],
    "trigger_counts_by_category": defaultdict(Counter),
}

for r, txt in zip(stress_rows, stress_texts):
    tok = tokens(txt)
    sh = shingles(tok, 5)
    overlap = len(sh & train_shingles)
    denom = max(1, len(sh))
    overlap_ratio = overlap / denom

    cat = r.get("category")

    lower = txt.lower()
    for group, pats in trigger_words.items():
        for pat in pats:
            if pat in lower:
                summary["trigger_counts_by_category"][cat][group] += 1
                break

    summary["case_overlap"].append({
        "case_id": r.get("case_id"),
        "category": cat,
        "tokens": len(tok),
        "fivegram_count": len(sh),
        "fivegram_train_overlap": overlap,
        "fivegram_train_overlap_ratio": overlap_ratio,
    })

summary["category_counts"] = dict(summary["category_counts"])
summary["trigger_counts_by_category"] = {
    k: dict(v) for k, v in summary["trigger_counts_by_category"].items()
}

ratios = [x["fivegram_train_overlap_ratio"] for x in summary["case_overlap"]]
summary["overlap_stats"] = {
    "mean": sum(ratios) / len(ratios),
    "max": max(ratios),
    "cases_over_0_10": sum(x > 0.10 for x in ratios),
    "cases_over_0_25": sum(x > 0.25 for x in ratios),
}

high = sorted(
    summary["case_overlap"],
    key=lambda x: x["fivegram_train_overlap_ratio"],
    reverse=True
)[:30]

(Path(OUT_DIR) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(Path(OUT_DIR) / "highest_overlap_cases.json").write_text(json.dumps(high, indent=2), encoding="utf-8")

print(json.dumps({
    "train_rows": summary["train_rows"],
    "stress_cases": summary["stress_cases"],
    "category_counts": summary["category_counts"],
    "overlap_stats": summary["overlap_stats"],
    "trigger_counts_by_category": summary["trigger_counts_by_category"],
    "highest_overlap_path": str(OUT_DIR / "highest_overlap_cases.json"),
}, indent=2))
