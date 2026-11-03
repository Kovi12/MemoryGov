import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


TRAIN_DIR = Path("processed/memorygovbench_operation_v1_3_2_no_confirmation_semantic_model_ready_noleak")
BENCH_PATH = Path("processed/external_baselines_tier2_v2_250/external_tier2_cases_v4_5_no_confirmation.jsonl")
OUT_DIR = Path("processed/external_baseline_results/tier2_v1/stress225_embedding_overlap_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def clean(s):
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def train_text(row):
    return clean(row.get("text") or row.get("input") or row.get("prompt") or "")


def bench_text(row):
    parts = []
    for key in ["case_id", "category", "query"]:
        if row.get(key):
            parts.append(str(row[key]))

    for turn in row.get("turns", []):
        if isinstance(turn, dict):
            for key in ["user_message", "message", "candidate"]:
                if turn.get(key):
                    parts.append(str(turn[key]))

    for key in ["prior_memories", "expected_active_contains", "expected_retrieval_contains", "expected_retrieval_not_contains"]:
        v = row.get(key)
        if isinstance(v, list):
            parts.extend(map(str, v))
        elif v:
            parts.append(str(v))

    return clean("\n".join(parts))


train_rows = []
for p in TRAIN_DIR.glob("*.jsonl"):
    train_rows.extend(read_jsonl(p))

bench_rows = read_jsonl(BENCH_PATH)

train_texts = [train_text(r) for r in train_rows]
bench_texts = [bench_text(r) for r in bench_rows]

print("train rows:", len(train_texts))
print("bench rows:", len(bench_texts))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

train_emb = model.encode(train_texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
bench_emb = model.encode(bench_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

# cosine because normalized
sims = bench_emb @ train_emb.T

top = []
for i, row in enumerate(bench_rows):
    j = int(np.argmax(sims[i]))
    score = float(sims[i, j])
    top.append({
        "case_id": row.get("case_id"),
        "category": row.get("category"),
        "best_train_similarity": score,
        "best_train_index": j,
        "bench_text": bench_texts[i][:1000],
        "best_train_text": train_texts[j][:1000],
    })

top_sorted = sorted(top, key=lambda x: x["best_train_similarity"], reverse=True)

summary = {
    "train_rows": len(train_texts),
    "bench_rows": len(bench_texts),
    "mean_best_similarity": float(np.mean([x["best_train_similarity"] for x in top])),
    "max_best_similarity": float(max(x["best_train_similarity"] for x in top)),
    "cases_over_0_80": sum(x["best_train_similarity"] >= 0.80 for x in top),
    "cases_over_0_85": sum(x["best_train_similarity"] >= 0.85 for x in top),
    "cases_over_0_90": sum(x["best_train_similarity"] >= 0.90 for x in top),
}

(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT_DIR / "top_similar_cases.json").write_text(json.dumps(top_sorted[:50], indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("top cases:", OUT_DIR / "top_similar_cases.json")
