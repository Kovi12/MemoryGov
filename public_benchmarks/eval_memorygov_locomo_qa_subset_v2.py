import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path("external_baselines").resolve()))
from run_external_tier2_eval_v1 import MemoryGovAdapter


DEFAULT_CASES = Path("public_benchmarks/processed/locomo/locomo_qa_audited_nonoverlap.jsonl")
DEFAULT_OUT_DIR = Path("public_benchmarks/results/locomo_qa_memorygov_subset_v2")


def norm(s):
    return " ".join(str(s or "").lower().split())


def toks(s):
    return re.findall(r"[a-z0-9_./:@-]+", norm(s))


def answer_parts(answer):
    a = str(answer or "").strip()
    if not a:
        return []

    parts = re.split(r"\s*,\s*|\s*;\s*|\s+\band\b\s+", a, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [a]


def contains_answer(texts, answer):
    joined = "\n".join(str(x or "") for x in texts)
    joined_n = norm(joined)

    parts = answer_parts(answer)
    if not parts:
        return False

    hits = 0
    for p in parts:
        p_n = norm(p)
        if not p_n:
            continue

        if p_n in joined_n:
            hits += 1
            continue

        p_toks = set(toks(p))
        j_toks = set(toks(joined))
        if p_toks and len(p_toks & j_toks) / max(1, len(p_toks)) >= 0.80:
            hits += 1

    if len(parts) <= 3:
        return hits == len(parts)
    return hits / len(parts) >= 0.75


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_root_object(source_file, record_id):
    data = load_json(source_file)

    m = re.search(r"\.json\[(\d+)\]", record_id)
    if m and isinstance(data, list):
        idx = int(m.group(1))
        if 0 <= idx < len(data):
            return data[idx]

    return data


def collect_texts(obj, path="", skip_qa=True):
    out = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()

            if skip_qa and kl == "qa":
                continue

            if kl in {"embedding", "vector"}:
                continue

            child_path = f"{path}.{k}" if path else str(k)
            out.extend(collect_texts(v, child_path, skip_qa=skip_qa))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_path = f"{path}[{i}]"
            out.extend(collect_texts(v, child_path, skip_qa=skip_qa))

    else:
        text = str(obj).strip()
        if text:
            out.append((path, text))

    return out


def chunk_text(text, max_chars=900, overlap=120):
    text = " ".join(str(text or "").split())
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def build_memories_for_case(case, max_memories=0):
    root = get_root_object(case["source_file"], case["record_id"])
    extracted = collect_texts(root, skip_qa=True)

    memories = []
    for path, text in extracted:
        for chunk in chunk_text(text):
            memories.append(f"{path}: {chunk}")

    # Add non-QA structured summaries as broader context if present.
    # This catches compact event/observation fields while still excluding QA answers.
    if isinstance(root, dict):
        for key in ["session_summary", "event_summary", "observation"]:
            if key in root:
                flat = json.dumps(root[key], ensure_ascii=False)
                for chunk in chunk_text(flat, max_chars=1200, overlap=150):
                    memories.append(f"{key}: {chunk}")

    seen = set()
    out = []
    for m in memories:
        m = " ".join(str(m).split())
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)

    if max_memories and len(out) > max_memories:
        out = out[:max_memories]

    return out


def evaluate_case(adapter, case, max_memories):
    adapter.reset(case["record_id"])

    memories = build_memories_for_case(case, max_memories=max_memories)
    adapter.seed(memories)

    question = case["question"]
    answer = case["answer"]

    retrieved = adapter.retrieve(question)
    retrieved_texts = [r.get("text", "") if isinstance(r, dict) else str(r) for r in retrieved]

    answer_in_seed = contains_answer(memories, answer)
    answer_in_retrieved = contains_answer(retrieved_texts, answer)

    return {
        "record_id": case["record_id"],
        "source_file": case["source_file"],
        "question": question,
        "gold_answer": answer,
        "category": case.get("category"),
        "evidence": case.get("evidence"),
        "overlap_ratio": case.get("overlap_ratio"),
        "seed_memory_count": len(memories),
        "retrieved_count": len(retrieved_texts),
        "answer_in_seed": answer_in_seed,
        "answer_in_retrieved": answer_in_retrieved,
        "retrieved_preview": retrieved_texts[:5],
        "score": 1.0 if answer_in_retrieved else 0.0,
    }


def summarize(rows):
    n = len(rows)
    overall = sum(r["score"] for r in rows) / n if n else None

    seed_rows = [r for r in rows if r["answer_in_seed"]]
    conditional = sum(r["score"] for r in seed_rows) / len(seed_rows) if seed_rows else None

    by_cat = defaultdict(list)
    by_cat_seed = defaultdict(list)

    for r in rows:
        cat = str(r.get("category"))
        by_cat[cat].append(r["score"])
        if r["answer_in_seed"]:
            by_cat_seed[cat].append(r["score"])

    return {
        "benchmark": "LoCoMo",
        "subset": "qa_audited_direct_nonoverlap",
        "system": "MemoryGov",
        "metric": "retrieval_contains_gold_answer",
        "cases": n,
        "mean_score": overall,
        "answer_in_seed_rate": sum(1 for r in rows if r["answer_in_seed"]) / n if n else None,
        "answer_in_retrieved_rate": sum(1 for r in rows if r["answer_in_retrieved"]) / n if n else None,
        "conditional_score_when_answer_in_seed": conditional,
        "conditional_cases": len(seed_rows),
        "category_mean_score": {
            k: sum(v) / len(v)
            for k, v in sorted(by_cat.items())
        },
        "category_conditional_score_when_answer_in_seed": {
            k: sum(v) / len(v)
            for k, v in sorted(by_cat_seed.items())
        },
        "important_note": (
            "This is a partial source-family non-independent sanity-check score on the audited direct-nonoverlap LoCoMo QA subset. "
            "It is not directly equivalent to published full-benchmark LoCoMo scores."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_memories", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--margin_threshold", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        json.loads(x)
        for x in Path(args.cases).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    if args.limit:
        cases = cases[:args.limit]

    adapter = MemoryGovAdapter(
        db_root=out_dir / "memorygov_dbs",
        margin_threshold=args.margin_threshold,
    )

    rows = []
    for i, case in enumerate(cases, start=1):
        row = evaluate_case(adapter, case, max_memories=args.max_memories)
        rows.append(row)
        print(
            f"{i}/{len(cases)} {row['record_id']} "
            f"score={row['score']} answer_in_seed={row['answer_in_seed']} "
            f"seed_memories={row['seed_memory_count']} retrieved={row['retrieved_count']}"
        )

    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    with results_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
