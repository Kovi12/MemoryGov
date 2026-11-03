import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path("external_baselines").resolve()))
from run_external_tier2_eval_v1 import MemoryGovAdapter

sys.path.insert(0, str(Path("public_benchmarks").resolve()))
from eval_memorygov_locomo_qa_subset_v2 import contains_answer


DEFAULT_CASES = Path("public_benchmarks/processed/locomo/locomo_qa_audited_nonoverlap.jsonl")
DEFAULT_OUT_DIR = Path("public_benchmarks/results/locomo_qa_memorygov_subset_v4_structured")


def root_key(case):
    rid = case["record_id"]
    m = re.search(r"(.+?\.json)(\[\d+\])", rid)
    if m:
        return m.group(1) + m.group(2)
    return case["source_file"]


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


def compact(obj, max_chars=1600):
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in {"embedding", "vector", "qa"}:
                continue
            if isinstance(v, (dict, list)):
                parts.append(f"{k}: {compact(v, max_chars=max_chars)}")
            else:
                parts.append(f"{k}: {v}")
        s = " ".join(parts)
    elif isinstance(obj, list):
        s = " ".join(compact(x, max_chars=max_chars) for x in obj)
    else:
        s = str(obj)

    s = " ".join(s.split())
    if len(s) > max_chars:
        return s[:max_chars]
    return s


def chunk_text(text, max_chars=1200, overlap=150):
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


def add_units_from_value(memories, prefix, value):
    if value is None:
        return

    if isinstance(value, list):
        for i, item in enumerate(value):
            txt = compact(item)
            for j, ch in enumerate(chunk_text(txt)):
                memories.append(f"{prefix}[{i}]" + (f".chunk{j}" if len(txt) > 1200 else "") + f": {ch}")

    elif isinstance(value, dict):
        for k, v in value.items():
            txt = compact(v)
            for j, ch in enumerate(chunk_text(txt)):
                memories.append(f"{prefix}.{k}" + (f".chunk{j}" if len(txt) > 1200 else "") + f": {ch}")

    else:
        txt = compact(value)
        for j, ch in enumerate(chunk_text(txt)):
            memories.append(f"{prefix}" + (f".chunk{j}" if len(txt) > 1200 else "") + f": {ch}")


def build_structured_memories(case, max_memories=0):
    root = get_root_object(case["source_file"], case["record_id"])

    memories = []

    if isinstance(root, dict):
        # Priority: keep the original LoCoMo memory structures as coherent units.
        for key in ["conversation", "session_summary", "event_summary", "observation"]:
            if key in root:
                add_units_from_value(memories, key, root[key])

        # Include any other non-QA fields that may contain persona or metadata.
        for key, value in root.items():
            kl = str(key).lower()
            if kl in {"qa", "conversation", "session_summary", "event_summary", "observation", "embedding", "vector"}:
                continue
            add_units_from_value(memories, key, value)

    else:
        add_units_from_value(memories, "root", root)

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


def evaluate_case(adapter, case, memories):
    question = case["question"]
    answer = case["answer"]

    retrieved = adapter.retrieve(question)
    retrieved_texts = [
        r.get("text", "") if isinstance(r, dict) else str(r)
        for r in retrieved
    ]

    answer_in_seed = contains_answer(memories, answer)
    answer_in_retrieved = contains_answer(retrieved_texts, answer)

    return {
        "record_id": case["record_id"],
        "source_file": case["source_file"],
        "root_key": root_key(case),
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
    seed_rows = [r for r in rows if r.get("answer_in_seed")]

    by_cat = defaultdict(list)
    by_cat_seed = defaultdict(list)

    for r in rows:
        cat = str(r.get("category"))
        by_cat[cat].append(r["score"])
        if r.get("answer_in_seed"):
            by_cat_seed[cat].append(r["score"])

    return {
        "benchmark": "LoCoMo",
        "subset": "qa_audited_direct_nonoverlap",
        "system": "MemoryGov",
        "evaluator": "v4_structured",
        "metric": "retrieval_contains_gold_answer",
        "cases": n,
        "mean_score": sum(r["score"] for r in rows) / n if n else None,
        "answer_in_seed_rate": sum(1 for r in rows if r.get("answer_in_seed")) / n if n else None,
        "answer_in_retrieved_rate": sum(1 for r in rows if r["answer_in_retrieved"]) / n if n else None,
        "conditional_score_when_answer_in_seed": (
            sum(r["score"] for r in seed_rows) / len(seed_rows) if seed_rows else None
        ),
        "conditional_cases": len(seed_rows),
        "groups": len(set(r["root_key"] for r in rows)),
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
            "Published LoCoMo scores are full-benchmark scores, while this is a subset retrieval-grounded score."
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

    groups = defaultdict(list)
    for c in cases:
        groups[root_key(c)].append(c)

    adapter = MemoryGovAdapter(
        db_root=out_dir / "memorygov_dbs",
        margin_threshold=args.margin_threshold,
    )

    rows = []

    for gi, (gkey, group_cases) in enumerate(sorted(groups.items()), start=1):
        print(f"\n=== Group {gi}/{len(groups)}: {gkey}, cases={len(group_cases)} ===")

        adapter.reset(gkey)

        memories = build_structured_memories(group_cases[0], max_memories=args.max_memories)
        print(f"Seeding group once: seed_memories={len(memories)}")
        adapter.seed(memories)

        for i, case in enumerate(group_cases, start=1):
            row = evaluate_case(adapter, case, memories)
            rows.append(row)

            print(
                f"{i}/{len(group_cases)} {row['record_id']} "
                f"score={row['score']} answer_in_seed={row['answer_in_seed']} "
                f"retrieved={row['retrieved_count']}"
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
