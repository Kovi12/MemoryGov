import argparse
import ast
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path("external_baselines").resolve()))
from run_external_tier2_eval_v1 import MemoryGovAdapter

sys.path.insert(0, str(Path("public_benchmarks").resolve()))
from eval_memorygov_locomo_qa_subset_v4_structured import (
    build_structured_memories,
    root_key,
)

DEFAULT_CASES = Path("public_benchmarks/processed/locomo/locomo_qa_audited_nonoverlap.jsonl")
DEFAULT_OUT_DIR = Path("public_benchmarks/results/locomo_qa_memorygov_subset_v5_evidence")


def strip_urls(s):
    return re.sub(r"https?://\S+|www\.\S+", " ", str(s or ""))


def norm(s):
    return " ".join(strip_urls(s).lower().split())


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
    joined = "\n".join(strip_urls(x) for x in texts)
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


def parse_evidence(x):
    def split_one(v):
        s = str(v or "").strip()
        if not s:
            return []
        out = []
        for part in re.split(r"[,;]", s):
            part = part.strip()
            if part:
                out.append(part)
        return out

    if x is None:
        return []

    if isinstance(x, list):
        out = []
        for v in x:
            out.extend(split_one(v))
        return out

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            y = ast.literal_eval(s)
            if isinstance(y, list):
                out = []
                for v in y:
                    out.extend(split_one(v))
                return out
        except Exception:
            pass
        return split_one(s)

    return split_one(x)


def evidence_hits(texts, evidence_ids):
    joined = "\n".join(str(x or "") for x in texts).lower()
    hits = []
    for ev in evidence_ids:
        evs = str(ev).strip()
        if not evs:
            continue
        hits.append(evs.lower() in joined)
    return hits


def evaluate_case(adapter, case, memories):
    question = case["question"]
    answer = case["answer"]
    evidence_ids = parse_evidence(case.get("evidence"))

    retrieved = adapter.retrieve(question)
    retrieved_texts = [
        r.get("text", "") if isinstance(r, dict) else str(r)
        for r in retrieved
    ]

    ev_hits = evidence_hits(retrieved_texts, evidence_ids)
    has_evidence = len(evidence_ids) > 0

    answer_in_seed = contains_answer(memories, answer)
    answer_in_retrieved = contains_answer(retrieved_texts, answer)

    return {
        "record_id": case["record_id"],
        "source_file": case["source_file"],
        "root_key": root_key(case),
        "question": question,
        "gold_answer": answer,
        "category": case.get("category"),
        "evidence": evidence_ids,
        "has_evidence": has_evidence,
        "overlap_ratio": case.get("overlap_ratio"),
        "seed_memory_count": len(memories),
        "retrieved_count": len(retrieved_texts),

        "evidence_hit_any": any(ev_hits) if has_evidence else None,
        "evidence_hit_all": all(ev_hits) if has_evidence else None,
        "evidence_hit_fraction": (sum(ev_hits) / len(ev_hits)) if has_evidence else None,

        "answer_in_seed": answer_in_seed,
        "answer_in_retrieved": answer_in_retrieved,

        "retrieved_preview": retrieved_texts[:5],
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows):
    evidence_rows = [r for r in rows if r["has_evidence"]]
    no_evidence_rows = [r for r in rows if not r["has_evidence"]]

    by_cat = defaultdict(list)
    by_cat_all = defaultdict(list)
    by_cat_answer = defaultdict(list)

    for r in rows:
        cat = str(r.get("category"))
        if r["has_evidence"]:
            by_cat[cat].append(1.0 if r["evidence_hit_any"] else 0.0)
            by_cat_all[cat].append(1.0 if r["evidence_hit_all"] else 0.0)
        by_cat_answer[cat].append(1.0 if r["answer_in_retrieved"] else 0.0)

    return {
        "benchmark": "LoCoMo",
        "subset": "qa_audited_direct_nonoverlap",
        "system": "MemoryGov",
        "evaluator": "v5_evidence",
        "primary_metric": "evidence_hit_any",
        "secondary_metrics": [
            "evidence_hit_all",
            "evidence_hit_fraction",
            "retrieval_contains_gold_answer"
        ],
        "cases": len(rows),
        "evidence_cases": len(evidence_rows),
        "no_evidence_cases": len(no_evidence_rows),

        "evidence_hit_any_rate": mean([1.0 if r["evidence_hit_any"] else 0.0 for r in evidence_rows]),
        "evidence_hit_all_rate": mean([1.0 if r["evidence_hit_all"] else 0.0 for r in evidence_rows]),
        "mean_evidence_hit_fraction": mean([r["evidence_hit_fraction"] for r in evidence_rows]),

        "answer_in_seed_rate": mean([1.0 if r["answer_in_seed"] else 0.0 for r in rows]),
        "answer_in_retrieved_rate": mean([1.0 if r["answer_in_retrieved"] else 0.0 for r in rows]),

        "category_evidence_hit_any_rate": {
            k: sum(v) / len(v)
            for k, v in sorted(by_cat.items())
        },
        "category_evidence_hit_all_rate": {
            k: sum(v) / len(v)
            for k, v in sorted(by_cat_all.items())
        },
        "category_answer_in_retrieved_rate": {
            k: sum(v) / len(v)
            for k, v in sorted(by_cat_answer.items())
        },

        "important_note": (
            "This is a partial source-family non-independent sanity-check score on the audited direct-nonoverlap LoCoMo QA subset. "
            "The primary metric checks whether MemoryGov retrieval returns the gold evidence dialog IDs. "
            "It is not directly equivalent to published full-benchmark LoCoMo answer-accuracy or LLM-as-judge scores."
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
                f"evidence_any={row['evidence_hit_any']} "
                f"evidence_all={row['evidence_hit_all']} "
                f"answer={row['answer_in_retrieved']} "
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
