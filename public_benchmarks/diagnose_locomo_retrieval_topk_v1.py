import json
import math
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path("external_baselines").resolve()))
from run_external_tier2_eval_v1 import MemoryGovAdapter

sys.path.insert(0, str(Path("public_benchmarks").resolve()))
from eval_memorygov_locomo_qa_subset_v4_structured import (
    build_structured_memories,
    root_key,
)
from eval_memorygov_locomo_qa_subset_v5_evidence import parse_evidence, evidence_hits

CASES = Path("public_benchmarks/processed/locomo/locomo_qa_audited_nonoverlap.jsonl")
OUT_DIR = Path("public_benchmarks/results/locomo_qa_retrieval_topk_diagnostics_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPKS = [1, 3, 5, 10, 20, 50, 100]

STOP = {
    "what", "when", "where", "who", "would", "did", "does", "do", "is", "are",
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "be",
    "considered", "caroline", "melanie"
}

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def toks(s):
    return [
        t for t in re.findall(r"[a-z0-9_./:@-]+", str(s or "").lower())
        if len(t) >= 3 and t not in STOP
    ]

def lexical_score(query, text):
    q = toks(query)
    if not q:
        return 0.0
    tf = Counter(toks(text))
    score = 0.0
    for t in q:
        if t in tf:
            score += 1.0 + math.log(1 + tf[t])
    return score / max(1, len(set(q)))

def rank_lexical(query, texts):
    rows = []
    for i, text in enumerate(texts):
        rows.append((lexical_score(query, text), i, text))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [x[2] for x in rows]

def hit_any(texts, evidence):
    hits = evidence_hits(texts, evidence)
    return any(hits) if evidence else None

def main():
    cases = read_jsonl(CASES)[:10]

    groups = defaultdict(list)
    for c in cases:
        groups[root_key(c)].append(c)

    adapter = MemoryGovAdapter(
        db_root=OUT_DIR / "memorygov_dbs",
        margin_threshold=0.10,
    )

    rows = []

    for gkey, group_cases in sorted(groups.items()):
        print(f"\n=== {gkey} cases={len(group_cases)} ===")
        adapter.reset(gkey)

        memories = build_structured_memories(group_cases[0])
        adapter.seed(memories)

        state = adapter.export_state()
        active_texts = state.get("active_texts", []) or []

        print("active_texts:", len(active_texts))

        for c in group_cases:
            evidence = parse_evidence(c.get("evidence"))
            if not evidence:
                continue

            ranked = rank_lexical(c["question"], active_texts)

            row = {
                "record_id": c["record_id"],
                "question": c["question"],
                "answer": c["answer"],
                "evidence": evidence,
                "active_evidence_any": hit_any(active_texts, evidence),
                "topk_hits": {},
            }

            for k in TOPKS:
                row["topk_hits"][str(k)] = hit_any(ranked[:k], evidence)

            rows.append(row)
            print(c["record_id"], "active_any=", row["active_evidence_any"], "topk=", row["topk_hits"])

    with (OUT_DIR / "topk_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "cases": len(cases),
        "evidence_cases_evaluated": len(rows),
        "active_evidence_any_rate": sum(1 for r in rows if r["active_evidence_any"]) / len(rows) if rows else None,
        "lexical_topk_evidence_any_rate": {
            str(k): sum(1 for r in rows if r["topk_hits"][str(k)]) / len(rows) if rows else None
            for k in TOPKS
        },
        "note": "Diagnostic only. Tests whether active MemoryGov texts contain gold evidence under a simple lexical top-k reranker.",
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
