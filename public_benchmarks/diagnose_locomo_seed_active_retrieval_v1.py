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
from eval_memorygov_locomo_qa_subset_v5_evidence import parse_evidence, evidence_hits

CASES = Path("public_benchmarks/processed/locomo/locomo_qa_audited_nonoverlap.jsonl")
OUT_DIR = Path("public_benchmarks/results/locomo_qa_diagnostics_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def hit_any(texts, evidence):
    hits = evidence_hits(texts, evidence)
    return any(hits) if evidence else None

def hit_all(texts, evidence):
    hits = evidence_hits(texts, evidence)
    return all(hits) if evidence else None

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
        print("raw seed memories:", len(memories))

        adapter.seed(memories)
        state = adapter.export_state()
        active_texts = state.get("active_texts", []) or []

        print("active memories:", len(active_texts))

        for c in group_cases:
            evidence = parse_evidence(c.get("evidence"))
            retrieved = adapter.retrieve(c["question"])
            retrieved_texts = [
                r.get("text", "") if isinstance(r, dict) else str(r)
                for r in retrieved
            ]

            row = {
                "record_id": c["record_id"],
                "question": c["question"],
                "answer": c["answer"],
                "category": c.get("category"),
                "evidence": evidence,
                "raw_seed_evidence_any": hit_any(memories, evidence),
                "raw_seed_evidence_all": hit_all(memories, evidence),
                "active_evidence_any": hit_any(active_texts, evidence),
                "active_evidence_all": hit_all(active_texts, evidence),
                "retrieved_evidence_any": hit_any(retrieved_texts, evidence),
                "retrieved_evidence_all": hit_all(retrieved_texts, evidence),
                "seed_memory_count": len(memories),
                "active_memory_count": len(active_texts),
                "retrieved_count": len(retrieved_texts),
                "retrieved_preview": retrieved_texts[:5],
            }
            rows.append(row)

            print(
                c["record_id"],
                "evidence=", evidence,
                "seed_any=", row["raw_seed_evidence_any"],
                "active_any=", row["active_evidence_any"],
                "retrieved_any=", row["retrieved_evidence_any"],
            )

    with (OUT_DIR / "diagnostics.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    evidence_rows = [r for r in rows if r["evidence"]]
    summary = {
        "cases": len(rows),
        "evidence_cases": len(evidence_rows),
        "raw_seed_evidence_any_rate": sum(1 for r in evidence_rows if r["raw_seed_evidence_any"]) / len(evidence_rows) if evidence_rows else None,
        "active_evidence_any_rate": sum(1 for r in evidence_rows if r["active_evidence_any"]) / len(evidence_rows) if evidence_rows else None,
        "retrieved_evidence_any_rate": sum(1 for r in evidence_rows if r["retrieved_evidence_any"]) / len(evidence_rows) if evidence_rows else None,
        "note": "Diagnostic only. Determines whether LoCoMo evidence IDs survive raw seed extraction, MemoryGov governance insertion, and retrieval top-5.",
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
