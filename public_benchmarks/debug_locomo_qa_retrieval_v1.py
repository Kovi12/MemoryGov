import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("public_benchmarks").resolve()))
from eval_memorygov_locomo_qa_subset_v4_structured import (
    build_structured_memories,
    contains_answer,
    root_key,
)

RESULTS = Path("public_benchmarks/results/locomo_qa_memorygov_subset_v4_structured/results.jsonl")
CASES = Path("public_benchmarks/processed/locomo/locomo_qa_audited_nonoverlap.jsonl")
OUT = Path("public_benchmarks/results/locomo_qa_memorygov_subset_v4_structured/debug_retrieval_misses.md")

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

cases = {r["record_id"]: r for r in read_jsonl(CASES)}
results = read_jsonl(RESULTS)

lines = []
lines.append("# LoCoMo QA retrieval debug - v4 structured")
lines.append("")
lines.append("This debug file compares gold answers, seed coverage, and retrieved previews for failed cases.")
lines.append("")

for r in results:
    if r.get("score") == 1.0:
        continue

    c = cases[r["record_id"]]
    memories = build_structured_memories(c)
    answer = r["gold_answer"]

    seed_hits = [m for m in memories if contains_answer([m], answer)]

    lines.append("## " + r["record_id"])
    lines.append("")
    lines.append(f"- Question: {r['question']}")
    lines.append(f"- Gold answer: {answer}")
    lines.append(f"- Category: {r.get('category')}")
    lines.append(f"- Evidence: {r.get('evidence')}")
    lines.append(f"- answer_in_seed: {r.get('answer_in_seed')}")
    lines.append(f"- seed_memory_count: {r.get('seed_memory_count')}")
    lines.append(f"- retrieved_count: {r.get('retrieved_count')}")
    lines.append("")
    lines.append("### Seed memories containing gold answer")
    if seed_hits:
        for h in seed_hits[:5]:
            lines.append("- " + h[:1000].replace("\n", " "))
    else:
        lines.append("- NONE")
    lines.append("")
    lines.append("### Retrieved preview")
    for item in r.get("retrieved_preview", []):
        lines.append("- " + str(item)[:1000].replace("\n", " "))
    lines.append("")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
