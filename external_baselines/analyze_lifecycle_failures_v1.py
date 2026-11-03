import json
from pathlib import Path
from collections import defaultdict

RESULTS = Path("processed/external_baseline_results/hard_ood_v2b_lifecycle_v3_free/results.jsonl")
OUT = Path("processed/external_baseline_results/hard_ood_v2b_lifecycle_v3_free/FAILURE_ANALYSIS.md")

rows = [json.loads(x) for x in RESULTS.read_text(encoding="utf-8").splitlines() if x.strip()]

by_system = defaultdict(list)
for r in rows:
    by_system[r["system"]].append(r)

lines = []
lines.append("# Lifecycle failure analysis")
lines.append("")
lines.append("This report lists non-perfect cases by system and operation.")
lines.append("")

for system in sorted(by_system):
    rs = by_system[system]
    bad = [r for r in rs if r.get("score") != 1.0]
    lines.append(f"## {system}")
    lines.append("")
    lines.append(f"- cases: {len(rs)}")
    lines.append(f"- non-perfect cases: {len(bad)}")
    lines.append("")

    by_op = defaultdict(list)
    for r in bad:
        by_op[r.get("expected_operation")].append(r)

    for op in sorted(by_op):
        lines.append(f"### {op} failures")
        lines.append("")
        for r in by_op[op][:20]:
            checks = r.get("checks", {})
            aux = r.get("aux", {})
            lines.append(f"- `{r.get('case_id')}` score={r.get('score')} observed={aux.get('observed_operation')}")
            lines.append(f"  - checks: `{json.dumps(checks, ensure_ascii=False)}`")
            pre = r.get("pre_retrieved_texts") or []
            post = r.get("post_retrieved_texts") or []
            if pre:
                lines.append(f"  - pre_retrieved[0]: `{str(pre[0])[:220]}`")
            if post:
                lines.append(f"  - post_retrieved[0]: `{str(post[0])[:220]}`")
        lines.append("")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
