import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("./ProiectD2/memory_gov_data")
RESULTS = ROOT / "processed/external_baseline_results/hard_ood_v2b_lifecycle_v5_runtime_v4_9/results.jsonl"
SUMMARY = ROOT / "processed/external_baseline_results/hard_ood_v2b_lifecycle_v5_runtime_v4_9/summary.json"

def bool_check_fraction(checks):
    if not isinstance(checks, dict) or not checks:
        return None, 0, 0

    vals = []
    for k, v in checks.items():
        if isinstance(v, bool):
            vals.append(v)

    if not vals:
        return None, 0, 0

    true_count = sum(1 for v in vals if v)
    total = len(vals)
    return true_count / total, true_count, total

rows = []
with RESULTS.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

print(f"Loaded rows: {len(rows)}")
print(f"Results file: {RESULTS}")

by_system = defaultdict(list)
mismatches = []
examples = []

for r in rows:
    system = r.get("system", "unknown")
    score = r.get("score")
    by_system[system].append(score)

    frac, true_count, total = bool_check_fraction(r.get("checks", {}))

    if frac is not None and isinstance(score, (int, float)):
        if abs(frac - score) > 1e-9:
            mismatches.append({
                "case_id": r.get("case_id"),
                "system": system,
                "category": r.get("category"),
                "score": score,
                "check_fraction": frac,
                "checks": r.get("checks"),
                "aux": r.get("aux"),
            })

    if len(examples) < 12:
        examples.append({
            "case_id": r.get("case_id"),
            "system": system,
            "category": r.get("category"),
            "expected_operation": r.get("expected_operation"),
            "observed_operation": (r.get("aux") or {}).get("observed_operation"),
            "score": score,
            "check_fraction": frac,
            "true_checks": true_count,
            "total_checks": total,
            "checks": r.get("checks"),
        })

print("\n=== Mean score computed from results.jsonl ===")
for system, scores in sorted(by_system.items()):
    valid = [s for s in scores if isinstance(s, (int, float))]
    mean = sum(valid) / len(valid) if valid else None
    print(f"{system}: cases={len(valid)} mean_score={mean}")

print("\n=== Summary file ===")
if SUMMARY.exists():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            print(json.dumps({
                "system": item.get("system"),
                "cases": item.get("cases"),
                "scored_cases": item.get("scored_cases"),
                "mean_score": item.get("mean_score"),
                "perfect_cases": item.get("perfect_cases"),
                "operation_match_rate": item.get("operation_match_rate"),
            }, ensure_ascii=False))
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:3000])
else:
    print("summary.json not found")

print("\n=== Check fraction vs stored score ===")
print(f"Rows where check_fraction != score: {len(mismatches)}")
if mismatches:
    print("First mismatches:")
    for m in mismatches[:10]:
        print(json.dumps(m, ensure_ascii=False, indent=2))
else:
    print("No mismatches found for rows with boolean checks.")

print("\n=== Example rows ===")
for e in examples:
    print(json.dumps(e, ensure_ascii=False, indent=2))
