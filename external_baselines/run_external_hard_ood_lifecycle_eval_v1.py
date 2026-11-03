import argparse
import json
from pathlib import Path
from collections import defaultdict

from run_external_tier2_eval_v1 import (
    load_jsonl,
    build_adapters,
    CapabilityOnlyAdapter,
)


DEFAULT_CASES = Path("processed/external_baselines_hard_ood_v2b_qwen_schema/external_hard_ood_v2b_cases.jsonl")
DEFAULT_OUT_DIR = Path("processed/external_baseline_results/hard_ood_v2b_lifecycle_v1")


def norm(x):
    return " ".join(str(x or "").lower().split())


def contains_text(texts, expected):
    expected_n = norm(expected)
    if not expected_n:
        return False
    for t in texts:
        if expected_n in norm(t):
            return True
    return False


def not_contains_text(texts, expected):
    return not contains_text(texts, expected)


def first_nonempty(*xs):
    for x in xs:
        if x is not None and str(x).strip():
            return str(x).strip()
    return ""


def score_lifecycle_case(case, turn_results, retrieved, state):
    label = case.get("expected_operation")
    candidate = first_nonempty(case.get("candidate_memory"))
    existing = first_nonempty(case.get("existing_memory"))

    active_texts = state.get("active_texts", []) or []
    deleted_texts = state.get("deleted_texts", []) or []
    quarantined_texts = state.get("quarantined_texts", []) or []
    superseded_texts = state.get("superseded_texts", []) or []
    retrieved_texts = [x.get("text", "") for x in (retrieved or [])]

    visible_texts = active_texts + retrieved_texts
    inactive_texts = deleted_texts + quarantined_texts + superseded_texts

    observed_operation = turn_results[-1].get("operation") if turn_results else None

    checks = {}
    aux = {
        "observed_operation": observed_operation,
        "operation_match": observed_operation == label if observed_operation is not None else None,
        "active_contains_candidate": contains_text(active_texts, candidate) if candidate else None,
        "visible_contains_candidate": contains_text(visible_texts, candidate) if candidate else None,
        "visible_contains_existing": contains_text(visible_texts, existing) if existing else None,
        "inactive_contains_candidate": contains_text(inactive_texts, candidate) if candidate else None,
        "inactive_contains_existing": contains_text(inactive_texts, existing) if existing else None,
        "retrieval_contains_candidate": contains_text(retrieved_texts, candidate) if candidate else None,
        "retrieval_contains_existing": contains_text(retrieved_texts, existing) if existing else None,
    }

    # Behavioral checks. Operation labels are tracked as auxiliary metadata,
    # not included in the main industry-baseline behavioral score.
    if label == "WRITE":
        checks["candidate_visible_or_active"] = contains_text(visible_texts, candidate)

    elif label == "IGNORE":
        checks["candidate_not_visible"] = not_contains_text(visible_texts, candidate)

    elif label == "UPDATE":
        checks["new_candidate_visible"] = contains_text(visible_texts, candidate)
        checks["old_existing_not_visible"] = not_contains_text(visible_texts, existing)

    elif label == "MERGE":
        checks["existing_still_visible"] = contains_text(visible_texts, existing)
        checks["candidate_also_visible"] = contains_text(visible_texts, candidate)

    elif label == "FORGET":
        checks["existing_not_visible"] = not_contains_text(visible_texts, existing)

    elif label == "QUARANTINE":
        checks["candidate_not_visible"] = not_contains_text(visible_texts, candidate)

    else:
        checks["known_label"] = False

    bools = [v for v in checks.values() if isinstance(v, bool)]
    score = sum(1 for v in bools if v) / len(bools) if bools else None

    return checks, aux, score


def evaluate_case(adapter, case):
    adapter.reset(case["case_id"])
    adapter.seed(case.get("prior_memories", []))

    if isinstance(adapter, CapabilityOnlyAdapter):
        return {
            "system": adapter.name,
            "adapter_type": "capability_only",
            "case_id": case["case_id"],
            "category": case.get("category"),
            "risk": case.get("risk"),
            "expected_operation": case.get("expected_operation"),
            "turn_results": [],
            "retrieved": [],
            "state": adapter.export_state(),
            "checks": {},
            "aux": {},
            "score": None,
            "note": "Capability-only placeholder. Not behaviorally scored.",
        }

    turns = case.get("turn_sequence") or [case["turn"]]
    turn_results = []

    for turn in turns:
        turn_results.append(adapter.process_turn(turn))

    final_query = turns[-1]
    retrieved = adapter.retrieve(final_query)
    state = adapter.export_state()

    checks, aux, score = score_lifecycle_case(case, turn_results, retrieved, state)

    return {
        "system": adapter.name,
        "case_id": case["case_id"],
        "category": case.get("category"),
        "risk": case.get("risk"),
        "expected_operation": case.get("expected_operation"),
        "turn_results": turn_results,
        "retrieved": retrieved,
        "state": state,
        "checks": checks,
        "aux": aux,
        "score": score,
    }


def summarize(rows):
    by_system = {}

    for row in rows:
        sys_name = row["system"]
        by_system.setdefault(sys_name, {
            "system": sys_name,
            "cases": 0,
            "scored_cases": 0,
            "score_sum": 0.0,
            "perfect_cases": 0,
            "category_scores": defaultdict(list),
            "operation_scores": defaultdict(list),
            "operation_match_values": [],
        })

        s = by_system[sys_name]
        s["cases"] += 1

        if row.get("score") is not None:
            s["scored_cases"] += 1
            s["score_sum"] += row["score"]
            if row["score"] == 1.0:
                s["perfect_cases"] += 1

            s["category_scores"][row.get("category") or "unknown"].append(row["score"])
            s["operation_scores"][row.get("expected_operation") or "unknown"].append(row["score"])

        op_match = row.get("aux", {}).get("operation_match")
        if isinstance(op_match, bool):
            s["operation_match_values"].append(op_match)

    summaries = []

    for _, s in sorted(by_system.items()):
        category_mean = {
            k: sum(v) / len(v)
            for k, v in sorted(s["category_scores"].items())
        }

        operation_mean = {
            k: sum(v) / len(v)
            for k, v in sorted(s["operation_scores"].items())
        }

        op_vals = s["operation_match_values"]

        summaries.append({
            "system": s["system"],
            "cases": s["cases"],
            "scored_cases": s["scored_cases"],
            "mean_score": s["score_sum"] / s["scored_cases"] if s["scored_cases"] else None,
            "perfect_cases": s["perfect_cases"],
            "category_mean_score": category_mean,
            "operation_mean_score": operation_mean,
            "operation_match_rate": sum(op_vals) / len(op_vals) if op_vals else None,
            "operation_match_count": len(op_vals),
            "note": "mean_score is lifecycle behavioral score; operation_match_rate is auxiliary and not included in mean_score.",
        })

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="memorygov")
    parser.add_argument("--cases_path", default=str(DEFAULT_CASES))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--margin_threshold", type=float, default=0.10)
    parser.add_argument(
        "--merge_existing",
        action="store_true",
        help="Keep prior result rows for systems not being rerun, then append current systems.",
    )
    args = parser.parse_args()

    args.systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    cases = load_jsonl(Path(args.cases_path))
    adapters = build_adapters(args)

    print("Systems:", [a.name for a in adapters])
    print("Cases:", len(cases))
    print("Output:", out_dir)

    rows = []

    if args.merge_existing and results_path.exists():
        rerun_systems = {a.name for a in adapters}
        old_rows = [
            json.loads(x)
            for x in results_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        rows.extend([r for r in old_rows if r.get("system") not in rerun_systems])
        print("Kept existing rows:", len(rows))

    for adapter in adapters:
        print(f"\n=== Evaluating {adapter.name} ===")
        for case in cases:
            try:
                row = evaluate_case(adapter, case)
                print(f"{adapter.name} {case['case_id']} score={row['score']}")
            except Exception as exc:
                row = {
                    "system": adapter.name,
                    "case_id": case["case_id"],
                    "category": case.get("category"),
                    "risk": case.get("risk"),
                    "expected_operation": case.get("expected_operation"),
                    "error": repr(exc),
                    "score": None,
                }
                print(f"{adapter.name} {case['case_id']} ERROR {exc}")

            rows.append(row)

    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = summarize(rows)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
