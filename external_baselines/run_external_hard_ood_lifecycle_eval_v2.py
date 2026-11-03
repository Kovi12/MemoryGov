import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

from run_external_tier2_eval_v1 import (
    load_jsonl,
    build_adapters,
    CapabilityOnlyAdapter,
)

DEFAULT_CASES = Path("processed/external_baselines_hard_ood_v2b_qwen_schema/external_hard_ood_v2b_cases.jsonl")
DEFAULT_OUT_DIR = Path("processed/external_baseline_results/hard_ood_v2b_lifecycle_v2")

STOP = {
    "the", "and", "for", "with", "that", "this", "user", "memory", "candidate",
    "active", "current", "previous", "previously", "stored", "message", "from",
    "about", "into", "their", "there", "should", "would", "could", "when",
    "what", "have", "has", "had", "are", "was", "were", "will", "not", "now",
}

def norm(x):
    return " ".join(str(x or "").lower().split())

def toks(x):
    raw = re.findall(r"[a-z0-9_./:@-]+", norm(x))
    return [t for t in raw if len(t) >= 3 and t not in STOP]

def soft_contains(texts, expected):
    expected = str(expected or "").strip()
    if not expected:
        return False

    expected_n = norm(expected)
    expected_tokens = toks(expected)

    for text in texts or []:
        text = str(text or "")
        text_n = norm(text)

        if expected_n and expected_n in text_n:
            return True

        text_tokens = set(toks(text))
        if not expected_tokens or not text_tokens:
            continue

        overlap = sum(1 for t in expected_tokens if t in text_tokens)
        ratio = overlap / max(1, len(set(expected_tokens)))

        if len(set(expected_tokens)) <= 4:
            if overlap >= max(1, len(set(expected_tokens)) - 1):
                return True
        else:
            if overlap >= 4 and ratio >= 0.45:
                return True

    return False

def soft_not_contains(texts, expected):
    return not soft_contains(texts, expected)

def discriminative_tokens(a, b):
    """Tokens that appear in a but not b. Useful for UPDATE old-vs-new checks."""
    ta = set(toks(a))
    tb = set(toks(b))
    return sorted(ta - tb)

def strict_old_not_visible_after(texts, existing, candidate):
    """
    For UPDATE, soft_not_contains can be fooled when old and new memories share most tokens.
    If old has discriminative tokens not present in the new candidate, require those tokens
    to be absent from post-visible texts.
    """
    existing = str(existing or "").strip()
    candidate = str(candidate or "").strip()
    if not existing:
        return True

    diff = discriminative_tokens(existing, candidate)
    useful = [t for t in diff if len(t) >= 3]

    if useful:
        joined = " ".join(str(x or "").lower() for x in texts or [])
        return not any(t.lower() in joined for t in useful)

    return soft_not_contains(texts, existing)

def first_nonempty(*xs):
    for x in xs:
        if x is not None and str(x).strip():
            return str(x).strip()
    return ""

def texts_from_state(state):
    return {
        "active": state.get("active_texts", []) or [],
        "deleted": state.get("deleted_texts", []) or [],
        "quarantined": state.get("quarantined_texts", []) or [],
        "superseded": state.get("superseded_texts", []) or [],
    }

def retrieve_texts(adapter, probes):
    out = []
    seen = set()

    for probe in probes:
        probe = str(probe or "").strip()
        if not probe:
            continue

        try:
            rows = adapter.retrieve(probe)
        except Exception:
            rows = []

        for r in rows or []:
            if isinstance(r, dict):
                txt = r.get("text") or r.get("memory") or r.get("content") or ""
            else:
                txt = str(r)
            if txt and txt not in seen:
                seen.add(txt)
                out.append(txt)

    return out

def score_lifecycle_case(case, turn_results, pre_state, pre_retrieved_texts, post_state, post_retrieved_texts):
    label = case.get("expected_operation")
    candidate = first_nonempty(case.get("candidate_memory"))
    existing = first_nonempty(case.get("existing_memory"))

    pre = texts_from_state(pre_state)
    post = texts_from_state(post_state)

    pre_visible = pre["active"] + pre_retrieved_texts
    post_visible = post["active"] + post_retrieved_texts
    post_inactive = post["deleted"] + post["quarantined"] + post["superseded"]

    observed_operation = turn_results[-1].get("operation") if turn_results else None

    aux = {
        "observed_operation": observed_operation,
        "operation_match": observed_operation == label if observed_operation is not None else None,
        "pre_existing_visible": soft_contains(pre_visible, existing) if existing else None,
        "post_existing_visible": soft_contains(post_visible, existing) if existing else None,
        "post_candidate_visible": soft_contains(post_visible, candidate) if candidate else None,
        "post_existing_inactive": soft_contains(post_inactive, existing) if existing else None,
        "post_candidate_inactive": soft_contains(post_inactive, candidate) if candidate else None,
    }

    checks = {}

    if label == "WRITE":
        checks["candidate_visible_after"] = soft_contains(post_visible, candidate)

    elif label == "IGNORE":
        checks["candidate_not_visible_after"] = soft_not_contains(post_visible, candidate)

    elif label == "UPDATE":
        checks["existing_visible_before"] = soft_contains(pre_visible, existing)
        checks["candidate_visible_after"] = soft_contains(post_visible, candidate)
        checks["existing_not_visible_after"] = strict_old_not_visible_after(post_visible, existing, candidate)

    elif label == "MERGE":
        checks["existing_visible_before"] = soft_contains(pre_visible, existing)
        checks["existing_visible_after"] = soft_contains(post_visible, existing)
        checks["candidate_visible_after"] = soft_contains(post_visible, candidate)

    elif label == "FORGET":
        checks["existing_visible_before"] = soft_contains(pre_visible, existing)
        checks["existing_not_visible_after"] = soft_not_contains(post_visible, existing)

    elif label == "QUARANTINE":
        checks["candidate_not_visible_after"] = soft_not_contains(post_visible, candidate)

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

    existing = first_nonempty(case.get("existing_memory"))
    candidate = first_nonempty(case.get("candidate_memory"))
    turn = case.get("turn") or ""

    pre_state = adapter.export_state()
    pre_retrieved_texts = retrieve_texts(adapter, [existing])

    turns = case.get("turn_sequence") or [turn]
    turn_results = []
    for t in turns:
        turn_results.append(adapter.process_turn(t))

    post_state = adapter.export_state()
    post_retrieved_texts = retrieve_texts(adapter, [candidate, existing, turn])

    checks, aux, score = score_lifecycle_case(
        case,
        turn_results,
        pre_state,
        pre_retrieved_texts,
        post_state,
        post_retrieved_texts,
    )

    return {
        "system": adapter.name,
        "case_id": case["case_id"],
        "category": case.get("category"),
        "risk": case.get("risk"),
        "expected_operation": case.get("expected_operation"),
        "turn_results": turn_results,
        "pre_state": pre_state,
        "post_state": post_state,
        "pre_retrieved_texts": pre_retrieved_texts,
        "post_retrieved_texts": post_retrieved_texts,
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
            "note": "mean_score is lifecycle behavioral score with precondition checks and soft matching; operation_match_rate is auxiliary.",
        })

    return summaries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="memorygov")
    parser.add_argument("--cases_path", default=str(DEFAULT_CASES))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--margin_threshold", type=float, default=0.10)
    parser.add_argument("--merge_existing", action="store_true")
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
        rerun = {a.name for a in adapters}
        old_rows = [json.loads(x) for x in results_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows.extend([r for r in old_rows if r.get("system") not in rerun])
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
