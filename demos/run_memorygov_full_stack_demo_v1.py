import argparse
import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
EXT = ROOT / "external_baselines"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))

from run_external_hard_ood_lifecycle_eval_v2 import evaluate_case
from run_external_hard_ood_lifecycle_eval_v3_free import build_free_adapters, DEFAULT_CASES
from run_external_tier2_eval_v1 import load_jsonl, write_jsonl


OPS_ORDER = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]


def short(x, n=220):
    x = str(x or "").replace("\n", " ").strip()
    return x if len(x) <= n else x[: n - 3] + "..."


def pick_cases(cases, ops):
    chosen = []
    seen = set()

    for op in ops:
        for c in cases:
            if c.get("expected_operation") == op and op not in seen:
                chosen.append(c)
                seen.add(op)
                break

    return chosen


def state_counts(state):
    return {
        "active": len(state.get("active_texts", []) or []),
        "deleted": len(state.get("deleted_texts", []) or []),
        "quarantined": len(state.get("quarantined_texts", []) or []),
        "superseded": len(state.get("superseded_texts", []) or []),
    }


def sample_state(state, key, limit=3):
    xs = state.get(key, []) or []
    return [short(x, 180) for x in xs[:limit]]


def make_markdown(rows, cases, out_path):
    by_case = {c["case_id"]: c for c in cases}

    lines = []
    lines.append("# MemoryGov full-stack lifecycle demo")
    lines.append("")
    lines.append("This demo runs complete lifecycle behavior: seeding prior memory, processing a new turn, mutating memory state, retrieving context, and scoring observable behavior.")
    lines.append("")
    lines.append("## Compact result table")
    lines.append("")
    lines.append("| Case | Expected | System | Score | Observed op | Active | Deleted | Superseded | Quarantined |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|")

    for r in rows:
        post = r.get("post_state", {}) or r.get("state", {}) or {}
        counts = state_counts(post)
        observed = r.get("aux", {}).get("observed_operation")
        lines.append(
            f"| {r.get('case_id')} "
            f"| {r.get('expected_operation')} "
            f"| {r.get('system')} "
            f"| {r.get('score') if r.get('score') is not None else ''} "
            f"| {observed or ''} "
            f"| {counts['active']} "
            f"| {counts['deleted']} "
            f"| {counts['superseded']} "
            f"| {counts['quarantined']} |"
        )

    lines.append("")
    lines.append("## Detailed case walkthroughs")

    for case in cases:
        cid = case["case_id"]
        lines.append("")
        lines.append(f"### {cid} - expected {case.get('expected_operation')}")
        lines.append("")
        lines.append(f"- Category: `{case.get('category')}`")
        lines.append(f"- Existing memory: `{short(case.get('existing_memory'))}`")
        lines.append(f"- Candidate memory: `{short(case.get('candidate_memory'))}`")
        lines.append(f"- Turn: `{short(case.get('turn'), 360)}`")
        lines.append("")

        case_rows = [r for r in rows if r.get("case_id") == cid]
        for r in case_rows:
            post = r.get("post_state", {}) or r.get("state", {}) or {}
            lines.append(f"#### System: {r.get('system')}")
            lines.append("")
            lines.append(f"- Score: `{r.get('score')}`")
            lines.append(f"- Observed operation: `{r.get('aux', {}).get('observed_operation')}`")
            lines.append(f"- Checks: `{json.dumps(r.get('checks', {}), ensure_ascii=False)}`")
            lines.append(f"- Retrieved after turn: `{[short(x, 180) for x in (r.get('post_retrieved_texts') or [])[:3]]}`")
            lines.append(f"- Active sample: `{sample_state(post, 'active_texts')}`")
            lines.append(f"- Deleted sample: `{sample_state(post, 'deleted_texts')}`")
            lines.append(f"- Superseded sample: `{sample_state(post, 'superseded_texts')}`")
            lines.append(f"- Quarantined sample: `{sample_state(post, 'quarantined_texts')}`")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--out_dir", default="processed/demo_runs/full_stack_demo_v1")
    ap.add_argument(
        "--systems",
        default="memorygov,rule_based_lifecycle_memory,semantic_vector_memory,append_only_memory",
    )
    ap.add_argument(
        "--ops",
        default="WRITE,IGNORE,UPDATE,MERGE,FORGET,QUARANTINE",
    )
    ap.add_argument("--margin_threshold", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]

    cases = load_jsonl(Path(args.cases))
    demo_cases = pick_cases(cases, ops)

    if not demo_cases:
        raise RuntimeError("No demo cases selected.")

    adapters = build_free_adapters(systems, out_dir, args.margin_threshold)

    rows = []
    for adapter in adapters:
        print(f"\n=== Demo system: {adapter.name} ===")
        for case in demo_cases:
            row = evaluate_case(adapter, case)
            rows.append(row)
            print(
                f"{adapter.name} | {case['case_id']} | "
                f"expected={case.get('expected_operation')} | score={row.get('score')}"
            )
        adapter.close()

    write_jsonl(out_dir / "demo_results.jsonl", rows)
    write_jsonl(out_dir / "demo_cases.jsonl", demo_cases)
    make_markdown(rows, demo_cases, out_dir / "DEMO_REPORT.md")

    print("")
    print("Wrote:")
    print(out_dir / "demo_cases.jsonl")
    print(out_dir / "demo_results.jsonl")
    print(out_dir / "DEMO_REPORT.md")


if __name__ == "__main__":
    main()
