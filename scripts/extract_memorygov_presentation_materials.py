#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


LIFECYCLE_LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE", "ASK_USER"]
RETRIEVAL_LABELS = ["RETRIEVE", "SUPPRESS"]

DEFAULT_ROOT = Path("./ProiectD2/memory_gov_data/processed")

OPERATION_FILES = [
    "memorygovbench_operation_v1_model_ready/operation_train_balanced_model_ready.jsonl",
    "memorygovbench_operation_v1_model_ready/operation_test_synthetic_model_ready.jsonl",
    "memorygovbench_operation_v1_model_ready/operation_test_benchmark_model_ready.jsonl",
    "memorygovbench_operation_v1_model_ready/operation_test_combined_model_ready.jsonl",
    "memorygovbench_operation_v1/operation_train_balanced.jsonl",
    "audit_longmemeval_atomic/samples_by_decision.json",
]

RETRIEVAL_FILES = [
    "longmemeval_retrieval_governance.jsonl",
    "locomo_retrieval_governance.jsonl",
    "memoryagentbench_conflict_retrieval_governance_balanced_qwen32b.jsonl",
    "memoryagentbench_conflict_retrieval_governance_natural_qwen32b.jsonl",
]

EVAL_FILES = [
    "embedding_classifier_results/retrieval_all/bge_base/linear_svc_balanced/metrics.json",
    "embedding_classifier_results/retrieval_all/bge_base/logreg_balanced/metrics.json",
    "embedding_classifier_results/retrieval_all/e5_base/linear_svc_balanced/metrics.json",
    "external_baseline_results/tier2_v1/tier2_external_summary.json",
    "external_baseline_results/hard_ood_v2b_lifecycle_v5_runtime_v4_9/summary.json",
    "demo_runs/full_stack_demo_v1/demo_results.jsonl",
    "demo_runs/memorygov_only_demo_v1/demo_results.jsonl",
]


def read_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if max_lines is not None and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield i, obj


def read_json_any(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return

    if isinstance(data, dict):
        emitted = False
        for key, val in data.items():
            if isinstance(val, list):
                for idx, item in enumerate(val, start=1):
                    if isinstance(item, dict):
                        row = dict(item)
                        row.setdefault("_group_key", key)
                        yield idx, row
                        emitted = True
        if not emitted:
            yield 1, data
    elif isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if isinstance(item, dict):
                yield idx, item


def iter_records(path: Path, max_lines: Optional[int] = None) -> Iterator[Tuple[int, Dict[str, Any]]]:
    if not path.exists():
        return
    if path.suffix.lower() == ".jsonl":
        yield from read_jsonl(path, max_lines=max_lines)
    elif path.suffix.lower() == ".json":
        yield from read_json_any(path)


def get_nested(obj: Any, path: List[str]) -> Any:
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def first_existing(obj: Dict[str, Any], paths: List[List[str]]) -> Any:
    for p in paths:
        val = get_nested(obj, p)
        if val is not None:
            return val
    return None


def normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, dict):
        for key in ["primary_decision", "decision", "label", "gold_label", "operation", "target", "class"]:
            if key in x:
                return normalize_label(x[key])
        return None
    if isinstance(x, list):
        return None
    s = str(x).strip().upper()
    return s or None


def get_label(row: Dict[str, Any]) -> Optional[str]:
    val = first_existing(row, [
        ["label"],
        ["target"],
        ["gold_label"],
        ["decision"],
        ["operation"],
        ["gold", "primary_decision"],
        ["gold", "decision"],
        ["raw", "gold", "primary_decision"],
        ["_group_key"],
    ])
    return normalize_label(val)


def compact_text(value: Any, max_len: int = 700) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text


def pretty_block(value: Any, max_len: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    text = text.strip()
    if len(text) > max_len:
        return text[:max_len].rstrip() + "\n..."
    return text


def source_name(row: Dict[str, Any]) -> str:
    val = first_existing(row, [
        ["source_dataset"],
        ["raw", "source_dataset"],
        ["source_file"],
        ["raw", "source_file"],
        ["dataset"],
        ["source"],
    ])
    if val is None:
        return "unknown"
    s = str(val)
    low = s.lower()
    if "longmemeval" in low:
        return "LongMemEval"
    if "locomo" in low:
        return "LoCoMo"
    if "memoryagentbench" in low or "conflict" in low or "factconsolidation" in low:
        return "MemoryAgentBench"
    if "synthetic" in low:
        return "SyntheticAnchored"
    return s[:80]


def get_candidate_memory(row: Dict[str, Any]) -> str:
    return compact_text(first_existing(row, [
        ["candidate_memory", "text"],
        ["raw", "candidate_memory", "text"],
        ["candidate_memory"],
        ["memory_text"],
        ["candidate"],
        ["text"],
    ]), max_len=650)


def get_query(row: Dict[str, Any]) -> str:
    return compact_text(first_existing(row, [
        ["current_query"],
        ["raw", "current_query"],
        ["query"],
        ["question"],
        ["user_query"],
    ]), max_len=500)


def get_context(row: Dict[str, Any]) -> str:
    return pretty_block(first_existing(row, [
        ["history_context"],
        ["raw", "history_context"],
        ["context"],
        ["conversation"],
        ["messages"],
    ]), max_len=1200)


def get_existing(row: Dict[str, Any]) -> str:
    return pretty_block(first_existing(row, [
        ["existing_memories"],
        ["raw", "existing_memories"],
        ["state", "memories"],
        ["pre_state"],
        ["post_state"],
    ]), max_len=1000)


def get_expected(row: Dict[str, Any]) -> str:
    return compact_text(first_existing(row, [
        ["expected_answer"],
        ["raw", "expected_answer"],
        ["expected_answers"],
        ["answer"],
        ["gold_answer"],
    ]), max_len=500)


def choose_examples(root: Path, file_list: List[str], labels: List[str], per_label: int, max_scan_per_file: int):
    buckets: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in labels}
    counts_rows: List[Dict[str, Any]] = []

    for rel_path in file_list:
        path = root / rel_path
        if not path.exists():
            continue

        label_counts = Counter()
        sampled = 0
        max_lines = max_scan_per_file if path.suffix.lower() == ".jsonl" else None

        for line_no, row in iter_records(path, max_lines=max_lines):
            sampled += 1
            lab = get_label(row)
            if not lab:
                continue
            label_counts[lab] += 1

            if lab in buckets and len(buckets[lab]) < per_label:
                ex = {
                    "label": lab,
                    "file": rel_path,
                    "line_or_index": line_no,
                    "source": source_name(row),
                    "example_id": compact_text(first_existing(row, [["example_id"], ["raw", "example_id"], ["task_id"], ["case_id"]]), 120),
                    "task_level": compact_text(first_existing(row, [["task_level"], ["raw", "task_level"], ["task_type"]]), 120),
                    "scenario_type": compact_text(first_existing(row, [["scenario_type"], ["raw", "scenario_type"], ["category"]]), 160),
                    "query": get_query(row),
                    "candidate_memory": get_candidate_memory(row),
                    "existing_memories": get_existing(row),
                    "context": get_context(row),
                    "expected_answer": get_expected(row),
                    "raw_short": pretty_block(row, max_len=1800),
                }
                buckets[lab].append(ex)

        counts_rows.append({
            "file": rel_path,
            "exists": True,
            "sampled_records_or_lines": sampled,
            "label_counts": dict(label_counts),
            "size_bytes": path.stat().st_size,
        })

    return buckets, counts_rows


def write_examples_md(path: Path, title: str, buckets: Dict[str, List[Dict[str, Any]]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n# {title}\n\n")
        for label, examples in buckets.items():
            f.write(f"\n## {label}\n\n")
            if not examples:
                f.write("_No example found in scanned files._\n\n")
                continue
            for idx, ex in enumerate(examples, start=1):
                f.write(f"### {label} example {idx}\n\n")
                f.write(f"- Source: **{ex['source']}**\n")
                f.write(f"- File: `{ex['file']}`\n")
                f.write(f"- Line/index: `{ex['line_or_index']}`\n")
                if ex["example_id"]:
                    f.write(f"- Example id: `{ex['example_id']}`\n")
                if ex["scenario_type"]:
                    f.write(f"- Scenario: `{ex['scenario_type']}`\n")
                if ex["query"]:
                    f.write(f"- Query: {ex['query']}\n")
                if ex["candidate_memory"]:
                    f.write(f"- Candidate memory:\n\n```text\n{ex['candidate_memory']}\n```\n\n")
                if ex["existing_memories"]:
                    f.write(f"- Existing memories:\n\n```text\n{ex['existing_memories']}\n```\n\n")
                if ex["expected_answer"]:
                    f.write(f"- Expected answer: {ex['expected_answer']}\n")
                if ex["context"]:
                    f.write(f"- Context/history excerpt:\n\n```text\n{ex['context']}\n```\n\n")
                f.write(f"- Gold label: **{label}**\n\n")


def write_jsonl(path: Path, buckets: Dict[str, List[Dict[str, Any]]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for examples in buckets.values():
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def write_counts_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["group", "file", "sampled_records_or_lines", "size_bytes", "label_counts"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "group": r.get("group", ""),
                "file": r.get("file", ""),
                "sampled_records_or_lines": r.get("sampled_records_or_lines", ""),
                "size_bytes": r.get("size_bytes", ""),
                "label_counts": json.dumps(r.get("label_counts", {}), ensure_ascii=False, sort_keys=True),
            })


def write_file_plan(root: Path, out: Path, op_counts, ret_counts) -> None:
    with (out / "00_dataset_file_plan.md").open("w", encoding="utf-8") as f:
        f.write("# MemoryGov presentation extraction plan\n\n")
        f.write(f"Root: `{root}`\n\n")
        f.write("## Operation governance files used\n\n")
        for row in op_counts:
            f.write(f"- `{row['file']}`: sampled `{row['sampled_records_or_lines']}`, labels `{row['label_counts']}`\n")
        f.write("\n## Retrieval governance files used\n\n")
        for row in ret_counts:
            f.write(f"- `{row['file']}`: sampled `{row['sampled_records_or_lines']}`, labels `{row['label_counts']}`\n")
        f.write("\n## Interpretation for slides\n\n")
        f.write("- Operation governance examples are used for WRITE / IGNORE / UPDATE / MERGE / FORGET / QUARANTINE / ASK_USER.\n")
        f.write("- Retrieval governance examples are used for RETRIEVE / SUPPRESS.\n")
        f.write("- LongMemEval and LoCoMo provide evidence-based retrieval labels.\n")
        f.write("- MemoryAgentBench provides fact-support selections converted to retrieval labels.\n")
        f.write("- SyntheticAnchored examples cover rare lifecycle operations.\n")


def write_slide_texts(out: Path) -> None:
    retrieval_text = """# Slide: Retrieval Governor

## Title
Retrieval Governor: de la labeluri RETRIEVE/SUPPRESS la ranking

## Slide text
Surse:
- LongMemEval: evidence sessions -> RETRIEVE, distractors -> SUPPRESS
- LoCoMo: evidence IDs -> RETRIEVE, non-evidence fragments -> SUPPRESS
- MemoryAgentBench: support facts -> RETRIEVE, non-support candidate facts -> SUPPRESS

Schema:
current_query + candidate_memory -> RETRIEVE / SUPPRESS

Training:
- antrenat separat de Operation Governor
- input: pereche query-memory
- output: RETRIEVE / SUPPRESS
- split natural + split echilibrat

Evaluare:
- problema de ranking
- metrici: MRR, Hit@K, Recall@K

Rezultate:
- MRR: 0.8748
- Hit@1: 0.7827
- Hit@3: 0.9702
- Hit@5: 0.9881
- Recall@5: 0.9373

Concluzie:
Retrieval Governor separa ce poate fi gasit semantic de ce are voie sa intre in context.
"""
    (out / "04_retrieval_slide_text.md").write_text(retrieval_text, encoding="utf-8")

    operation_text = """# Slide: Operation Governance Dataset

## Title
Operation Governance Dataset: etichete lifecycle pentru starea memoriei

## Slide text
Surse:
- LongMemEval: conversii si extragere atomica de memorii
- LoCoMo: observatii si event summaries transformate in memorii candidate
- SyntheticAnchored: cazuri rare pentru UPDATE, MERGE, FORGET, QUARANTINE, IGNORE

Schema:
history_context + candidate_memory + existing_memories -> operatie lifecycle

Etichete:
WRITE / IGNORE / UPDATE / MERGE / FORGET / QUARANTINE

Training:
- split complet pentru distributia naturala
- split echilibrat pentru operatii rare
- split-uri group-aware pentru reducerea leakage-ului

Concluzie:
Datasetul transforma benchmark-uri de memorie conversationala intr-o problema explicita de guvernanta a starii.
"""
    (out / "05_operation_dataset_slide_text.md").write_text(operation_text, encoding="utf-8")


def write_mermaid(out: Path) -> None:
    retrieval = """flowchart LR
    A[LongMemEval<br/>evidence vs distractor] --> D[(Retrieval dataset)]
    B[LoCoMo<br/>evidence IDs vs non-evidence] --> D
    C[MemoryAgentBench<br/>support facts vs non-support facts] --> D

    D --> E[Training<br/>query + candidate_memory]
    E --> F[Retrieval Governor<br/>RETRIEVE / SUPPRESS]
    F --> G[Ranking evaluation<br/>MRR, Hit@K, Recall@K]

    classDef source fill:#f3f4f6,stroke:#4b5563,color:#111827;
    classDef data fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef eval fill:#fef3c7,stroke:#d97706,color:#111827;

    class A,B,C source;
    class D,E,F data;
    class G eval;
"""
    (out / "06_mermaid_retrieval_dataset.mmd").write_text(retrieval, encoding="utf-8")

    operation = """flowchart LR
    A[LongMemEval<br/>atomic extraction] --> D[(Operation dataset)]
    B[LoCoMo<br/>observations + events] --> D
    C[SyntheticAnchored<br/>rare lifecycle cases] --> D

    D --> E[Training<br/>history + candidate + existing memories]
    E --> F[Operation Governor<br/>WRITE / IGNORE / UPDATE / MERGE / FORGET / QUARANTINE]
    F --> G[State transition<br/>active / superseded / deleted / quarantined]

    classDef source fill:#f3f4f6,stroke:#4b5563,color:#111827;
    classDef data fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef state fill:#ede9fe,stroke:#7c3aed,color:#111827;

    class A,B,C source;
    class D,E,F data;
    class G state;
"""
    (out / "07_mermaid_operation_dataset.mmd").write_text(operation, encoding="utf-8")


def try_make_figures(out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics = {
        "MRR": 0.8748,
        "Hit@1": 0.7827,
        "Hit@3": 0.9702,
        "Hit@5": 0.9881,
        "Recall@5": 0.9373,
    }
    plt.figure(figsize=(8, 4.5))
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.ylim(0, 1.05)
    plt.title("Retrieval Governor ranking metrics")
    plt.ylabel("Score")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "retrieval_governor_metrics.png", dpi=220)
    plt.close()

    e2e = {
        "MemoryGov": 0.7817,
        "Rule-based": 0.6417,
        "Regex safety": 0.6350,
        "Append-only": 0.4700,
        "Vector": 0.4700,
        "Recency/summary": 0.4700,
    }
    plt.figure(figsize=(9, 4.8))
    plt.bar(list(e2e.keys()), list(e2e.values()))
    plt.ylim(0, 1.0)
    plt.title("End-to-end baseline comparison")
    plt.ylabel("Mean score")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(e2e.values()):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "end_to_end_baselines.png", dpi=220)
    plt.close()

    multi = {
        "MemoryGov + decomposer": 0.8843,
        "Rule-based": 0.7187,
        "MemoryGov no decomposition": 0.6383,
        "Append-only": 0.4467,
        "Vector memory": 0.4467,
    }
    plt.figure(figsize=(9, 4.8))
    plt.bar(list(multi.keys()), list(multi.values()))
    plt.ylim(0, 1.0)
    plt.title("Multi-intent evaluation")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(multi.values()):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "multi_intent_comparison.png", dpi=220)
    plt.close()


def extract_demo_cases(root: Path, out: Path, max_cases: int = 8) -> None:
    candidates = [
        "demo_runs/full_stack_demo_v1/demo_results.jsonl",
        "demo_runs/full_stack_demo_v1/demo_cases.jsonl",
        "demo_runs/memorygov_only_demo_v1/demo_results.jsonl",
        "external_baseline_results/tier2_v1/tier2_external_results.jsonl",
    ]

    written = 0
    with (out / "08_demo_cases_examples.md").open("w", encoding="utf-8") as f:
        f.write("# Demo and end-to-end examples\n\n")
        for rel in candidates:
            p = root / rel
            if not p.exists() or written >= max_cases:
                continue
            f.write(f"\n## `{rel}`\n\n")
            for line_no, row in read_jsonl(p, max_lines=200):
                if written >= max_cases:
                    break
                case_id = compact_text(first_existing(row, [["case_id"], ["example_id"], ["id"]]), 100)
                category = compact_text(first_existing(row, [["category"], ["expected_operation"], ["risk"]]), 150)
                score = compact_text(first_existing(row, [["score"], ["mean_score"]]), 80)
                turns = first_existing(row, [["turn_results"]])
                retrieved = first_existing(row, [["retrieved"], ["post_retrieved_texts"], ["pre_retrieved_texts"]])
                state = first_existing(row, [["state"], ["post_state"]])

                if not (turns or retrieved or state):
                    continue

                f.write(f"### Case {case_id or line_no}\n\n")
                f.write(f"- File: `{rel}` line `{line_no}`\n")
                if category:
                    f.write(f"- Category/risk: `{category}`\n")
                if score:
                    f.write(f"- Score: `{score}`\n")
                if turns:
                    f.write("- Turn results:\n\n```json\n" + pretty_block(turns, 1200) + "\n```\n\n")
                if retrieved:
                    f.write("- Retrieved:\n\n```json\n" + pretty_block(retrieved, 900) + "\n```\n\n")
                if state:
                    f.write("- State:\n\n```json\n" + pretty_block(state, 1200) + "\n```\n\n")
                written += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=None)
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument("--max-scan-per-file", type=int, default=50000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root.parent / "_presentation_materials"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    if not root.exists():
        print(f"[ERROR] root does not exist: {root}", file=sys.stderr)
        return 2

    for name in ["01_selected_examples.md", "02_selected_examples.jsonl"]:
        p = out / name
        if p.exists():
            p.unlink()

    op_buckets, op_counts = choose_examples(root, OPERATION_FILES, LIFECYCLE_LABELS, args.per_label, args.max_scan_per_file)
    for r in op_counts:
        r["group"] = "operation"

    ret_buckets, ret_counts = choose_examples(root, RETRIEVAL_FILES, RETRIEVAL_LABELS, args.per_label, args.max_scan_per_file)
    for r in ret_counts:
        r["group"] = "retrieval"

    write_file_plan(root, out, op_counts, ret_counts)
    write_examples_md(out / "01_selected_examples.md", "Operation governance examples", op_buckets)
    write_examples_md(out / "01_selected_examples.md", "Retrieval governance examples", ret_buckets)
    write_jsonl(out / "02_selected_examples.jsonl", op_buckets)
    write_jsonl(out / "02_selected_examples.jsonl", ret_buckets)
    write_counts_csv(out / "03_label_counts_sampled.csv", op_counts + ret_counts)
    write_slide_texts(out)
    write_mermaid(out)
    try_make_figures(out)
    extract_demo_cases(root, out)

    print("[OK] Presentation materials extracted.")
    print(f"Output folder: {out}")
    print("\nMain files:")
    for name in [
        "00_dataset_file_plan.md",
        "01_selected_examples.md",
        "02_selected_examples.jsonl",
        "03_label_counts_sampled.csv",
        "04_retrieval_slide_text.md",
        "05_operation_dataset_slide_text.md",
        "06_mermaid_retrieval_dataset.mmd",
        "07_mermaid_operation_dataset.mmd",
        "08_demo_cases_examples.md",
    ]:
        print(f"  {out / name}")

    fig_dir = out / "figures"
    if fig_dir.exists():
        figs = list(fig_dir.glob("*.png"))
        if figs:
            print("\nFigures:")
            for fig in figs:
                print(f"  {fig}")

    print("\nPreview commands:")
    print(f"  sed -n '1,260p' {out}/01_selected_examples.md")
    print(f"  sed -n '1,220p' {out}/04_retrieval_slide_text.md")
    print(f"  column -s, -t < {out}/03_label_counts_sampled.csv | sed -n '1,80p'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
