import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_TASKS = BASE_DIR / "processed/memoryagentbench_conflict_support_selection_tasks.jsonl"
DEFAULT_VALID = BASE_DIR / "processed/memoryagentbench_conflict_support_selections_valid_qwen32b.jsonl"
DEFAULT_NATURAL = BASE_DIR / "processed/memoryagentbench_conflict_retrieval_governance_natural_qwen32b.jsonl"
DEFAULT_BALANCED = BASE_DIR / "processed/memoryagentbench_conflict_retrieval_governance_balanced_qwen32b.jsonl"
DEFAULT_REPORT = BASE_DIR / "processed/memoryagentbench_conflict_retrieval_governance_report.json"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tasks(path):
    tasks = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if task_id:
            tasks[task_id] = row
    return tasks


def stable_hash(text, length=16):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def group_key_for_task(task):
    split_group = {
        "source_dataset": task.get("source_dataset", "MemoryAgentBench"),
        "source_file": task.get("source_file"),
        "source_subset": task.get("source_subset"),
        "row_idx": task.get("row_idx"),
        "qa_idx": task.get("qa_idx"),
        "qa_pair_id": task.get("qa_pair_id")
    }
    return split_group


def first_answer(expected_answers):
    if isinstance(expected_answers, list) and expected_answers:
        return str(expected_answers[0])
    if isinstance(expected_answers, str):
        return expected_answers
    return None


def fact_to_example(task, selection_row, fact, selected_ids):
    fact_id = str(fact.get("fact_id"))
    fact_text = str(fact.get("text", "")).strip()
    is_support = fact_id in selected_ids

    decision = "RETRIEVE" if is_support else "SUPPRESS"
    failure_type = None if is_support else "non_supporting_or_distractor_fact"

    example_id = "mab_conflict_ret_" + stable_hash(
        task.get("task_id", "") + "|" + fact_id + "|" + decision
    )

    expected_answers = task.get("expected_answers", [])

    annotation_confidence = 0.85 if is_support else 0.70

    return {
        "example_id": example_id,
        "source_dataset": "MemoryAgentBench",
        "source_file": task.get("source_file", "Conflict_Resolution.jsonl"),
        "source_subset": task.get("source_subset"),
        "split_group": group_key_for_task(task),
        "task_level": "retrieval_governance",
        "scenario_type": "memoryagentbench_conflict_resolution",
        "failure_type": failure_type,
        "history_context": [],
        "candidate_memory": {
            "text": fact_text,
            "memory_type": "semantic",
            "source_type": "benchmark_fact",
            "source_trust": "benchmark_llm_assisted",
            "sensitivity": "low",
            "confidence": 1.0,
            "valid_from": None,
            "valid_to": None
        },
        "existing_memories": [],
        "current_query": task.get("current_query", ""),
        "expected_answer": first_answer(expected_answers),
        "gold": {
            "primary_decision": decision,
            "secondary_decision": None,
            "target_memory_ids": [],
            "new_status_for_targets": None,
            "should_retrieve_candidate": is_support,
            "should_retrieve_existing": None,
            "reason": (
                "The support-selection model selected this fact as part of the evidence chain required to answer the benchmark question."
                if is_support
                else "This candidate fact was not selected as support evidence for the benchmark question."
            )
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": False,
            "requires_contradiction_resolution": True,
            "requires_sensitivity_reasoning": False,
            "requires_appropriateness_reasoning": True,
            "requires_provenance_reasoning": True
        },
        "annotation_metadata": {
            "label_source": "llm_assisted_benchmark_derived",
            "human_audited": False,
            "confidence": annotation_confidence,
            "task_id": task.get("task_id"),
            "qa_pair_id": task.get("qa_pair_id"),
            "row_idx": task.get("row_idx"),
            "qa_idx": task.get("qa_idx"),
            "fact_id": fact_id,
            "expected_answers": expected_answers,
            "support_fact_ids": sorted(selected_ids, key=lambda x: int(x) if x.isdigit() else x),
            "selection_model": selection_row.get("model"),
            "selection_generation_seconds": selection_row.get("generation_seconds"),
            "selection_reason": selection_row.get("parsed_output", {}).get("reason", "")
        }
    }


def build_examples_for_task(task, selection_row, negative_ratio, rng):
    candidate_facts = task.get("candidate_facts", [])
    if not isinstance(candidate_facts, list):
        return [], []

    selected_ids = set(selection_row.get("parsed_output", {}).get("support_fact_ids", []))
    natural_rows = []

    for fact in candidate_facts:
        if not isinstance(fact, dict):
            continue
        if fact.get("fact_id") is None:
            continue
        natural_rows.append(fact_to_example(task, selection_row, fact, selected_ids))

    positive_rows = [row for row in natural_rows if row["gold"]["primary_decision"] == "RETRIEVE"]
    negative_rows = [row for row in natural_rows if row["gold"]["primary_decision"] == "SUPPRESS"]

    max_negatives = negative_ratio * len(positive_rows)
    if len(negative_rows) > max_negatives:
        negative_rows = rng.sample(negative_rows, max_negatives)

    balanced_rows = positive_rows + negative_rows
    balanced_rows.sort(key=lambda row: row["example_id"])

    return natural_rows, balanced_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--valid-selections", default=str(DEFAULT_VALID))
    parser.add_argument("--natural-output", default=str(DEFAULT_NATURAL))
    parser.add_argument("--balanced-output", default=str(DEFAULT_BALANCED))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--include-zero-support", action="store_true")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    valid_path = Path(args.valid_selections)
    natural_output = Path(args.natural_output)
    balanced_output = Path(args.balanced_output)
    report_path = Path(args.report)

    natural_output.parent.mkdir(parents=True, exist_ok=True)
    balanced_output.parent.mkdir(parents=True, exist_ok=True)
    natural_output.write_text("", encoding="utf-8")
    balanced_output.write_text("", encoding="utf-8")

    tasks = load_tasks(tasks_path)

    total_selection_rows = 0
    missing_task_rows = 0
    zero_support_skipped = 0
    natural_count = 0
    balanced_count = 0

    natural_label_counter = Counter()
    balanced_label_counter = Counter()
    subset_counter = Counter()
    scenario_counter = Counter()
    support_count_counter = Counter()
    task_counter_by_subset = Counter()
    natural_by_subset = defaultdict(Counter)
    balanced_by_subset = defaultdict(Counter)

    for selection_row in read_jsonl(valid_path):
        total_selection_rows += 1
        task_id = selection_row.get("task_id")
        task = tasks.get(task_id)

        if task is None:
            missing_task_rows += 1
            continue

        support_ids = selection_row.get("parsed_output", {}).get("support_fact_ids", [])
        if not support_ids and not args.include_zero_support:
            zero_support_skipped += 1
            continue

        group_seed = args.seed + int(stable_hash(str(task_id), length=8), 16)
        rng = random.Random(group_seed)

        natural_rows, balanced_rows = build_examples_for_task(
            task=task,
            selection_row=selection_row,
            negative_ratio=args.negative_ratio,
            rng=rng
        )

        if not natural_rows:
            continue

        subset = task.get("source_subset", "unknown")
        scenario = task.get("scenario_type", "memoryagentbench_conflict_resolution")

        subset_counter[subset] += 1
        scenario_counter[scenario] += 1
        support_count_counter[str(len(support_ids))] += 1
        task_counter_by_subset[subset] += 1

        for row in natural_rows:
            append_jsonl(natural_output, row)
            natural_count += 1
            label = row["gold"]["primary_decision"]
            natural_label_counter[label] += 1
            natural_by_subset[subset][label] += 1

        for row in balanced_rows:
            append_jsonl(balanced_output, row)
            balanced_count += 1
            label = row["gold"]["primary_decision"]
            balanced_label_counter[label] += 1
            balanced_by_subset[subset][label] += 1

    report = {
        "version": "v3.29",
        "tasks": str(tasks_path),
        "valid_selections": str(valid_path),
        "natural_output": str(natural_output),
        "balanced_output": str(balanced_output),
        "negative_ratio": args.negative_ratio,
        "include_zero_support": args.include_zero_support,
        "total_selection_rows": total_selection_rows,
        "missing_task_rows": missing_task_rows,
        "zero_support_skipped": zero_support_skipped,
        "natural_examples": natural_count,
        "balanced_examples": balanced_count,
        "natural_label_counter": dict(natural_label_counter),
        "balanced_label_counter": dict(balanced_label_counter),
        "source_subset_counter_used_tasks": dict(subset_counter),
        "scenario_counter_used_tasks": dict(scenario_counter),
        "support_count_counter_used_tasks": dict(support_count_counter),
        "task_counter_by_subset": dict(task_counter_by_subset),
        "natural_label_by_subset": {k: dict(v) for k, v in natural_by_subset.items()},
        "balanced_label_by_subset": {k: dict(v) for k, v in balanced_by_subset.items()}
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
