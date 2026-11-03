import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = BASE_DIR / "processed/memorygovbench_current"
REPORT_PATH = OUT_DIR / "memorygovbench_current_report.json"

INPUTS = {
    "operation": {
        "train": [
            BASE_DIR / "processed/splits_longmemeval/operation_train.jsonl",
            BASE_DIR / "processed/splits_locomo/operation_train.jsonl",
        ],
        "val": [
            BASE_DIR / "processed/splits_longmemeval/operation_val.jsonl",
            BASE_DIR / "processed/splits_locomo/operation_val.jsonl",
        ],
        "test": [
            BASE_DIR / "processed/splits_longmemeval/operation_test.jsonl",
            BASE_DIR / "processed/splits_locomo/operation_test.jsonl",
        ],
    },
    "retrieval_balanced": {
        "train": [
            BASE_DIR / "processed/splits_longmemeval/retrieval_balanced_train.jsonl",
            BASE_DIR / "processed/splits_locomo/retrieval_balanced_train.jsonl",
            BASE_DIR / "processed/splits_memoryagentbench_conflict/retrieval_balanced_train.jsonl",
        ],
        "val": [
            BASE_DIR / "processed/splits_longmemeval/retrieval_balanced_val.jsonl",
            BASE_DIR / "processed/splits_locomo/retrieval_balanced_val.jsonl",
            BASE_DIR / "processed/splits_memoryagentbench_conflict/retrieval_balanced_val.jsonl",
        ],
        "test": [
            BASE_DIR / "processed/splits_longmemeval/retrieval_balanced_test.jsonl",
            BASE_DIR / "processed/splits_locomo/retrieval_balanced_test.jsonl",
            BASE_DIR / "processed/splits_memoryagentbench_conflict/retrieval_balanced_test.jsonl",
        ],
    },
    "retrieval_natural": {
        "train": [
            BASE_DIR / "processed/splits_longmemeval/retrieval_natural_train.jsonl",
            BASE_DIR / "processed/splits_locomo/retrieval_natural_train.jsonl",
            BASE_DIR / "processed/splits_memoryagentbench_conflict/retrieval_natural_train.jsonl",
        ],
        "val": [
            BASE_DIR / "processed/splits_longmemeval/retrieval_natural_val.jsonl",
            BASE_DIR / "processed/splits_locomo/retrieval_natural_val.jsonl",
            BASE_DIR / "processed/splits_memoryagentbench_conflict/retrieval_natural_val.jsonl",
        ],
        "test": [
            BASE_DIR / "processed/splits_longmemeval/retrieval_natural_test.jsonl",
            BASE_DIR / "processed/splits_locomo/retrieval_natural_test.jsonl",
            BASE_DIR / "processed/splits_memoryagentbench_conflict/retrieval_natural_test.jsonl",
        ],
    },
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_label(row):
    return row.get("gold", {}).get("primary_decision", "unknown")


def write_merged(task_name, split_name, paths):
    output_path = OUT_DIR / f"{task_name}_{split_name}.jsonl"
    stats = {
        "output": str(output_path),
        "input_files": [str(p) for p in paths],
        "total": 0,
        "label_counter": Counter(),
        "source_dataset_counter": Counter(),
        "source_file_counter": Counter(),
        "scenario_counter": Counter(),
        "memory_type_counter": Counter(),
        "sensitivity_counter": Counter(),
        "duplicate_example_ids": 0,
    }

    seen_ids = set()

    with open(output_path, "w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(str(path))

            for row in read_jsonl(path):
                example_id = row.get("example_id")
                if example_id in seen_ids:
                    stats["duplicate_example_ids"] += 1
                    continue

                seen_ids.add(example_id)

                row["benchmark_build"] = {
                    "name": "MemoryGovBench-current",
                    "version": "v3.31",
                    "merged_from": str(path.relative_to(BASE_DIR))
                }

                out.write(json.dumps(row, ensure_ascii=False) + "\n")

                candidate = row.get("candidate_memory", {})
                stats["total"] += 1
                stats["label_counter"][get_label(row)] += 1
                stats["source_dataset_counter"][row.get("source_dataset", "unknown")] += 1
                stats["source_file_counter"][row.get("source_file", "unknown")] += 1
                stats["scenario_counter"][row.get("scenario_type", "unknown")] += 1
                stats["memory_type_counter"][candidate.get("memory_type", "unknown")] += 1
                stats["sensitivity_counter"][candidate.get("sensitivity", "unknown")] += 1

    return {
        "output": stats["output"],
        "input_files": stats["input_files"],
        "total": stats["total"],
        "label_counter": dict(stats["label_counter"]),
        "source_dataset_counter": dict(stats["source_dataset_counter"]),
        "source_file_counter": dict(stats["source_file_counter"]),
        "scenario_counter": dict(stats["scenario_counter"]),
        "memory_type_counter": dict(stats["memory_type_counter"]),
        "sensitivity_counter": dict(stats["sensitivity_counter"]),
        "duplicate_example_ids_removed": stats["duplicate_example_ids"],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "version": "v3.31",
        "dataset_name": "MemoryGovBench-current",
        "description": "Merged current dataset from LongMemEval, LoCoMo, and MemoryAgentBench Conflict Resolution.",
        "splits": {},
        "totals": defaultdict(Counter),
    }

    for task_name, split_map in INPUTS.items():
        report["splits"][task_name] = {}

        for split_name, paths in split_map.items():
            split_stats = write_merged(task_name, split_name, paths)
            report["splits"][task_name][split_name] = split_stats

            report["totals"][task_name]["total"] += split_stats["total"]
            for label, count in split_stats["label_counter"].items():
                report["totals"][task_name][f"label::{label}"] += count

    report["totals"] = {k: dict(v) for k, v in report["totals"].items()}

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
