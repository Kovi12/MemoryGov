import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    input_path = BASE_DIR / "processed/longmemeval_candidate_extractions_raw_qwen32b.jsonl"
    invalid_path = BASE_DIR / "processed/longmemeval_candidate_extractions_invalid_qwen32b.jsonl"
    report_path = BASE_DIR / "processed/longmemeval_candidate_extractions_validation_report.json"

    total_rows = 0
    valid_rows = 0
    invalid_rows = 0

    total_memories = 0
    should_write_counter = Counter()
    memory_type_counter = Counter()
    sensitivity_counter = Counter()
    stability_counter = Counter()
    scenario_counter = Counter()
    validation_error_counter = Counter()
    memories_per_task = []

    invalid_examples = []
    task_ids = set()
    duplicate_task_ids = 0

    for row in read_jsonl(input_path):
        total_rows += 1

        task_id = row.get("task_id")
        if task_id in task_ids:
            duplicate_task_ids += 1
        task_ids.add(task_id)

        scenario_counter[row.get("scenario_type", "UNKNOWN")] += 1

        if not row.get("valid_json_schema", False):
            invalid_rows += 1
            validation_error_counter[row.get("validation_error", "UNKNOWN")] += 1
            invalid_examples.append(row)
            continue

        valid_rows += 1
        parsed = row.get("parsed_output") or {}
        memories = parsed.get("memories") or []
        memories_per_task.append(len(memories))
        total_memories += len(memories)

        for mem in memories:
            should_write_counter[str(mem.get("should_write"))] += 1
            memory_type_counter[mem.get("memory_type", "UNKNOWN")] += 1
            sensitivity_counter[mem.get("sensitivity", "UNKNOWN")] += 1
            stability_counter[mem.get("stability", "UNKNOWN")] += 1

    if memories_per_task:
        sorted_counts = sorted(memories_per_task)
        n = len(sorted_counts)

        def pct(p):
            idx = int((p / 100) * (n - 1))
            return sorted_counts[idx]

        memories_per_task_stats = {
            "min": sorted_counts[0],
            "p50": pct(50),
            "p90": pct(90),
            "p95": pct(95),
            "p99": pct(99),
            "max": sorted_counts[-1],
        }
    else:
        memories_per_task_stats = {}

    report = {
        "input": str(input_path),
        "total_rows": total_rows,
        "unique_task_ids": len(task_ids),
        "duplicate_task_ids": duplicate_task_ids,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "valid_row_rate": valid_rows / total_rows if total_rows else 0,
        "total_extracted_memories_from_valid_rows": total_memories,
        "memories_per_valid_task": memories_per_task_stats,
        "should_write_counter": dict(should_write_counter),
        "memory_type_counter": dict(memory_type_counter),
        "sensitivity_counter": dict(sensitivity_counter),
        "stability_counter": dict(stability_counter),
        "scenario_counter": dict(scenario_counter),
        "validation_error_counter": dict(validation_error_counter),
        "invalid_output": str(invalid_path),
    }

    write_jsonl(invalid_path, invalid_examples)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved invalid examples to: {invalid_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
