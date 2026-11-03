import json
from pathlib import Path


BASE = Path(__file__).resolve().parent

RAW_OUTPUT = BASE / "processed/operation_synthetic_v1/raw_pilot_qwen32b.jsonl"
TASKS_INPUT = BASE / "processed/operation_synthetic_v1/tasks_pilot.jsonl"
RETRY_TASKS = BASE / "processed/operation_synthetic_v1/tasks_pilot_retry_failed.jsonl"
VALID_ROWS = BASE / "processed/operation_synthetic_v1/raw_pilot_qwen32b_valid_only.jsonl"
REPORT = BASE / "processed/operation_synthetic_v1/retry_task_report.json"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    valid_rows = []
    failed_task_ids = set()

    for row in read_jsonl(RAW_OUTPUT):
        if row.get("parsed_output") and row.get("valid_json_schema") is True:
            valid_rows.append(row)
        else:
            failed_task_ids.add(row["task_id"])

    retry_tasks = []
    for task in read_jsonl(TASKS_INPUT):
        if task["task_id"] in failed_task_ids:
            task["examples_requested"] = 2
            task["retry_from_failed_pilot"] = True
            retry_tasks.append(task)

    write_jsonl(VALID_ROWS, valid_rows)
    write_jsonl(RETRY_TASKS, retry_tasks)

    report = {
        "raw_output": str(RAW_OUTPUT),
        "tasks_input": str(TASKS_INPUT),
        "valid_rows_output": str(VALID_ROWS),
        "retry_tasks_output": str(RETRY_TASKS),
        "valid_rows": len(valid_rows),
        "failed_tasks": len(failed_task_ids),
        "retry_tasks": len(retry_tasks),
        "retry_examples_requested_per_task": 2,
    }

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
