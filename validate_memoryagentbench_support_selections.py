import argparse
import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_TASKS = BASE_DIR / "processed/memoryagentbench_conflict_support_selection_tasks.jsonl"
DEFAULT_INPUT = BASE_DIR / "processed/memoryagentbench_conflict_support_selections_test50_raw_qwen32b.jsonl"
DEFAULT_VALID = BASE_DIR / "processed/memoryagentbench_conflict_support_selections_valid_qwen32b.jsonl"
DEFAULT_INVALID = BASE_DIR / "processed/memoryagentbench_conflict_support_selections_invalid_qwen32b.jsonl"
DEFAULT_REPORT = BASE_DIR / "processed/memoryagentbench_conflict_support_selections_validation_report.json"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tasks(path):
    tasks = {}
    for _, row in read_jsonl(path):
        task_id = row.get("task_id")
        if task_id:
            tasks[task_id] = row
    return tasks


def extract_json_object(text):
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def normalize_support_ids(value):
    if value is None:
        return None

    if not isinstance(value, list):
        return None

    normalized = []
    for item in value:
        if isinstance(item, int):
            normalized.append(str(item))
        elif isinstance(item, str):
            item = item.strip()
            if item:
                normalized.append(item)
        else:
            return None

    seen = set()
    unique = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


def get_candidate_fact_map(task):
    facts = task.get("candidate_facts", [])
    fact_map = {}

    if not isinstance(facts, list):
        return fact_map

    for fact in facts:
        if not isinstance(fact, dict):
            continue

        fact_id = fact.get("fact_id")
        if fact_id is None:
            continue

        fact_map[str(fact_id)] = fact

    return fact_map


def validate_row(row, tasks):
    task_id = row.get("task_id")
    if not task_id:
        return False, "missing_task_id", None

    task = tasks.get(task_id)
    if task is None:
        return False, "task_id_not_found_in_tasks_file", None

    parsed = row.get("parsed_output")
    if not isinstance(parsed, dict):
        parsed = extract_json_object(row.get("raw_output"))

    if not isinstance(parsed, dict):
        return False, "parse_failed", None

    support_ids = normalize_support_ids(parsed.get("support_fact_ids"))
    if support_ids is None:
        return False, "support_fact_ids_missing_or_not_list", None

    reason = parsed.get("reason")
    if reason is not None and not isinstance(reason, str):
        return False, "reason_not_string", None

    fact_map = get_candidate_fact_map(task)
    if not fact_map:
        return False, "task_has_no_candidate_facts", None

    missing_ids = [fact_id for fact_id in support_ids if fact_id not in fact_map]
    if missing_ids:
        return False, "support_fact_id_not_in_candidate_facts:" + ",".join(missing_ids), None

    selected_facts = []
    for fact_id in support_ids:
        fact = fact_map[fact_id]
        selected_facts.append({
            "fact_id": fact_id,
            "text": fact.get("text", "")
        })

    fixed = dict(row)
    fixed["parsed_output"] = {
        "support_fact_ids": support_ids,
        "reason": reason or ""
    }
    fixed["valid_json_schema"] = True
    fixed["validation_error"] = "ok"
    fixed["selected_support_facts"] = selected_facts

    return True, "ok", fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--valid-output", default=str(DEFAULT_VALID))
    parser.add_argument("--invalid-output", default=str(DEFAULT_INVALID))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    input_path = Path(args.input)
    valid_output = Path(args.valid_output)
    invalid_output = Path(args.invalid_output)
    report_path = Path(args.report)

    tasks = load_tasks(tasks_path)

    valid_rows = []
    invalid_rows = []

    error_counter = Counter()
    source_counter = Counter()
    subset_counter = Counter()
    support_count_counter = Counter()
    support_count_values = []

    total_rows = 0

    for line_no, row in read_jsonl(input_path):
        total_rows += 1

        ok, error, fixed = validate_row(row, tasks)

        source_counter[row.get("source_file", "unknown")] += 1
        subset_counter[row.get("source_subset", "unknown")] += 1

        if ok:
            support_count = len(fixed["parsed_output"]["support_fact_ids"])
            support_count_counter[str(support_count)] += 1
            support_count_values.append(support_count)
            valid_rows.append(fixed)
        else:
            bad = dict(row)
            bad["line_no"] = line_no
            bad["validation_error"] = error
            bad["valid_json_schema"] = False
            invalid_rows.append(bad)
            error_counter[error] += 1

    write_jsonl(valid_output, valid_rows)
    write_jsonl(invalid_output, invalid_rows)

    support_count_values_sorted = sorted(support_count_values)

    def percentile(p):
        if not support_count_values_sorted:
            return None
        idx = int(round((len(support_count_values_sorted) - 1) * p))
        return support_count_values_sorted[idx]

    report = {
        "version": "v3.29",
        "tasks": str(tasks_path),
        "input": str(input_path),
        "valid_output": str(valid_output),
        "invalid_output": str(invalid_output),
        "total_rows": total_rows,
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_rows),
        "valid_rate": len(valid_rows) / total_rows if total_rows else 0.0,
        "unique_task_ids_valid": len({row.get("task_id") for row in valid_rows}),
        "error_counter": dict(error_counter),
        "source_counter": dict(source_counter),
        "source_subset_counter": dict(subset_counter),
        "support_fact_count_counter": dict(support_count_counter),
        "support_fact_count_stats": {
            "min": min(support_count_values_sorted) if support_count_values_sorted else None,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(support_count_values_sorted) if support_count_values_sorted else None
        }
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
