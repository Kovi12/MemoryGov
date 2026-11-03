import argparse
import json
from collections import Counter
from pathlib import Path


VALID_LABELS = {
    "WRITE",
    "IGNORE",
    "UPDATE",
    "MERGE",
    "FORGET",
    "ASK_USER",
    "QUARANTINE",
}


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_examples_array_start(text):
    key_idx = text.find('"examples"')
    if key_idx < 0:
        return -1

    bracket_idx = text.find("[", key_idx)
    if bracket_idx < 0:
        return -1

    return bracket_idx + 1


def extract_complete_json_objects_from_array(text, start_idx):
    objects = []
    i = start_idx
    n = len(text)

    while i < n:
        while i < n and text[i] in " \n\r\t,":
            i += 1

        if i >= n or text[i] == "]":
            break

        if text[i] != "{":
            i += 1
            continue

        obj_start = i
        depth = 0
        in_string = False
        escape = False

        while i < n:
            ch = text[i]

            if escape:
                escape = False
                i += 1
                continue

            if ch == "\\":
                escape = True
                i += 1
                continue

            if ch == '"':
                in_string = not in_string
                i += 1
                continue

            if in_string:
                i += 1
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj_text = text[obj_start:i + 1]
                    try:
                        objects.append(json.loads(obj_text))
                    except Exception:
                        pass
                    i += 1
                    break

            i += 1
        else:
            break

    return objects


def extract_examples(raw_output):
    if not raw_output:
        return []

    start = find_examples_array_start(raw_output)
    if start < 0:
        return []

    return extract_complete_json_objects_from_array(raw_output, start)


def example_label(example):
    return example.get("gold", {}).get("primary_decision")


def valid_example(example, target_label=None):
    if not isinstance(example, dict):
        return False

    if not isinstance(example.get("candidate_memory"), dict):
        return False

    if not example["candidate_memory"].get("text"):
        return False

    if not isinstance(example.get("gold"), dict):
        return False

    label = example["gold"].get("primary_decision")
    if label not in VALID_LABELS:
        return False

    if target_label is not None and label != target_label:
        return False

    return True


def validate_row_schema(row, min_examples):
    parsed = row.get("parsed_output")
    if not isinstance(parsed, dict):
        return False, "missing parsed_output"

    examples = parsed.get("examples")
    if not isinstance(examples, list):
        return False, "missing examples list"

    if len(examples) < min_examples:
        return False, f"too few examples: {len(examples)}"

    target_label = row.get("target_label")
    for i, example in enumerate(examples):
        if not valid_example(example, target_label=target_label):
            return False, f"invalid example {i}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--limit_examples", type=int, default=2)
    parser.add_argument("--min_examples", type=int, default=1)
    args = parser.parse_args()

    output_rows = []
    status_counter = Counter()
    label_counter = Counter()
    example_counter = Counter()
    salvage_count_by_label = Counter()

    for row in read_jsonl(args.input):
        target_label = row.get("target_label")
        label_counter[target_label] += 1

        parsed = row.get("parsed_output")
        already_valid = row.get("valid_json_schema") is True and isinstance(parsed, dict)

        if already_valid:
            examples = [
                ex for ex in parsed.get("examples", [])
                if valid_example(ex, target_label=target_label)
            ]

            examples = examples[:args.limit_examples]
            row["parsed_output"] = {"examples": examples}

            ok, err = validate_row_schema(row, args.min_examples)
            row["valid_json_schema"] = ok
            row["validation_error"] = err
            row["salvage_status"] = "already_valid_trimmed" if ok else "already_valid_but_failed_after_trim"

            if ok:
                status_counter["already_valid_trimmed"] += 1
                example_counter[target_label] += len(examples)
            else:
                status_counter["already_valid_but_failed_after_trim"] += 1

            output_rows.append(row)
            continue

        raw_output = row.get("raw_output") or ""
        extracted = extract_examples(raw_output)

        valid_extracted = [
            ex for ex in extracted
            if valid_example(ex, target_label=target_label)
        ]

        selected = valid_extracted[:args.limit_examples]
        row["parsed_output"] = {"examples": selected}

        ok, err = validate_row_schema(row, args.min_examples)
        row["valid_json_schema"] = ok
        row["validation_error"] = err

        if ok:
            row["salvage_status"] = "salvaged_from_truncated_raw_output"
            status_counter["salvaged_from_truncated_raw_output"] += 1
            salvage_count_by_label[target_label] += 1
            example_counter[target_label] += len(selected)
        else:
            row["salvage_status"] = "not_salvageable"
            row["salvage_error"] = err
            status_counter["not_salvageable"] += 1

        row["salvaged_complete_objects_found"] = len(extracted)
        row["salvaged_valid_examples_found"] = len(valid_extracted)
        row["salvaged_examples_used"] = len(selected)

        output_rows.append(row)

    write_jsonl(args.output, output_rows)

    report = {
        "input": args.input,
        "output": args.output,
        "total_rows": len(output_rows),
        "limit_examples": args.limit_examples,
        "min_examples": args.min_examples,
        "status_counter": dict(status_counter),
        "task_label_counter": dict(label_counter),
        "generated_example_counter": dict(example_counter),
        "salvaged_row_counter_by_label": dict(salvage_count_by_label),
    }

    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
