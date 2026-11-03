import argparse
import json
from collections import Counter
from pathlib import Path


STATUS_BY_LABEL = {
    "UPDATE": "superseded",
    "MERGE": "merged",
    "FORGET": "deleted",
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


def get_existing_ids(example):
    ids = []
    for mem in example.get("existing_memories", []) or []:
        memory_id = mem.get("memory_id")
        if memory_id:
            ids.append(memory_id)
    return ids


def normalize_example(example):
    changes = []

    gold = example.get("gold")
    if not isinstance(gold, dict):
        return changes

    label = gold.get("primary_decision")

    if label in STATUS_BY_LABEL:
        wanted_status = STATUS_BY_LABEL[label]

        if gold.get("new_status_for_targets") != wanted_status:
            gold["new_status_for_targets"] = wanted_status
            changes.append(f"{label.lower()}_status_set_to_{wanted_status}")

        target_ids = gold.get("target_memory_ids")
        if not isinstance(target_ids, list) or not target_ids:
            existing_ids = get_existing_ids(example)
            if existing_ids:
                gold["target_memory_ids"] = existing_ids[:1]
                changes.append(f"{label.lower()}_target_memory_id_filled")

    if label in {"IGNORE", "WRITE", "ASK_USER", "QUARANTINE"}:
        if gold.get("new_status_for_targets") not in {None, ""}:
            gold["new_status_for_targets"] = None
            changes.append(f"{label.lower()}_status_cleared")

    metadata = example.setdefault("annotation_metadata", {})
    if changes:
        metadata.setdefault("normalization_notes", [])
        metadata["normalization_notes"].extend(changes)

    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rows = []
    row_counter = Counter()
    example_counter = Counter()
    change_counter = Counter()
    label_counter = Counter()

    for row in read_jsonl(args.input):
        row_counter["total_rows"] += 1

        parsed = row.get("parsed_output")
        if not isinstance(parsed, dict):
            rows.append(row)
            row_counter["rows_without_parsed_output"] += 1
            continue

        examples = parsed.get("examples")
        if not isinstance(examples, list):
            rows.append(row)
            row_counter["rows_without_examples_list"] += 1
            continue

        row_changed = False

        for example in examples:
            label = example.get("gold", {}).get("primary_decision")
            label_counter[label] += 1
            changes = normalize_example(example)

            if changes:
                row_changed = True
                example_counter["examples_changed"] += 1
                for change in changes:
                    change_counter[change] += 1
            else:
                example_counter["examples_unchanged"] += 1

        if row_changed:
            row_counter["rows_changed"] += 1
        else:
            row_counter["rows_unchanged"] += 1

        rows.append(row)

    write_jsonl(args.output, rows)

    report = {
        "version": "v3.55",
        "task": "normalize_operation_synthetic_gold_fields",
        "input": args.input,
        "output": args.output,
        "rows": dict(row_counter),
        "examples": dict(example_counter),
        "labels": dict(label_counter),
        "changes": dict(change_counter),
        "policy": {
            "UPDATE": "new_status_for_targets must be superseded",
            "MERGE": "new_status_for_targets must be merged",
            "FORGET": "new_status_for_targets must be deleted",
            "IGNORE_WRITE_ASK_USER_QUARANTINE": "new_status_for_targets should be null unless a target mutation is explicitly required",
        },
    }

    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
