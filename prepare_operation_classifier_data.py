import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
CURRENT_DIR = PROCESSED_DIR / "memorygovbench_current"
OUT_DIR = PROCESSED_DIR / "memorygovbench_model_ready"

VERSION = "v3.33"

INPUTS = {
    "train": CURRENT_DIR / "operation_train.jsonl",
    "val": CURRENT_DIR / "operation_val.jsonl",
    "test": CURRENT_DIR / "operation_test.jsonl",
}

OUTPUTS = {
    "train": OUT_DIR / "operation_train_model_ready.jsonl",
    "val": OUT_DIR / "operation_val_model_ready.jsonl",
    "test": OUT_DIR / "operation_test_model_ready.jsonl",
}

LABEL_TO_ID = {
    "WRITE": 0,
    "IGNORE": 1,
    "UPDATE": 2,
    "MERGE": 3,
    "FORGET": 4,
    "REVIEW": 5,
    "QUARANTINE": 6,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def format_existing_memories(existing_memories):
    if not existing_memories:
        return "None"

    lines = []
    for idx, memory in enumerate(existing_memories, start=1):
        memory_id = safe_text(memory.get("memory_id")) or f"existing_{idx}"
        memory_text = safe_text(memory.get("memory_text") or memory.get("text"))
        memory_type = safe_text(memory.get("memory_type"))
        sensitivity = safe_text(memory.get("sensitivity"))
        status = safe_text(memory.get("status"))
        valid_from = safe_text(memory.get("valid_from") or memory.get("created_at"))

        parts = [f"- id: {memory_id}"]
        if memory_type:
            parts.append(f"type: {memory_type}")
        if sensitivity:
            parts.append(f"sensitivity: {sensitivity}")
        if status:
            parts.append(f"status: {status}")
        if valid_from:
            parts.append(f"valid_from: {valid_from}")
        if memory_text:
            parts.append(f"text: {memory_text}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)


def build_input_text(row):
    candidate = row.get("candidate_memory") or {}
    gold = row.get("gold") or {}

    candidate_text = safe_text(candidate.get("text"))
    candidate_type = safe_text(candidate.get("memory_type"))
    candidate_source_type = safe_text(candidate.get("source_type"))
    candidate_source_trust = safe_text(candidate.get("source_trust"))
    candidate_sensitivity = safe_text(candidate.get("sensitivity"))
    candidate_confidence = safe_text(candidate.get("confidence"))
    candidate_valid_from = safe_text(candidate.get("valid_from"))
    candidate_valid_to = safe_text(candidate.get("valid_to"))

    current_query = safe_text(row.get("current_query"))
    expected_answer = safe_text(row.get("expected_answer"))
    scenario_type = safe_text(row.get("scenario_type"))
    failure_type = safe_text(row.get("failure_type"))
    source_dataset = safe_text(row.get("source_dataset"))
    source_file = safe_text(row.get("source_file"))
    task_level = safe_text(row.get("task_level"))
    existing_memories = format_existing_memories(row.get("existing_memories") or [])

    diagnostic_tags = row.get("diagnostic_tags") or {}
    diagnostic_parts = []
    for key in sorted(diagnostic_tags):
        diagnostic_parts.append(f"{key}: {diagnostic_tags[key]}")
    diagnostic_text = "\n".join(diagnostic_parts) if diagnostic_parts else "None"

    secondary_decision = safe_text(gold.get("secondary_decision"))
    target_memory_ids = gold.get("target_memory_ids") or []
    target_memory_text = ", ".join(str(x) for x in target_memory_ids) if target_memory_ids else "None"

    sections = [
        "[TASK]",
        f"Task level: {task_level}",
        f"Source dataset: {source_dataset}",
        f"Source file: {source_file}",
        f"Scenario type: {scenario_type}",
        f"Failure type: {failure_type or 'None'}",
        "",
        "[CURRENT QUERY]",
        current_query or "None",
        "",
        "[EXPECTED ANSWER]",
        expected_answer or "None",
        "",
        "[CANDIDATE MEMORY]",
        f"Text: {candidate_text}",
        f"Memory type: {candidate_type or 'unknown'}",
        f"Source type: {candidate_source_type or 'unknown'}",
        f"Source trust: {candidate_source_trust or 'unknown'}",
        f"Sensitivity: {candidate_sensitivity or 'unknown'}",
        f"Confidence: {candidate_confidence or 'unknown'}",
        f"Valid from: {candidate_valid_from or 'unknown'}",
        f"Valid to: {candidate_valid_to or 'None'}",
        "",
        "[EXISTING MEMORIES]",
        existing_memories,
        "",
        "[DIAGNOSTIC TAGS]",
        diagnostic_text,
        "",
        "[TARGET MEMORY IDS]",
        target_memory_text,
        "",
        "[SECONDARY DECISION]",
        secondary_decision or "None",
    ]

    return "\n".join(sections).strip()


def convert_row(row, split):
    gold = row.get("gold") or {}
    label = safe_text(gold.get("primary_decision"))

    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown operation label: {label}")

    candidate = row.get("candidate_memory") or {}

    return {
        "example_id": row.get("example_id"),
        "split": split,
        "task_level": row.get("task_level"),
        "source_dataset": row.get("source_dataset"),
        "source_file": row.get("source_file"),
        "scenario_type": row.get("scenario_type"),
        "failure_type": row.get("failure_type"),
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "text": build_input_text(row),
        "current_query": row.get("current_query"),
        "candidate_memory_text": candidate.get("text"),
        "candidate_memory_type": candidate.get("memory_type"),
        "candidate_sensitivity": candidate.get("sensitivity"),
        "candidate_source_type": candidate.get("source_type"),
        "candidate_source_trust": candidate.get("source_trust"),
        "candidate_valid_from": candidate.get("valid_from"),
        "candidate_valid_to": candidate.get("valid_to"),
        "gold_reason": gold.get("reason"),
        "split_group": row.get("split_group"),
        "annotation_metadata": row.get("annotation_metadata"),
        "diagnostic_tags": row.get("diagnostic_tags"),
    }


def convert_split(split, input_path, output_path):
    label_counter = Counter()
    source_dataset_counter = Counter()
    scenario_counter = Counter()
    memory_type_counter = Counter()
    sensitivity_counter = Counter()
    rows = []

    for row in read_jsonl(input_path):
        out = convert_row(row, split)
        rows.append(out)

        label_counter[out["label"]] += 1
        source_dataset_counter[out["source_dataset"]] += 1
        scenario_counter[out["scenario_type"]] += 1
        memory_type_counter[out["candidate_memory_type"]] += 1
        sensitivity_counter[out["candidate_sensitivity"]] += 1

    total = write_jsonl(output_path, rows)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "label_counter": dict(label_counter),
        "source_dataset_counter": dict(source_dataset_counter),
        "scenario_counter": dict(scenario_counter),
        "memory_type_counter": dict(memory_type_counter),
        "sensitivity_counter": dict(sensitivity_counter),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "version": VERSION,
        "task": "operation_governance_model_ready_conversion",
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "outputs": {},
    }

    for split in ["train", "val", "test"]:
        report["outputs"][split] = convert_split(split, INPUTS[split], OUTPUTS[split])

    report_path = OUT_DIR / "operation_model_ready_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
