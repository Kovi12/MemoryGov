import json
import re
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent

INPUTS = {
    "train_full": BASE / "processed/memorygovbench_operation_v1/operation_train_full.jsonl",
    "train_balanced": BASE / "processed/memorygovbench_operation_v1/operation_train_balanced.jsonl",
    "val": BASE / "processed/memorygovbench_operation_v1/operation_val.jsonl",
    "test_benchmark": BASE / "processed/memorygovbench_operation_v1/operation_test_benchmark.jsonl",
    "test_synthetic": BASE / "processed/memorygovbench_operation_v1/operation_test_synthetic.jsonl",
    "test_combined": BASE / "processed/memorygovbench_operation_v1/operation_test_combined.jsonl",
}

OUT_DIR = BASE / "processed/memorygovbench_operation_v1_model_ready_noleak"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS = {
    "train_full": OUT_DIR / "operation_train_full_model_ready.jsonl",
    "train_balanced": OUT_DIR / "operation_train_balanced_model_ready.jsonl",
    "val": OUT_DIR / "operation_val_model_ready.jsonl",
    "test_benchmark": OUT_DIR / "operation_test_benchmark_model_ready.jsonl",
    "test_synthetic": OUT_DIR / "operation_test_synthetic_model_ready.jsonl",
    "test_combined": OUT_DIR / "operation_test_combined_model_ready.jsonl",
}

REPORT = OUT_DIR / "operation_v1_model_ready_noleak_report.json"


LEAK_PATTERNS = [
    r"synthetic_[a-z_]+",
    r"operation_synthetic_generation",
    r"SyntheticAnchored",
    r"benchmark_current_[a-z_]+",
    r"strict_existing_expansion_train_only",
    r"operation_synthetic_[a-z0-9_]+",
]


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value):
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_history(history_context):
    if not history_context:
        return "None"

    parts = []
    for item in history_context:
        if isinstance(item, dict):
            speaker = clean_text(item.get("speaker", "unknown"))
            text = clean_text(item.get("text", ""))
            if text:
                parts.append(f"{speaker}: {text}")
        else:
            text = clean_text(item)
            if text:
                parts.append(text)

    return " | ".join(parts) if parts else "None"


def format_existing_memories(existing_memories):
    if not existing_memories:
        return "None"

    parts = []
    for idx, mem in enumerate(existing_memories, 1):
        if not isinstance(mem, dict):
            continue

        memory_id = clean_text(mem.get("memory_id", f"memory_{idx}"))
        text = clean_text(mem.get("text", ""))
        memory_type = clean_text(mem.get("memory_type", "unknown"))
        sensitivity = clean_text(mem.get("sensitivity", "unknown"))
        status = clean_text(mem.get("status", "active"))

        if text:
            parts.append(
                f"{memory_id}: {text} "
                f"[type={memory_type}; sensitivity={sensitivity}; status={status}]"
            )

    return " | ".join(parts) if parts else "None"


def build_input_text(row, include_metadata=True):
    candidate = row.get("candidate_memory", {}) or {}

    candidate_text = clean_text(candidate.get("text", ""))
    candidate_type = clean_text(candidate.get("memory_type", "unknown"))
    candidate_sensitivity = clean_text(candidate.get("sensitivity", "unknown"))

    existing = format_existing_memories(row.get("existing_memories", []))
    history = format_history(row.get("history_context", []))

    metadata_block = ""
    if include_metadata:
        metadata_block = f"""
Candidate metadata:
type={candidate_type}
sensitivity={candidate_sensitivity}
""".strip()

    text = f"""Task: operation_governance
Question: Decide what the memory system should do with the candidate memory.

Allowed labels:
WRITE, IGNORE, UPDATE, MERGE, FORGET, ASK_USER, QUARANTINE

History context:
{history}

Candidate memory:
{candidate_text}

{metadata_block}

Existing memories:
{existing}

Decision target:
Choose the correct operation label for the candidate memory.
"""

    return text.strip()


def label_of(row):
    return row.get("gold", {}).get("primary_decision")


def convert_row(row):
    label = label_of(row)
    text = build_input_text(row, include_metadata=True)

    metadata = row.get("annotation_metadata") or {}
    candidate = row.get("candidate_memory") or {}

    return {
        "example_id": row.get("example_id"),
        "label": label,
        "target": label,
        "text": text,
        "input_text": text,
        "task_level": row.get("task_level"),
        "source_dataset": row.get("source_dataset"),
        "source_file": row.get("source_file"),
        "scenario_type": row.get("scenario_type"),
        "failure_type": row.get("failure_type"),
        "memory_type": candidate.get("memory_type"),
        "sensitivity": candidate.get("sensitivity"),
        "operation_v1_source": metadata.get("operation_v1_source"),
        "raw": row,
    }


def summarize(rows):
    label_counter = Counter()
    source_counter = Counter()
    scenario_counter = Counter()
    memory_type_counter = Counter()
    sensitivity_counter = Counter()
    operation_v1_source_counter = Counter()
    text_lengths = []
    leak_counter = Counter()

    for row in rows:
        label_counter[row.get("label")] += 1
        source_counter[row.get("source_dataset")] += 1
        scenario_counter[row.get("scenario_type")] += 1
        memory_type_counter[row.get("memory_type")] += 1
        sensitivity_counter[row.get("sensitivity")] += 1
        operation_v1_source_counter[row.get("operation_v1_source")] += 1

        text = row.get("text", "")
        text_lengths.append(len(text))

        for pattern in LEAK_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                leak_counter[pattern] += 1

    text_lengths_sorted = sorted(text_lengths)

    def percentile(p):
        if not text_lengths_sorted:
            return 0
        idx = int(round((len(text_lengths_sorted) - 1) * p))
        return text_lengths_sorted[idx]

    return {
        "total": len(rows),
        "label_counter": dict(label_counter),
        "source_dataset_counter": dict(source_counter),
        "scenario_counter": dict(scenario_counter),
        "memory_type_counter": dict(memory_type_counter),
        "sensitivity_counter": dict(sensitivity_counter),
        "operation_v1_source_counter": dict(operation_v1_source_counter),
        "text_length_chars": {
            "min": min(text_lengths) if text_lengths else 0,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": max(text_lengths) if text_lengths else 0,
        },
        "leak_pattern_counter_in_input_text": dict(leak_counter),
    }


def main():
    report = {
        "version": "v3.65",
        "task": "prepare_operation_v1_classifier_data_noleak",
        "description": "Convert MemoryGovBench-operation-v1 into model-ready text classification format while excluding source/scenario fields that leak labels.",
        "removed_from_input_text": [
            "scenario_type",
            "failure_type",
            "source_dataset",
            "source_file",
            "operation_v1_source",
            "source_type",
            "current_query generic prompt",
            "synthetic_* markers",
        ],
        "outputs": {},
    }

    for split_name, input_path in INPUTS.items():
        raw_rows = read_jsonl(input_path)
        converted = [convert_row(row) for row in raw_rows]

        output_path = OUTPUTS[split_name]
        write_jsonl(output_path, converted)

        report["outputs"][split_name] = {
            "input": str(input_path),
            "output": str(output_path),
            **summarize(converted),
        }

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
