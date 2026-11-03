import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "processed/memorygovbench_current"
OUTPUT_DIR = BASE_DIR / "processed/memorygovbench_model_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

INPUT_FILES = {
    "train": INPUT_DIR / "retrieval_balanced_train.jsonl",
    "val": INPUT_DIR / "retrieval_balanced_val.jsonl",
    "test_balanced": INPUT_DIR / "retrieval_balanced_test.jsonl",
    "test_natural": INPUT_DIR / "retrieval_natural_test.jsonl",
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_input_text(row):
    candidate = row.get("candidate_memory", {})
    memory_text = candidate.get("text", "")
    memory_type = candidate.get("memory_type", "unknown")
    sensitivity = candidate.get("sensitivity", "unknown")
    source_dataset = row.get("source_dataset", "unknown")
    scenario_type = row.get("scenario_type", "unknown")
    current_query = row.get("current_query", "")

    return (
        "Task: Decide whether the candidate memory should be retrieved for the current query.\n"
        f"Source dataset: {source_dataset}\n"
        f"Scenario type: {scenario_type}\n"
        f"Memory type: {memory_type}\n"
        f"Sensitivity: {sensitivity}\n"
        f"Current query: {current_query}\n"
        f"Candidate memory: {memory_text}\n"
        "Decision:"
    )


def convert_file(input_path, output_path):
    rows = []
    label_counter = {}

    for row in read_jsonl(input_path):
        label = row.get("gold", {}).get("primary_decision")
        if label not in {"RETRIEVE", "SUPPRESS"}:
            continue

        label_counter[label] = label_counter.get(label, 0) + 1

        rows.append({
            "id": row.get("example_id"),
            "input_text": build_input_text(row),
            "label": label,
            "label_id": 1 if label == "RETRIEVE" else 0,
            "source_dataset": row.get("source_dataset"),
            "source_file": row.get("source_file"),
            "scenario_type": row.get("scenario_type"),
            "memory_type": row.get("candidate_memory", {}).get("memory_type"),
            "sensitivity": row.get("candidate_memory", {}).get("sensitivity"),
            "current_query": row.get("current_query"),
            "candidate_memory": row.get("candidate_memory", {}).get("text"),
            "expected_answer": row.get("expected_answer"),
        })

    write_jsonl(output_path, rows)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "total": len(rows),
        "label_counter": label_counter,
    }


def main():
    report = {
        "version": "v3.32",
        "task": "retrieval_governance_model_ready_conversion",
        "outputs": {},
    }

    for split_name, input_path in INPUT_FILES.items():
        output_path = OUTPUT_DIR / f"retrieval_{split_name}_model_ready.jsonl"
        report["outputs"][split_name] = convert_file(input_path, output_path)

    report_path = OUTPUT_DIR / "retrieval_model_ready_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
