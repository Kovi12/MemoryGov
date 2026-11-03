import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent

IN_DIR = BASE / "processed/memorygovbench_operation_v1"
OUT_DIR = BASE / "processed/memorygovbench_operation_v1_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_IDS_PATH = BASE / "processed/operation_label_audit_v1/exclude_example_ids.txt"

SPLITS = {
    "operation_train_full": "operation_train_full.jsonl",
    "operation_train_balanced": "operation_train_balanced.jsonl",
    "operation_val": "operation_val.jsonl",
    "operation_test_benchmark": "operation_test_benchmark.jsonl",
    "operation_test_synthetic": "operation_test_synthetic.jsonl",
    "operation_test_combined": "operation_test_combined.jsonl",
}

OUT_REPORT = OUT_DIR / "report.json"


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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_of(row):
    return row.get("gold", {}).get("primary_decision")


def summarize(rows):
    return {
        "total": len(rows),
        "label_counter": dict(Counter(label_of(row) for row in rows)),
        "source_dataset_counter": dict(Counter(row.get("source_dataset") for row in rows)),
        "scenario_counter": dict(Counter(row.get("scenario_type") for row in rows)),
    }


def main():
    exclude_ids = set()
    if EXCLUDE_IDS_PATH.exists():
        exclude_ids = set(x.strip() for x in EXCLUDE_IDS_PATH.read_text(encoding="utf-8").splitlines() if x.strip())

    report = {
        "version": "v3.69",
        "dataset_name": "MemoryGovBench-operation-v1.1",
        "purpose": "Operation-v1 with audited noisy LongMemEval IGNORE examples excluded.",
        "exclude_ids_path": str(EXCLUDE_IDS_PATH),
        "unique_excluded_ids": len(exclude_ids),
        "splits": {},
    }

    for split_name, filename in SPLITS.items():
        in_path = IN_DIR / filename
        out_path = OUT_DIR / filename

        rows = read_jsonl(in_path)
        kept = []
        removed = []

        for row in rows:
            if row.get("example_id") in exclude_ids:
                removed.append(row)
            else:
                kept.append(row)

        write_jsonl(out_path, kept)

        report["splits"][split_name] = {
            "input": str(in_path),
            "output": str(out_path),
            "before": summarize(rows),
            "after": summarize(kept),
            "removed": summarize(removed),
        }

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
