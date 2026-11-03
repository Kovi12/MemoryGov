import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent

IN_DIR = BASE / "processed/memorygovbench_operation_v1_1_model_ready_noleak"
AUG_PATH = BASE / "processed/operation_forget_targeted_aug_v1/forget_targeted_aug_train_only.jsonl"
OUT_DIR = BASE / "processed/memorygovbench_operation_v1_2_model_ready_noleak"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_full": "operation_train_full_model_ready.jsonl",
    "train_balanced": "operation_train_balanced_model_ready.jsonl",
    "val": "operation_val_model_ready.jsonl",
    "test_benchmark": "operation_test_benchmark_model_ready.jsonl",
    "test_synthetic": "operation_test_synthetic_model_ready.jsonl",
    "test_combined": "operation_test_combined_model_ready.jsonl",
}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows):
    return {
        "total": len(rows),
        "label_counter": dict(Counter(row.get("label") for row in rows)),
        "source_dataset_counter": dict(Counter(row.get("source_dataset") for row in rows)),
        "scenario_counter": dict(Counter(row.get("scenario_type") for row in rows)),
        "sensitivity_counter": dict(Counter(row.get("sensitivity") for row in rows)),
    }


def main():
    aug_rows = read_jsonl(AUG_PATH)

    report = {
        "version": "v3.85",
        "task": "build_operation_v1_2_model_ready_from_v1_1",
        "purpose": "Create operation v1.2 model-ready data by adding train-only targeted FORGET augmentation to v1.1.",
        "augmentation": {
            "path": str(AUG_PATH),
            **summarize(aug_rows),
        },
        "splits": {},
    }

    for split, filename in FILES.items():
        rows = read_jsonl(IN_DIR / filename)
        before = summarize(rows)

        if split in {"train_balanced", "train_full"}:
            out_rows = rows + aug_rows
        else:
            out_rows = rows

        write_jsonl(OUT_DIR / filename, out_rows)

        report["splits"][split] = {
            "input": str(IN_DIR / filename),
            "output": str(OUT_DIR / filename),
            "before": before,
            "after": summarize(out_rows),
            "augmentation_added": len(aug_rows) if split in {"train_balanced", "train_full"} else 0,
        }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
