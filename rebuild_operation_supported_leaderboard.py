import argparse
import csv
import json
from pathlib import Path


LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]
SPLITS = ["val", "test_benchmark", "test_synthetic", "test_combined"]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_tsv(path, rows):
    keys = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def supported_macro_f1(split_metrics):
    vals = []

    for label in LABELS:
        info = split_metrics.get("per_label", {}).get(label, {})
        support = int(info.get("support", 0) or 0)
        if support > 0:
            vals.append(float(info.get("f1", 0.0) or 0.0))

    return sum(vals) / len(vals) if vals else 0.0


def supported_balanced_accuracy(split_metrics):
    vals = []

    for label in LABELS:
        info = split_metrics.get("per_label", {}).get(label, {})
        support = int(info.get("support", 0) or 0)
        if support > 0:
            vals.append(float(info.get("recall", 0.0) or 0.0))

    return sum(vals) / len(vals) if vals else 0.0


def lifecycle_macro_f1(split_metrics):
    vals = []

    for label in ["IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]:
        info = split_metrics.get("per_label", {}).get(label, {})
        support = int(info.get("support", 0) or 0)
        if support > 0:
            vals.append(float(info.get("f1", 0.0) or 0.0))

    return sum(vals) / len(vals) if vals else 0.0


def collect(results_dir):
    rows = []

    for metrics_path in sorted(Path(results_dir).glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        model_name = metrics.get("model_name") or metrics_path.parent.name

        row = {
            "model": model_name,
            "train_seconds": metrics.get("train_seconds"),
            "metrics_file": str(metrics_path),
        }

        split_metrics_all = metrics.get("metrics", {})

        for split in SPLITS:
            split_metrics = split_metrics_all.get(split, {})
            row[f"{split}_accuracy"] = split_metrics.get("accuracy")
            row[f"{split}_weighted_f1"] = split_metrics.get("weighted_f1")
            row[f"{split}_macro_f1_all_labels"] = split_metrics.get("macro_f1")
            row[f"{split}_supported_macro_f1"] = supported_macro_f1(split_metrics)
            row[f"{split}_supported_balanced_accuracy"] = supported_balanced_accuracy(split_metrics)
            row[f"{split}_lifecycle_macro_f1"] = lifecycle_macro_f1(split_metrics)

            for label in LABELS:
                info = split_metrics.get("per_label", {}).get(label, {})
                row[f"{split}_{label}_support"] = info.get("support", 0)
                row[f"{split}_{label}_f1"] = info.get("f1", 0.0)
                row[f"{split}_{label}_recall"] = info.get("recall", 0.0)
                row[f"{split}_{label}_precision"] = info.get("precision", 0.0)

        rows.append(row)

    rows.sort(
        key=lambda r: (
            float(r.get("val_supported_macro_f1", 0.0) or 0.0),
            float(r.get("test_synthetic_supported_macro_f1", 0.0) or 0.0),
            float(r.get("test_combined_supported_macro_f1", 0.0) or 0.0),
        ),
        reverse=True,
    )

    for i, row in enumerate(rows, 1):
        row["rank"] = i

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = collect(args.results_dir)

    output = args.output
    if not output:
        output = str(Path(args.results_dir) / "leaderboard_supported_metrics.tsv")

    write_tsv(output, rows)

    print(json.dumps({
        "results_dir": args.results_dir,
        "output": output,
        "models": len(rows),
        "best_model": rows[0]["model"] if rows else None,
        "best_val_supported_macro_f1": rows[0]["val_supported_macro_f1"] if rows else None,
        "best_test_synthetic_supported_macro_f1": rows[0]["test_synthetic_supported_macro_f1"] if rows else None,
        "best_test_benchmark_supported_macro_f1": rows[0]["test_benchmark_supported_macro_f1"] if rows else None,
        "best_test_combined_supported_macro_f1": rows[0]["test_combined_supported_macro_f1"] if rows else None,
    }, indent=2))


if __name__ == "__main__":
    main()
