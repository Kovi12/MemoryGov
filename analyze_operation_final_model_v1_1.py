import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent

MODEL_NAME = "tfidf_sgd_logloss_balanced"

RESULTS_DIR = BASE / "processed/operation_classifier_results/non_llm_v1_1_balanced_noleak_all"
MODEL_DIR = RESULTS_DIR / MODEL_NAME
METRICS_PATH = MODEL_DIR / "metrics.json"

DATA_DIR = BASE / "processed/memorygovbench_operation_v1_1_model_ready_noleak"

SPLIT_FILES = {
    "val": DATA_DIR / "operation_val_model_ready.jsonl",
    "test_benchmark": DATA_DIR / "operation_test_benchmark_model_ready.jsonl",
    "test_synthetic": DATA_DIR / "operation_test_synthetic_model_ready.jsonl",
    "test_combined": DATA_DIR / "operation_test_combined_model_ready.jsonl",
}

PREDICTION_FILES = {
    "val": MODEL_DIR / "predictions_val.jsonl",
    "test_benchmark": MODEL_DIR / "predictions_test_benchmark.jsonl",
    "test_synthetic": MODEL_DIR / "predictions_test_synthetic.jsonl",
    "test_combined": MODEL_DIR / "predictions_test_combined.jsonl",
}

OUT_DIR = BASE / "processed/operation_final_analysis_v1_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
ERROR_DIR = OUT_DIR / "errors"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]
LIFECYCLE_LABELS = ["IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows, fieldnames=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_data_by_id(path):
    rows = read_jsonl(path)
    return {row["example_id"]: row for row in rows}


def safe_score(pred, label):
    scores = pred.get("scores") or {}
    value = scores.get(label)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def make_joined_rows(split):
    data_by_id = load_data_by_id(SPLIT_FILES[split])
    preds = read_jsonl(PREDICTION_FILES[split])

    joined = []
    for pred in preds:
        example_id = pred.get("example_id")
        data = data_by_id.get(example_id, {})

        raw = data.get("raw") or {}
        candidate = raw.get("candidate_memory") or {}
        gold = raw.get("gold") or {}

        joined.append({
            "example_id": example_id,
            "split": split,
            "true_label": pred.get("true_label"),
            "pred_label": pred.get("pred_label"),
            "correct": bool(pred.get("correct")),
            "source_dataset": data.get("source_dataset"),
            "scenario_type": data.get("scenario_type"),
            "operation_v1_source": data.get("operation_v1_source"),
            "memory_type": data.get("memory_type"),
            "sensitivity": data.get("sensitivity"),
            "candidate_memory": candidate.get("text"),
            "existing_memories": [
                m.get("text") for m in raw.get("existing_memories", []) if isinstance(m, dict)
            ],
            "history_context": [
                f"{m.get('speaker', 'unknown')}: {m.get('text', '')}"
                for m in raw.get("history_context", []) if isinstance(m, dict)
            ],
            "gold_reason": gold.get("reason"),
            "input_text": data.get("input_text") or data.get("text"),
            "scores": pred.get("scores"),
            "score_true": safe_score(pred, pred.get("true_label")),
            "score_pred": safe_score(pred, pred.get("pred_label")),
            "top_score": pred.get("top_score"),
        })

    return joined


def build_confusion_counts(rows):
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["true_label"]][row["pred_label"]] += 1
    return counts


def save_confusion_csv(split, rows):
    counts = build_confusion_counts(rows)

    matrix_rows = []
    for true_label in LABELS:
        row = {"true_label": true_label}
        for pred_label in LABELS:
            row[pred_label] = counts[true_label][pred_label]
        matrix_rows.append(row)

    write_csv(TABLE_DIR / f"confusion_matrix_{split}.csv", matrix_rows)


def save_confusion_plot(split, rows):
    counts = build_confusion_counts(rows)
    matrix = [[counts[t][p] for p in LABELS] for t in LABELS]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix)

    ax.set_title(f"Confusion matrix: {MODEL_NAME} on {split}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = matrix[i][j]
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_matrix_{split}.png", dpi=180)
    plt.close(fig)


def save_per_label_f1_plot(split, metrics):
    per_label = metrics.get("per_label", {})
    labels = []
    values = []

    for label in LABELS:
        support = per_label.get(label, {}).get("support", 0)
        if support > 0:
            labels.append(label)
            values.append(per_label.get(label, {}).get("f1", 0.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-label F1: {MODEL_NAME} on {split}")
    ax.set_ylabel("F1")
    ax.set_xticklabels(labels, rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"per_label_f1_{split}.png", dpi=180)
    plt.close(fig)


def summarize_errors(split, rows):
    errors = [row for row in rows if not row["correct"]]
    confusion_counter = Counter((row["true_label"], row["pred_label"]) for row in errors)

    confusion_rows = []
    for (true_label, pred_label), count in confusion_counter.most_common():
        confusion_rows.append({
            "split": split,
            "true_label": true_label,
            "pred_label": pred_label,
            "count": count,
        })

    return errors, confusion_rows


def make_error_previews(errors, max_per_confusion=10):
    by_confusion = defaultdict(list)

    for row in errors:
        key = f"{row['true_label']}->{row['pred_label']}"
        if len(by_confusion[key]) < max_per_confusion:
            by_confusion[key].append({
                "example_id": row["example_id"],
                "split": row["split"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "source_dataset": row["source_dataset"],
                "scenario_type": row["scenario_type"],
                "memory_type": row["memory_type"],
                "sensitivity": row["sensitivity"],
                "candidate_memory": row["candidate_memory"],
                "existing_memories": row["existing_memories"],
                "history_context": row["history_context"],
                "gold_reason": row["gold_reason"],
                "score_true": row["score_true"],
                "score_pred": row["score_pred"],
                "top_score": row["top_score"],
            })

    return by_confusion


def collect_per_label_metrics(metrics):
    out = []

    for split, split_metrics in metrics.get("metrics", {}).items():
        per_label = split_metrics.get("per_label", {})
        for label in LABELS:
            info = per_label.get(label, {})
            out.append({
                "model": MODEL_NAME,
                "split": split,
                "label": label,
                "support": info.get("support", 0),
                "precision": info.get("precision", 0.0),
                "recall": info.get("recall", 0.0),
                "f1": info.get("f1", 0.0),
                "accuracy": split_metrics.get("accuracy"),
                "supported_macro_f1": split_metrics.get("supported_macro_f1"),
                "lifecycle_macro_f1": split_metrics.get("lifecycle_macro_f1"),
                "weighted_f1": split_metrics.get("weighted_f1"),
            })

    return out


def main():
    metrics = read_json(METRICS_PATH)

    write_csv(
        TABLE_DIR / "per_label_metrics.csv",
        collect_per_label_metrics(metrics),
    )

    all_error_confusions = []
    summary = {
        "version": "v3.76",
        "task": "operation_final_model_error_analysis",
        "model": MODEL_NAME,
        "metrics_path": str(METRICS_PATH),
        "splits": {},
    }

    for split in ["val", "test_benchmark", "test_synthetic", "test_combined"]:
        rows = make_joined_rows(split)
        errors, confusion_rows = summarize_errors(split, rows)
        previews = make_error_previews(errors)

        save_confusion_csv(split, rows)
        save_confusion_plot(split, rows)
        save_per_label_f1_plot(split, metrics["metrics"][split])

        write_jsonl(ERROR_DIR / f"errors_{split}.jsonl", errors)
        write_json(ERROR_DIR / f"error_previews_by_confusion_{split}.json", previews)

        all_error_confusions.extend(confusion_rows)

        label_support = Counter(row["true_label"] for row in rows)
        label_errors = Counter(row["true_label"] for row in errors)

        per_true_label_error_rate = {}
        for label in LABELS:
            total = label_support.get(label, 0)
            wrong = label_errors.get(label, 0)
            if total:
                per_true_label_error_rate[label] = {
                    "errors": wrong,
                    "support": total,
                    "error_rate": wrong / total,
                }

        summary["splits"][split] = {
            "total": len(rows),
            "correct": len(rows) - len(errors),
            "errors": len(errors),
            "accuracy_from_predictions": (len(rows) - len(errors)) / len(rows) if rows else 0.0,
            "label_support": dict(label_support),
            "label_errors": dict(label_errors),
            "per_true_label_error_rate": per_true_label_error_rate,
            "top_confusions": confusion_rows[:20],
            "metrics": {
                "accuracy": metrics["metrics"][split].get("accuracy"),
                "supported_macro_f1": metrics["metrics"][split].get("supported_macro_f1"),
                "lifecycle_macro_f1": metrics["metrics"][split].get("lifecycle_macro_f1"),
                "weighted_f1": metrics["metrics"][split].get("weighted_f1"),
            },
        }

    write_csv(TABLE_DIR / "top_confusions_all_splits.csv", all_error_confusions)
    write_json(OUT_DIR / "final_model_error_analysis_report.json", summary)

    print(json.dumps({
        "output_dir": str(OUT_DIR),
        "report": str(OUT_DIR / "final_model_error_analysis_report.json"),
        "tables": str(TABLE_DIR),
        "figures": str(FIG_DIR),
        "errors": str(ERROR_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
