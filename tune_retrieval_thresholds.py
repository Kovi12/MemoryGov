import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


VERSION = "v3.47"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_label(value, positive_label, negative_label):
    if value is None:
        return None

    if isinstance(value, bool):
        return positive_label if value else negative_label

    if isinstance(value, int):
        return positive_label if value == 1 else negative_label

    value = str(value).strip()

    if value in {positive_label, negative_label}:
        return value

    upper = value.upper()

    if upper in {"1", "TRUE", "YES", "RETRIEVE", "WRITE"}:
        return positive_label

    if upper in {"0", "FALSE", "NO", "SUPPRESS", "IGNORE"}:
        return negative_label

    return value


def get_first_present(row, keys):
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def extract_true_label(row, positive_label, negative_label):
    value = get_first_present(
        row,
        [
            "true_label",
            "gold_label",
            "label",
            "target",
            "y_true",
            "gold",
            "decision",
            "primary_decision",
        ],
    )

    if isinstance(value, dict):
        value = value.get("primary_decision") or value.get("label") or value.get("decision")

    return normalize_label(value, positive_label, negative_label)


def extract_pred_label(row, positive_label, negative_label):
    value = get_first_present(
        row,
        [
            "pred_label",
            "predicted_label",
            "prediction",
            "y_pred",
            "model_prediction",
            "default_pred_label",
        ],
    )
    return normalize_label(value, positive_label, negative_label)


def value_to_float(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)

    try:
        value = str(value).strip()
        if not value:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def extract_score(row, positive_label, negative_label):
    direct_keys = [
        "positive_score",
        "positive_probability",
        "probability",
        "score",
        "retrieve_score",
        "retrieve_probability",
        "prob_RETRIEVE",
        "proba_RETRIEVE",
        "RETRIEVE_score",
        "RETRIEVE_probability",
        "decision_score",
        "confidence",
    ]

    for key in direct_keys:
        if key in row:
            score = value_to_float(row.get(key))
            if score is not None:
                return score

    dict_keys = [
        "scores",
        "probabilities",
        "probs",
        "label_scores",
        "class_probabilities",
        "decision_scores",
    ]

    for key in dict_keys:
        obj = row.get(key)
        if isinstance(obj, dict):
            for label_key in [positive_label, positive_label.lower(), positive_label.upper(), "positive", "1"]:
                if label_key in obj:
                    score = value_to_float(obj.get(label_key))
                    if score is not None:
                        return score

    pred = extract_pred_label(row, positive_label, negative_label)
    if pred == positive_label:
        return 1.0
    if pred == negative_label:
        return 0.0

    return None


def load_prediction_split(path, positive_label, negative_label):
    rows = read_jsonl(path)
    y_true = []
    scores = []
    usable_rows = []

    for row in rows:
        true_label = extract_true_label(row, positive_label, negative_label)
        score = extract_score(row, positive_label, negative_label)

        if true_label not in {positive_label, negative_label}:
            continue

        if score is None:
            continue

        y_true.append(1 if true_label == positive_label else 0)
        scores.append(score)
        usable_rows.append(row)

    return usable_rows, np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)


def metric_bundle(y_true, scores, threshold, positive_label, negative_label):
    y_pred = (scores >= threshold).astype(int)

    labels_binary = [1, 0]
    target_names = [positive_label, negative_label]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision, recall, pos_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_binary)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_binary,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y_true, scores) if len(set(y_true.tolist())) == 2 else None
    except Exception:
        roc_auc = None

    try:
        avg_precision = average_precision_score(y_true, scores) if len(set(y_true.tolist())) == 2 else None
    except Exception:
        avg_precision = None

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "positive_label": positive_label,
        "positive_precision": float(precision[0]),
        "positive_recall": float(recall[0]),
        "positive_f1": float(pos_f1[0]),
        "confusion_matrix_labels": [positive_label, negative_label],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_counter_true": {
            positive_label: int((y_true == 1).sum()),
            negative_label: int((y_true == 0).sum()),
        },
        "label_counter_pred": {
            positive_label: int((y_pred == 1).sum()),
            negative_label: int((y_pred == 0).sum()),
        },
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "average_precision": None if avg_precision is None else float(avg_precision),
    }


def get_metric_value(metrics, metric_name):
    value = metrics.get(metric_name)
    if value is None:
        return -1.0
    return float(value)


def build_threshold_candidates(scores, grid_size):
    grid = np.linspace(0.0, 1.0, grid_size)
    unique_scores = np.unique(scores)

    if len(unique_scores) > 5000:
        percentiles = np.linspace(0, 100, 1001)
        unique_scores = np.percentile(unique_scores, percentiles)

    candidates = np.unique(np.concatenate([grid, unique_scores]))
    candidates = candidates[np.isfinite(candidates)]
    return candidates


def find_best_threshold(y_true, scores, selection_metric, positive_label, negative_label, grid_size):
    best_threshold = 0.5
    best_metrics = metric_bundle(y_true, scores, best_threshold, positive_label, negative_label)
    best_key = (
        get_metric_value(best_metrics, selection_metric),
        best_metrics["positive_f1"],
        best_metrics["macro_f1"],
        best_metrics["accuracy"],
    )

    for threshold in build_threshold_candidates(scores, grid_size):
        metrics = metric_bundle(y_true, scores, threshold, positive_label, negative_label)
        key = (
            get_metric_value(metrics, selection_metric),
            metrics["positive_f1"],
            metrics["macro_f1"],
            metrics["accuracy"],
        )

        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def save_thresholded_predictions(path, rows, y_true, scores, threshold, positive_label, negative_label):
    output_rows = []

    for row, y, score in zip(rows, y_true, scores):
        pred_label = positive_label if score >= threshold else negative_label
        true_label = positive_label if y == 1 else negative_label

        out = dict(row)
        out["true_label_normalized"] = true_label
        out["thresholded_pred_label"] = pred_label
        out["positive_score"] = float(score)
        out["threshold"] = float(threshold)
        out["thresholded_correct"] = pred_label == true_label
        output_rows.append(out)

    write_jsonl(path, output_rows)


def discover_model_dirs(results_dir):
    results_dir = Path(results_dir)
    model_dirs = []

    for val_path in results_dir.rglob("predictions_val.jsonl"):
        model_dir = val_path.parent
        test_balanced = model_dir / "predictions_test_balanced.jsonl"
        test_natural = model_dir / "predictions_test_natural.jsonl"

        if test_balanced.exists() and test_natural.exists():
            model_dirs.append(model_dir)

    return sorted(model_dirs)


def model_name_from_dir(results_dir, model_dir):
    rel = model_dir.relative_to(results_dir)
    return "__".join(rel.parts)


def process_model_dir(
    results_dir,
    output_dir,
    model_dir,
    selection_metric,
    positive_label,
    negative_label,
    grid_size,
):
    model_name = model_name_from_dir(results_dir, model_dir)

    split_paths = {
        "val": model_dir / "predictions_val.jsonl",
        "test_balanced": model_dir / "predictions_test_balanced.jsonl",
        "test_natural": model_dir / "predictions_test_natural.jsonl",
    }

    split_data = {}
    for split_name, split_path in split_paths.items():
        rows, y_true, scores = load_prediction_split(split_path, positive_label, negative_label)
        split_data[split_name] = {
            "path": str(split_path),
            "rows": rows,
            "y_true": y_true,
            "scores": scores,
            "total_loaded": len(read_jsonl(split_path)),
            "total_usable": len(rows),
        }

    if len(split_data["val"]["rows"]) == 0:
        return {
            "model_name": model_name,
            "status": "skipped",
            "reason": "no usable validation predictions",
        }

    threshold, val_metrics = find_best_threshold(
        split_data["val"]["y_true"],
        split_data["val"]["scores"],
        selection_metric,
        positive_label,
        negative_label,
        grid_size,
    )

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_name": model_name,
        "source_model_dir": str(model_dir),
        "selection_metric": selection_metric,
        "best_threshold": float(threshold),
        "splits": {},
        "metrics": {},
    }

    for split_name, data in split_data.items():
        split_metrics = metric_bundle(
            data["y_true"],
            data["scores"],
            threshold,
            positive_label,
            negative_label,
        )

        metrics["splits"][split_name] = {
            "input": data["path"],
            "total_loaded": data["total_loaded"],
            "total_usable": data["total_usable"],
            "label_counter": {
                positive_label: int((data["y_true"] == 1).sum()),
                negative_label: int((data["y_true"] == 0).sum()),
            },
        }
        metrics["metrics"][split_name] = split_metrics

        save_thresholded_predictions(
            model_output_dir / f"predictions_{split_name}_thresholded.jsonl",
            data["rows"],
            data["y_true"],
            data["scores"],
            threshold,
            positive_label,
            negative_label,
        )

    write_json(model_output_dir / "threshold_metrics.json", metrics)

    return {
        "model_name": model_name,
        "status": "ok",
        "best_threshold": float(threshold),
        "val_score": metrics["metrics"]["val"].get(selection_metric),
        "metrics": metrics,
    }


def make_leaderboard(results, selection_metric):
    rows = []

    for item in results:
        if item.get("status") != "ok":
            continue

        metrics = item["metrics"]["metrics"]
        row = {
            "model": item["model_name"],
            "threshold": item["best_threshold"],
            "val_selection_score": metrics["val"].get(selection_metric),
            "val_positive_f1": metrics["val"].get("positive_f1"),
            "val_macro_f1": metrics["val"].get("macro_f1"),
            "val_positive_precision": metrics["val"].get("positive_precision"),
            "val_positive_recall": metrics["val"].get("positive_recall"),
            "val_roc_auc": metrics["val"].get("roc_auc"),
            "val_average_precision": metrics["val"].get("average_precision"),
            "test_balanced_positive_f1": metrics["test_balanced"].get("positive_f1"),
            "test_balanced_macro_f1": metrics["test_balanced"].get("macro_f1"),
            "test_balanced_positive_precision": metrics["test_balanced"].get("positive_precision"),
            "test_balanced_positive_recall": metrics["test_balanced"].get("positive_recall"),
            "test_balanced_roc_auc": metrics["test_balanced"].get("roc_auc"),
            "test_balanced_average_precision": metrics["test_balanced"].get("average_precision"),
            "test_natural_positive_f1": metrics["test_natural"].get("positive_f1"),
            "test_natural_macro_f1": metrics["test_natural"].get("macro_f1"),
            "test_natural_positive_precision": metrics["test_natural"].get("positive_precision"),
            "test_natural_positive_recall": metrics["test_natural"].get("positive_recall"),
            "test_natural_roc_auc": metrics["test_natural"].get("roc_auc"),
            "test_natural_average_precision": metrics["test_natural"].get("average_precision"),
            "test_natural_accuracy": metrics["test_natural"].get("accuracy"),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["val_selection_score"] if r["val_selection_score"] is not None else -1,
            r["val_positive_f1"] if r["val_positive_f1"] is not None else -1,
            r["test_balanced_positive_f1"] if r["test_balanced_positive_f1"] is not None else -1,
        ),
        reverse=True,
    )

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return rows


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "rank",
        "model",
        "threshold",
        "val_selection_score",
        "val_positive_f1",
        "val_macro_f1",
        "val_positive_precision",
        "val_positive_recall",
        "val_roc_auc",
        "val_average_precision",
        "test_balanced_positive_f1",
        "test_balanced_macro_f1",
        "test_balanced_positive_precision",
        "test_balanced_positive_recall",
        "test_balanced_roc_auc",
        "test_balanced_average_precision",
        "test_natural_positive_f1",
        "test_natural_macro_f1",
        "test_natural_positive_precision",
        "test_natural_positive_recall",
        "test_natural_roc_auc",
        "test_natural_average_precision",
        "test_natural_accuracy",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col)
                if isinstance(value, float):
                    value = f"{value:.6f}"
                elif value is None:
                    value = ""
                else:
                    value = str(value)
                values.append(value)
            f.write("\t".join(values) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_metric", default="positive_f1")
    parser.add_argument("--positive_label", default="RETRIEVE")
    parser.add_argument("--negative_label", default="SUPPRESS")
    parser.add_argument("--grid_size", type=int, default=1001)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = discover_model_dirs(results_dir)

    results = []
    for model_dir in model_dirs:
        model_name = model_name_from_dir(results_dir, model_dir)
        print(f"Processing {model_name}", flush=True)
        result = process_model_dir(
            results_dir=results_dir,
            output_dir=output_dir,
            model_dir=model_dir,
            selection_metric=args.selection_metric,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            grid_size=args.grid_size,
        )
        results.append(result)

    leaderboard = make_leaderboard(results, args.selection_metric)

    write_tsv(output_dir / "threshold_tuned_leaderboard.tsv", leaderboard)
    write_jsonl(output_dir / "threshold_tuned_model_comparison.jsonl", results)

    status_counter = Counter(item.get("status", "unknown") for item in results)

    report = {
        "version": VERSION,
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "selection_metric": args.selection_metric,
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "models_discovered": len(model_dirs),
        "status_counter": dict(status_counter),
        "best_model": leaderboard[0]["model"] if leaderboard else None,
        "best_score": leaderboard[0]["val_selection_score"] if leaderboard else None,
        "leaderboard": str(output_dir / "threshold_tuned_leaderboard.tsv"),
        "model_comparison": str(output_dir / "threshold_tuned_model_comparison.jsonl"),
    }

    write_json(output_dir / "threshold_tuning_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
