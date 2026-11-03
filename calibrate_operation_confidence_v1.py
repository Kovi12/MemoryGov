import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score


LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_tsv(path, rows):
    if not rows:
        return

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


def get_scores(model, encoder, texts):
    if hasattr(model, "decision_function"):
        raw = model.decision_function(texts)
        class_ids = model.classes_
        class_labels = encoder.inverse_transform(class_ids)
        return [
            {str(label): float(score) for label, score in zip(class_labels, row)}
            for row in raw
        ]

    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(texts)
        class_ids = model.classes_
        class_labels = encoder.inverse_transform(class_ids)
        return [
            {str(label): float(score) for label, score in zip(class_labels, row)}
            for row in raw
        ]

    return [{} for _ in texts]


def top_margin(scores):
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return None, None, None
    if len(items) == 1:
        return items[0][0], items[0][1], 999.0
    return items[0][0], items[0][1], items[0][1] - items[1][1]


def evaluate_split(model, encoder, rows, split_name):
    texts = [row.get("input_text") or row.get("text") or "" for row in rows]
    y_true = [row["label"] for row in rows]
    y_pred = list(encoder.inverse_transform(model.predict(texts)))
    scores_list = get_scores(model, encoder, texts)

    pred_rows = []
    for row, true_label, pred_label, scores in zip(rows, y_true, y_pred, scores_list):
        top_label, top_score, margin = top_margin(scores)
        pred_rows.append({
            "split": split_name,
            "example_id": row.get("example_id"),
            "true_label": true_label,
            "pred_label": str(pred_label),
            "correct": bool(true_label == pred_label),
            "top_label": top_label,
            "top_score": top_score,
            "margin": margin,
        })

    return pred_rows


def summarize_at_threshold(pred_rows, threshold):
    auto = [r for r in pred_rows if r["margin"] is not None and r["margin"] >= threshold]
    ask = [r for r in pred_rows if r["margin"] is None or r["margin"] < threshold]

    out = {
        "threshold": threshold,
        "total": len(pred_rows),
        "auto_count": len(auto),
        "ask_count": len(ask),
        "auto_rate": len(auto) / len(pred_rows) if pred_rows else 0.0,
        "ask_rate": len(ask) / len(pred_rows) if pred_rows else 0.0,
        "overall_accuracy": sum(r["correct"] for r in pred_rows) / len(pred_rows) if pred_rows else 0.0,
        "auto_accuracy": sum(r["correct"] for r in auto) / len(auto) if auto else None,
        "ask_error_capture_rate": (
            sum((not r["correct"]) for r in ask) / max(1, sum((not r["correct"]) for r in pred_rows))
        ),
        "auto_error_count": sum((not r["correct"]) for r in auto),
        "ask_error_count": sum((not r["correct"]) for r in ask),
        "ask_true_label_counter": dict(Counter(r["true_label"] for r in ask)),
        "auto_true_label_counter": dict(Counter(r["true_label"] for r in auto)),
    }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="processed/operation_classifier_results/non_llm_v1_2_forget_aug_selected/tfidf_sgd_hinge_balanced/model.joblib",
    )
    parser.add_argument(
        "--data_dir",
        default="processed/memorygovbench_operation_v1_2_model_ready_noleak",
    )
    parser.add_argument(
        "--manual",
        default="processed/manual_gold_operation_v1_1/hand_authored/manual_gold_model_ready.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="processed/operation_confidence_calibration_v1",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(args.model)
    model = payload["model"]
    encoder = payload["label_encoder"]

    split_paths = {
        "val": Path(args.data_dir) / "operation_val_model_ready.jsonl",
        "test_synthetic": Path(args.data_dir) / "operation_test_synthetic_model_ready.jsonl",
        "test_combined": Path(args.data_dir) / "operation_test_combined_model_ready.jsonl",
        "manual_gold": Path(args.manual),
    }

    all_pred_rows = []
    split_summaries = {}

    for split, path in split_paths.items():
        if not path.exists():
            continue
        rows = read_jsonl(path)
        pred_rows = evaluate_split(model, encoder, rows, split)
        all_pred_rows.extend(pred_rows)

        thresholds = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
        threshold_rows = [summarize_at_threshold(pred_rows, t) for t in thresholds]

        write_tsv(output_dir / f"threshold_sweep_{split}.tsv", threshold_rows)

        split_summaries[split] = {
            "path": str(path),
            "total": len(pred_rows),
            "accuracy": accuracy_score([r["true_label"] for r in pred_rows], [r["pred_label"] for r in pred_rows]),
            "errors": sum(not r["correct"] for r in pred_rows),
            "threshold_sweep": str(output_dir / f"threshold_sweep_{split}.tsv"),
        }

    write_tsv(output_dir / "all_predictions_with_margins.tsv", all_pred_rows)

    report = {
        "version": "v3.89",
        "model": args.model,
        "data_dir": args.data_dir,
        "manual": args.manual,
        "note": "Scores from hinge/SVM-style models are margins, not calibrated probabilities. Use top1-top2 margin for low-confidence gating.",
        "splits": split_summaries,
        "recommendation": {
            "initial_margin_threshold": 0.10,
            "high_risk_labels_always_confirm": ["FORGET"],
            "labels_that_require_confirmation_by_policy": ["ASK_USER"],
            "quarantine_policy": "Auto-quarantine may be allowed, but show a visible note in demo mode.",
        },
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
