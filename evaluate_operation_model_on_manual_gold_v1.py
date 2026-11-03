import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]
LIFECYCLE_LABELS = ["IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def supported_macro_f1(per_label):
    vals = []
    for label in LABELS:
        info = per_label.get(label, {})
        if int(info.get("support", 0) or 0) > 0:
            vals.append(float(info.get("f1", 0.0) or 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def supported_balanced_accuracy(per_label):
    vals = []
    for label in LABELS:
        info = per_label.get(label, {})
        if int(info.get("support", 0) or 0) > 0:
            vals.append(float(info.get("recall", 0.0) or 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def lifecycle_macro_f1(per_label):
    vals = []
    for label in LIFECYCLE_LABELS:
        info = per_label.get(label, {})
        if int(info.get("support", 0) or 0) > 0:
            vals.append(float(info.get("f1", 0.0) or 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def extract_candidate(row):
    raw = row.get("raw") or {}
    candidate = raw.get("candidate_memory") or {}
    return candidate.get("text", "")


def extract_existing(row):
    raw = row.get("raw") or {}
    existing = []
    for mem in raw.get("existing_memories", []) or []:
        if isinstance(mem, dict):
            existing.append(mem.get("text", ""))
        else:
            existing.append(str(mem))
    return existing


def extract_history(row):
    raw = row.get("raw") or {}
    history = []
    for item in raw.get("history_context", []) or []:
        if isinstance(item, dict):
            history.append(f"{item.get('speaker', 'unknown')}: {item.get('text', '')}")
        else:
            history.append(str(item))
    return history


def predict_scores(model, texts, encoder):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(texts)
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                classes = getattr(model.named_steps.get("clf"), "classes_", None)

            decoded = encoder.inverse_transform(classes)
            return [
                {str(decoded[j]): float(probs[i][j]) for j in range(len(decoded))}
                for i in range(len(texts))
            ]

        if hasattr(model, "decision_function"):
            scores = model.decision_function(texts)
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                classes = getattr(model.named_steps.get("clf"), "classes_", None)

            decoded = encoder.inverse_transform(classes)

            if len(scores.shape) == 1:
                scores = [[-s, s] for s in scores]

            return [
                {str(decoded[j]): float(scores[i][j]) for j in range(len(decoded))}
                for i in range(len(texts))
            ]
    except Exception:
        return [{} for _ in texts]

    return [{} for _ in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="processed/operation_classifier_results/non_llm_v1_1_balanced_noleak_all/tfidf_sgd_logloss_balanced/model.joblib",
    )
    parser.add_argument(
        "--input",
        default="processed/manual_gold_operation_v1_1/hand_authored/manual_gold_model_ready.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="processed/manual_gold_operation_v1_1/hand_authored/evaluation_tfidf_sgd_logloss",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(args.model)
    model = payload["model"]
    encoder = payload["label_encoder"]

    rows = read_jsonl(args.input)
    texts = [row.get("input_text") or row.get("text") or "" for row in rows]
    y_true_labels = [row["label"] for row in rows]

    y_true = encoder.transform(y_true_labels)
    y_pred = model.predict(texts)
    y_pred_labels = encoder.inverse_transform(y_pred)

    label_ids = encoder.transform(LABELS)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )

    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    score_rows = predict_scores(model, texts, encoder)

    predictions = []
    for row, pred_label, scores in zip(rows, y_pred_labels, score_rows):
        true_label = row["label"]
        predictions.append({
            "example_id": row.get("example_id"),
            "true_label": true_label,
            "pred_label": str(pred_label),
            "correct": bool(true_label == pred_label),
            "memory_type": row.get("memory_type"),
            "sensitivity": row.get("sensitivity"),
            "candidate_memory": extract_candidate(row),
            "existing_memories": extract_existing(row),
            "history_context": extract_history(row),
            "manual_reason": row.get("manual_gold", {}).get("manual_reason"),
            "review_status": row.get("manual_gold", {}).get("authoring_status"),
            "scores": scores,
            "text": row.get("input_text") or row.get("text"),
        })

    errors = [row for row in predictions if not row["correct"]]

    confusion = confusion_matrix(y_true, y_pred, labels=label_ids).tolist()

    confusion_rows = []
    for i, true_label in enumerate(LABELS):
        row = {"true_label": true_label}
        for j, pred_label in enumerate(LABELS):
            row[pred_label] = confusion[i][j]
        confusion_rows.append(row)

    per_label_rows = []
    for label, info in per_label.items():
        per_label_rows.append({
            "label": label,
            **info,
        })

    report = {
        "version": "v3.82",
        "task": "evaluate_operation_model_on_manual_gold_v1",
        "model": args.model,
        "input": args.input,
        "output_dir": str(output_dir),
        "total": len(rows),
        "errors": len(errors),
        "label_counter_true": dict(Counter(y_true_labels)),
        "label_counter_pred": dict(Counter(str(x) for x in y_pred_labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_all_labels": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
        "supported_macro_f1": float(supported_macro_f1(per_label)),
        "supported_balanced_accuracy": float(supported_balanced_accuracy(per_label)),
        "lifecycle_macro_f1": float(lifecycle_macro_f1(per_label)),
        "per_label": per_label,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "outputs": {
            "predictions": str(output_dir / "predictions.jsonl"),
            "errors": str(output_dir / "errors.jsonl"),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
            "per_label_metrics_csv": str(output_dir / "per_label_metrics.csv"),
            "report": str(output_dir / "report.json"),
        },
        "note": "This manual gold set is AI-assisted and should be user-reviewed before being described as human-curated.",
    }

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "errors.jsonl", errors)
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    write_csv(output_dir / "per_label_metrics.csv", per_label_rows)

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
