import csv
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


BASE = Path(__file__).resolve().parent

RESULTS_DIR = BASE / "processed/operation_classifier_results/non_llm_v1_1_balanced_noleak_all"
INPUT = BASE / "processed/manual_gold_operation_v1_1/hand_authored/manual_gold_model_ready.jsonl"
OUT_DIR = BASE / "processed/manual_gold_operation_v1_1/hand_authored/evaluation_all_non_llm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def supported_macro(per_label, labels):
    vals = []
    for label in labels:
        info = per_label.get(label, {})
        if int(info.get("support", 0) or 0) > 0:
            vals.append(float(info.get("f1", 0.0) or 0.0))
    return sum(vals) / len(vals) if vals else 0.0


rows = read_jsonl(INPUT)
texts = [row.get("input_text") or row.get("text") or "" for row in rows]
y_true_labels = [row["label"] for row in rows]

leaderboard = []

for model_path in sorted(RESULTS_DIR.glob("*/model.joblib")):
    model_name = model_path.parent.name

    payload = joblib.load(model_path)
    model = payload["model"]
    encoder = payload["label_encoder"]

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

    preds = []
    for row, pred_label in zip(rows, y_pred_labels):
        preds.append({
            "example_id": row.get("example_id"),
            "true_label": row["label"],
            "pred_label": str(pred_label),
            "correct": bool(row["label"] == pred_label),
            "memory_type": row.get("memory_type"),
            "sensitivity": row.get("sensitivity"),
        })

    model_out = OUT_DIR / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    write_jsonl(model_out / "predictions.jsonl", preds)
    write_jsonl(model_out / "errors.jsonl", [p for p in preds if not p["correct"]])

    row = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_all_labels": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
        "supported_macro_f1": supported_macro(per_label, LABELS),
        "lifecycle_macro_f1": supported_macro(per_label, LIFECYCLE_LABELS),
        "errors": sum(1 for p in preds if not p["correct"]),
        "label_counter_pred": dict(Counter(str(x) for x in y_pred_labels)),
        "metrics_file": str(model_out / "metrics.json"),
    }

    for label in LABELS:
        row[f"{label}_precision"] = per_label[label]["precision"]
        row[f"{label}_recall"] = per_label[label]["recall"]
        row[f"{label}_f1"] = per_label[label]["f1"]

    metrics = {
        "model": model_name,
        "input": str(INPUT),
        "total": len(rows),
        "label_counter_true": dict(Counter(y_true_labels)),
        **row,
        "per_label": per_label,
    }

    (model_out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    leaderboard.append(row)

leaderboard.sort(
    key=lambda r: (
        r["supported_macro_f1"],
        r["FORGET_recall"],
        r["lifecycle_macro_f1"],
    ),
    reverse=True,
)

for i, row in enumerate(leaderboard, 1):
    row["rank"] = i

write_tsv(OUT_DIR / "leaderboard.tsv", leaderboard)

print(json.dumps({
    "output_dir": str(OUT_DIR),
    "models": len(leaderboard),
    "best_model": leaderboard[0]["model"] if leaderboard else None,
    "best_supported_macro_f1": leaderboard[0]["supported_macro_f1"] if leaderboard else None,
    "best_forget_recall": leaderboard[0]["FORGET_recall"] if leaderboard else None,
    "leaderboard": str(OUT_DIR / "leaderboard.tsv"),
}, indent=2, ensure_ascii=False))
