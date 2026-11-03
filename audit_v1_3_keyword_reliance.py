import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import pandas as pd


DATA_DIR = Path("processed/memorygovbench_operation_v1_3_model_ready_noleak")
MODEL_DIR = Path("processed/operation_classifier_results/non_llm_v1_3_no_ask_user_selected")
OUT_DIR = MODEL_DIR / "keyword_reliance_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

MODELS = {
    "macro_selected_tfidf_sgd_hinge": MODEL_DIR / "tfidf_sgd_hinge_balanced/model.joblib",
    "safety_selected_char_sgd_hinge": MODEL_DIR / "char_sgd_hinge_balanced/model.joblib",
}

TRIGGER_PATTERNS = [
    # UPDATE-like
    (r"\bno longer\b", "MASK_UPDATE"),
    (r"\bstopped\b", "MASK_UPDATE"),
    (r"\bswitched\b", "MASK_UPDATE"),
    (r"\binstead\b", "MASK_UPDATE"),
    (r"\bchanged\b", "MASK_UPDATE"),
    (r"\bnow prefer\b", "MASK_UPDATE"),

    # MERGE-like
    (r"\balso\b", "MASK_MERGE"),
    (r"\bin addition\b", "MASK_MERGE"),
    (r"\badditionally\b", "MASK_MERGE"),
    (r"\bnow includes\b", "MASK_MERGE"),
    (r"\bincludes\b", "MASK_MERGE"),
    (r"\badd this\b", "MASK_MERGE"),

    # FORGET-like
    (r"\bforget\b", "MASK_FORGET"),
    (r"\bdelete\b", "MASK_FORGET"),
    (r"\bremove\b", "MASK_FORGET"),
    (r"\berase\b", "MASK_FORGET"),

    # IGNORE-like
    (r"\btemporary\b", "MASK_IGNORE"),
    (r"\bcurrently\b", "MASK_IGNORE"),
    (r"\btoday\b", "MASK_IGNORE"),
    (r"\bthis session\b", "MASK_IGNORE"),
    (r"\bright now\b", "MASK_IGNORE"),

    # QUARANTINE-like
    (r"\bpassword\b", "MASK_SENSITIVE"),
    (r"\btoken\b", "MASK_SENSITIVE"),
    (r"\bapi key\b", "MASK_SENSITIVE"),
    (r"\bpassport\b", "MASK_SENSITIVE"),
    (r"\bbank account\b", "MASK_SENSITIVE"),
    (r"\biban\b", "MASK_SENSITIVE"),
    (r"\bcredit card\b", "MASK_SENSITIVE"),
    (r"\bprivate key\b", "MASK_SENSITIVE"),
]


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def get_label(row):
    for key in ["operation_label_v1_3", "label", "operation_label", "target", "class_label", "y"]:
        if key in row and isinstance(row[key], str):
            return row[key]
    raise KeyError(row.keys())


def get_text(row):
    for key in ["text", "input", "prompt"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]

    parts = []
    for key in ["user_message", "message", "candidate", "candidate_memory", "target_memory", "context"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(f"{key}: {v}")
    return "\n".join(parts)


def mask_triggers(text):
    out = text
    for pat, repl in TRIGGER_PATTERNS:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def count_trigger_hits(text):
    hits = []
    for pat, repl in TRIGGER_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(repl)
    return hits


def eval_preds(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "report": classification_report(y, pred, labels=LABELS, zero_division=0, output_dict=True),
    }


def print_brief(name, metrics):
    print(f"{name} accuracy: {metrics['accuracy']:.4f}")
    print(f"{name} macro_f1: {metrics['macro_f1']:.4f}")
    for lab in LABELS:
        f1 = metrics["report"].get(lab, {}).get("f1-score", 0)
        print(f"  {lab:12s} f1={f1:.4f}")


def top_features_for_pipeline(model, model_name, top_n=25):
    if not hasattr(model, "named_steps"):
        return []

    if "tfidf" not in model.named_steps or "clf" not in model.named_steps:
        return []

    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    if not hasattr(clf, "coef_"):
        return []

    names = np.array(vec.get_feature_names_out())
    classes = list(clf.classes_)
    coefs = clf.coef_

    rows = []
    for i, cls in enumerate(classes):
        coef = coefs[i]
        top_idx = np.argsort(coef)[-top_n:][::-1]
        for rank, idx in enumerate(top_idx, 1):
            rows.append({
                "model": model_name,
                "label": str(cls),
                "rank": rank,
                "feature": str(names[idx]),
                "weight": float(coef[idx]),
            })
    return rows


rows = read_jsonl(DATA_DIR / "operation_val_model_ready.jsonl")
X = [get_text(r) for r in rows]
X_masked = [mask_triggers(x) for x in X]
y = [get_label(r) for r in rows]

trigger_by_label = defaultdict(Counter)
for label, text in zip(y, X):
    for h in count_trigger_hits(text):
        trigger_by_label[label][h] += 1

print("\nTrigger marker counts by gold label:")
for label in LABELS:
    print(label, dict(trigger_by_label[label]))

all_summary = {}

for model_name, model_path in MODELS.items():
    model = joblib.load(model_path)

    pred = list(model.predict(X))
    pred_masked = list(model.predict(X_masked))

    base_metrics = eval_preds(y, pred)
    masked_metrics = eval_preds(y, pred_masked)

    print("\n" + "=" * 96)
    print(model_name)
    print("=" * 96)

    print("\nOriginal validation:")
    print_brief("original", base_metrics)

    print("\nMasked trigger validation:")
    print_brief("masked", masked_metrics)

    print("\nDelta:")
    print(f"accuracy_delta: {masked_metrics['accuracy'] - base_metrics['accuracy']:.4f}")
    print(f"macro_f1_delta: {masked_metrics['macro_f1'] - base_metrics['macro_f1']:.4f}")

    cm = confusion_matrix(y, pred_masked, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"gold_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
    print("\nMasked confusion matrix:")
    print(cm_df.to_string())

    changed = []
    for row, gold, a, b, orig, masked in zip(rows, y, pred, pred_masked, X, X_masked):
        if a != b:
            changed.append({
                "gold": gold,
                "pred_original": a,
                "pred_masked": b,
                "text_original": orig,
                "text_masked": masked,
            })

    changed_path = OUT_DIR / f"{model_name}_prediction_changes_after_masking.jsonl"
    with changed_path.open("w", encoding="utf-8") as f:
        for item in changed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    feature_rows = top_features_for_pipeline(model, model_name)
    feature_path = OUT_DIR / f"{model_name}_top_features.jsonl"
    with feature_path.open("w", encoding="utf-8") as f:
        for item in feature_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    all_summary[model_name] = {
        "original_accuracy": base_metrics["accuracy"],
        "original_macro_f1": base_metrics["macro_f1"],
        "masked_accuracy": masked_metrics["accuracy"],
        "masked_macro_f1": masked_metrics["macro_f1"],
        "accuracy_delta": masked_metrics["accuracy"] - base_metrics["accuracy"],
        "macro_f1_delta": masked_metrics["macro_f1"] - base_metrics["macro_f1"],
        "changed_predictions": len(changed),
        "changed_predictions_path": str(changed_path),
        "top_features_path": str(feature_path),
    }

summary_path = OUT_DIR / "summary.json"
summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

print("\nWrote:", summary_path)
print(json.dumps(all_summary, indent=2))
