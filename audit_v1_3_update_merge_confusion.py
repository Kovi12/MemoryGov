import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


DATA_DIR = Path("processed/memorygovbench_operation_v1_3_model_ready_noleak")
MODEL_DIR = Path("processed/operation_classifier_results/non_llm_v1_3_no_ask_user_selected")

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

MODELS = {
    "macro_selected_tfidf_sgd_hinge": MODEL_DIR / "tfidf_sgd_hinge_balanced/model.joblib",
    "safety_selected_char_sgd_hinge": MODEL_DIR / "char_sgd_hinge_balanced/model.joblib",
}


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


def collapse(label):
    if label in {"UPDATE", "MERGE"}:
        return "REVISE"
    return label


rows = read_jsonl(DATA_DIR / "operation_val_model_ready.jsonl")
X = [get_text(r) for r in rows]
y = [get_label(r) for r in rows]

out_dir = MODEL_DIR / "update_merge_audit"
out_dir.mkdir(parents=True, exist_ok=True)

for model_name, model_path in MODELS.items():
    model = joblib.load(model_path)
    pred = list(model.predict(X))

    print("\n" + "=" * 88)
    print(model_name)
    print("=" * 88)

    print("\nClassification report:")
    print(classification_report(y, pred, labels=LABELS, zero_division=0))

    cm = confusion_matrix(y, pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"gold_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
    print("\nConfusion matrix:")
    print(cm_df.to_string())

    merge_as_update = sum(1 for a, b in zip(y, pred) if a == "MERGE" and b == "UPDATE")
    update_as_merge = sum(1 for a, b in zip(y, pred) if a == "UPDATE" and b == "MERGE")
    merge_total = sum(1 for a in y if a == "MERGE")
    update_total = sum(1 for a in y if a == "UPDATE")

    print("\nUPDATE/MERGE confusion:")
    print(f"MERGE -> UPDATE: {merge_as_update}/{merge_total}")
    print(f"UPDATE -> MERGE: {update_as_merge}/{update_total}")

    y_collapsed = [collapse(v) for v in y]
    p_collapsed = [collapse(v) for v in pred]
    collapsed_labels = ["WRITE", "IGNORE", "REVISE", "FORGET", "QUARANTINE"]

    print("\nIf UPDATE and MERGE are collapsed into REVISE:")
    print("collapsed accuracy:", accuracy_score(y_collapsed, p_collapsed))
    print("collapsed macro F1:", f1_score(y_collapsed, p_collapsed, labels=collapsed_labels, average="macro", zero_division=0))
    print(classification_report(y_collapsed, p_collapsed, labels=collapsed_labels, zero_division=0))

    errors = []
    for row, gold, got in zip(rows, y, pred):
        if gold in {"UPDATE", "MERGE"} or got in {"UPDATE", "MERGE"}:
            if gold != got:
                errors.append({
                    "gold": gold,
                    "pred": got,
                    "text": get_text(row),
                    "row": row,
                })

    err_path = out_dir / f"{model_name}_update_merge_errors.jsonl"
    with err_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("\nWrote:", err_path)
