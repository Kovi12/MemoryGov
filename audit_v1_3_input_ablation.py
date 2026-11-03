import json
import re
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd


DATA_DIR = Path("processed/memorygovbench_operation_v1_3_model_ready_noleak")
MODEL_DIR = Path("processed/operation_classifier_results/non_llm_v1_3_no_ask_user_selected")
OUT_DIR = MODEL_DIR / "input_ablation_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def remove_allowed_labels(text):
    text = re.sub(
        r"Allowed labels:\s*(?:WRITE|IGNORE|UPDATE|MERGE|FORGET|ASK_USER|QUARANTINE|,\s*|\s*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


def remove_boilerplate(text):
    lines = []
    skip_next = False

    for line in text.splitlines():
        s = line.strip()

        if not s:
            lines.append(line)
            continue

        if s.startswith("Task:"):
            continue

        if s.startswith("Question:"):
            continue

        if s.startswith("Decision target:"):
            continue

        if s.startswith("Choose the correct operation label"):
            continue

        if s.startswith("Allowed labels:"):
            skip_next = True
            continue

        if skip_next:
            # Skip the label list line after "Allowed labels:"
            skip_next = False
            continue

        lines.append(line)

    return "\n".join(lines)


def remove_metadata(text):
    text = re.sub(r"Candidate metadata:\n(?:.*\n?){0,5}?(?=\n(?:Existing memories:|Decision target:|$))", "", text)
    text = re.sub(r"\[type=.*?; sensitivity=.*?; status=.*?\]", "", text)
    text = re.sub(r"\[type=.*?; sensitivity=.*?\]", "", text)
    text = re.sub(r"\btype\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsensitivity\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstatus\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)
    return text


def semantic_only(text):
    text = remove_boilerplate(text)
    text = remove_allowed_labels(text)
    text = remove_metadata(text)

    # Normalize section headers but keep actual semantic content.
    text = text.replace("History context:", "History:")
    text = text.replace("Candidate memory:", "Candidate:")
    text = text.replace("Existing memories:", "Existing:")

    # Drop repeated whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def eval_variant(model, X, y):
    pred = list(model.predict(X))
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "report": classification_report(y, pred, labels=LABELS, zero_division=0, output_dict=True),
        "pred": pred,
    }
    return metrics


rows = read_jsonl(DATA_DIR / "operation_val_model_ready.jsonl")
y = [get_label(r) for r in rows]
X_raw = [get_text(r) for r in rows]

variants = {
    "raw": X_raw,
    "no_allowed_labels": [remove_allowed_labels(x) for x in X_raw],
    "no_boilerplate": [remove_boilerplate(x) for x in X_raw],
    "no_metadata": [remove_metadata(x) for x in X_raw],
    "semantic_only": [semantic_only(x) for x in X_raw],
}

summary = {}

for model_name, model_path in MODELS.items():
    model = joblib.load(model_path)

    print("\n" + "=" * 96)
    print(model_name)
    print("=" * 96)

    model_summary = {}

    raw_pred = None

    for variant_name, X in variants.items():
        m = eval_variant(model, X, y)

        if variant_name == "raw":
            raw_pred = m["pred"]

        changed = 0 if raw_pred is None else sum(1 for a, b in zip(raw_pred, m["pred"]) if a != b)

        print(f"\nVariant: {variant_name}")
        print(f"accuracy: {m['accuracy']:.4f}")
        print(f"macro_f1: {m['macro_f1']:.4f}")
        print(f"changed_vs_raw: {changed}")

        for lab in LABELS:
            f1 = m["report"].get(lab, {}).get("f1-score", 0.0)
            print(f"  {lab:12s} f1={f1:.4f}")

        cm = confusion_matrix(y, m["pred"], labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"gold_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])

        cm_path = OUT_DIR / f"{model_name}_{variant_name}_confusion.csv"
        cm_df.to_csv(cm_path)

        changed_rows = []
        if raw_pred is not None and variant_name != "raw":
            for row, gold, pred_raw, pred_variant, text_raw, text_variant in zip(rows, y, raw_pred, m["pred"], X_raw, X):
                if pred_raw != pred_variant:
                    changed_rows.append({
                        "gold": gold,
                        "pred_raw": pred_raw,
                        "pred_variant": pred_variant,
                        "text_raw": text_raw,
                        "text_variant": text_variant,
                    })

        changed_path = OUT_DIR / f"{model_name}_{variant_name}_changed.jsonl"
        with changed_path.open("w", encoding="utf-8") as f:
            for item in changed_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        model_summary[variant_name] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "changed_vs_raw": changed,
            "confusion_path": str(cm_path),
            "changed_path": str(changed_path),
            "class_f1": {
                lab: m["report"].get(lab, {}).get("f1-score", 0.0)
                for lab in LABELS
            },
        }

    summary[model_name] = model_summary

summary_path = OUT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nWrote:", summary_path)
print(json.dumps(summary, indent=2))
