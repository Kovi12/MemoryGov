import json
from pathlib import Path
from collections import Counter

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE = Path("processed/operation_classifier_results/non_llm_v1_3_2_no_confirmation_semantic_selected")
TEST_PATH = Path("processed/memorygovbench_operation_hard_ood_v2b_qwen_schema/operation_hard_ood_v2b_qwen_schema.jsonl")
OUT_DIR = BASE / "hard_ood_v2b_qwen_schema_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def find_model_path():
    # Preferred: selected model metadata.
    meta = BASE / "final_selected_model.json"
    if meta.exists():
        try:
            obj = json.loads(meta.read_text())
            for key in ["selected_model", "model", "name"]:
                if key in obj:
                    cand = BASE / obj[key] / "model.joblib"
                    if cand.exists():
                        return cand
        except Exception:
            pass

    paths = list(BASE.glob("*/model.joblib")) + list(BASE.glob("model.joblib"))
    if not paths:
        raise FileNotFoundError(f"No model.joblib found under {BASE}")
    return paths[0]


rows = read_jsonl(TEST_PATH)
model_path = find_model_path()
print("model_path:", model_path)

model = joblib.load(model_path)

texts = [r["text"] for r in rows]
y_true = [r["label"] for r in rows]
y_pred = list(model.predict(texts))

report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)

summary = {
    "model": "non_llm_selected_v1_3_2",
    "model_path": str(model_path),
    "test_path": str(TEST_PATH),
    "n": len(rows),
    "accuracy": accuracy_score(y_true, y_pred),
    "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    "prediction_counts": dict(Counter(y_pred)),
    "gold_counts": dict(Counter(y_true)),
    "classification_report": report,
    "class_f1": {lab: report.get(lab, {}).get("f1-score", 0.0) for lab in LABELS},
}

(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

with (OUT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as f:
    for r, pred in zip(rows, y_pred):
        f.write(json.dumps({
            "case_id": r["case_id"],
            "category": r["category"],
            "gold": r["label"],
            "pred": pred,
            "correct": r["label"] == pred,
            "text": r["text"],
        }, ensure_ascii=False) + "\n")

print(json.dumps(summary, indent=2))
