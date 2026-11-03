import json
import re
from pathlib import Path
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, f1_score

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

POLICY_PATH = Path("processed/operation_policy_override_eval/hard_ood_v1/predictions.jsonl")

MODEL_PATHS = {
    "mpnet_plus_policy": Path("processed/operation_classifier_results/transformer_mpnet_base_v1_3_2_no_confirmation/hard_ood_v1_eval/predictions.jsonl"),
    "nonllm_plus_policy": Path("processed/operation_classifier_results/non_llm_v1_3_2_no_confirmation_semantic_selected/hard_ood_v1_eval/predictions.jsonl"),
}

OUT_DIR = Path("processed/operation_policy_override_eval/hard_ood_v1_policy_plus_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


policy_rows = read_jsonl(POLICY_PATH)
policy_by_id = {r["case_id"]: r["pred"] for r in policy_rows}

for name, model_path in MODEL_PATHS.items():
    if not model_path.exists():
        print("missing:", name, model_path)
        continue

    model_rows = read_jsonl(model_path)

    y_true = []
    y_pred = []
    records = []

    for r in model_rows:
        case_id = r["case_id"]
        gold = r["gold"]
        model_pred = r["pred"]
        policy_pred = policy_by_id.get(case_id, "NO_OVERRIDE")

        if policy_pred in LABELS:
            final_pred = policy_pred
            decision_source = "policy"
        else:
            final_pred = model_pred
            decision_source = "model"

        y_true.append(gold)
        y_pred.append(final_pred)

        records.append({
            "case_id": case_id,
            "category": r.get("category"),
            "gold": gold,
            "model_pred": model_pred,
            "policy_pred": policy_pred,
            "final_pred": final_pred,
            "decision_source": decision_source,
            "correct": final_pred == gold,
            "text": r.get("text", ""),
        })

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )

    summary = {
        "name": name,
        "model_predictions_path": str(model_path),
        "policy_predictions_path": str(POLICY_PATH),
        "n": len(records),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "prediction_counts": dict(Counter(y_pred)),
        "gold_counts": dict(Counter(y_true)),
        "decision_source_counts": dict(Counter(r["decision_source"] for r in records)),
        "classification_report": report,
        "class_f1": {
            lab: report.get(lab, {}).get("f1-score", 0.0)
            for lab in LABELS
        },
    }

    (OUT_DIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (OUT_DIR / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print("=" * 80)
    print(name)
    print(json.dumps(summary, indent=2))
