import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

MODEL_DIR = Path("processed/operation_classifier_results/transformer_mpnet_base_v1_3_2_no_confirmation/final_model")
TEST_PATH = Path("processed/memorygovbench_operation_hard_ood_v2_qwen/operation_hard_ood_v2_qwen.jsonl")
OUT_DIR = Path("processed/operation_classifier_results/transformer_mpnet_base_v1_3_2_no_confirmation/hard_ood_v2_qwen_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]
label2id = {x: i for i, x in enumerate(LABELS)}
id2label = {i: x for x, i in label2id.items()}

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

class D(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok = tok
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=384)
        enc["labels"] = label2id[r["label"]]
        return enc

rows = read_jsonl(TEST_PATH)
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)

args = TrainingArguments(
    output_dir=str(OUT_DIR / "tmp"),
    per_device_eval_batch_size=32,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=D(rows, tok),
    data_collator=DataCollatorWithPadding(tok),
)

pred = trainer.predict(D(rows, tok))
pred_ids = np.argmax(pred.predictions, axis=-1)

y_true = [r["label"] for r in rows]
y_pred = [id2label[int(x)] for x in pred_ids]

report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)

summary = {
    "model": "MPNet",
    "model_dir": str(MODEL_DIR),
    "test_path": str(TEST_PATH),
    "n": len(rows),
    "accuracy": accuracy_score(y_true, y_pred),
    "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    "classification_report": report,
    "class_f1": {lab: report.get(lab, {}).get("f1-score", 0.0) for lab in LABELS},
}

(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

with (OUT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as f:
    for r, gold, pred_lab in zip(rows, y_true, y_pred):
        f.write(json.dumps({
            "case_id": r["case_id"],
            "category": r["category"],
            "gold": gold,
            "pred": pred_lab,
            "correct": gold == pred_lab,
            "text": r["text"],
        }, ensure_ascii=False) + "\n")

print(json.dumps(summary, indent=2))
