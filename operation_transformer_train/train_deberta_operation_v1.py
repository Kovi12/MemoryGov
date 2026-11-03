import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)


DATA_DIR = Path("processed/memorygovbench_operation_v1_3_2_no_confirmation_semantic_model_ready_noleak")
OUT_DIR = Path("processed/operation_classifier_results/transformer_deberta_v1_3_2_no_confirmation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]
label2id = {x: i for i, x in enumerate(LABELS)}
id2label = {i: x for x, i in label2id.items()}


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def get_label(row):
    for key in [
        "operation_label_v1_3_2_no_confirmation",
        "operation_label_v1_3_1",
        "operation_label_v1_3",
        "label",
        "operation_label",
        "target",
        "class_label",
        "y",
    ]:
        if key in row and isinstance(row[key], str):
            return row[key]
    raise KeyError(f"No label field found: {sorted(row.keys())}")


def get_text(row):
    v = row.get("text")
    if isinstance(v, str) and v.strip():
        return v
    for key in ["input", "prompt"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    raise KeyError(f"No text field found: {sorted(row.keys())}")


class OpDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=384):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = [get_text(r) for r in rows]
        self.labels = [label2id[get_label(r)] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = self.labels[idx]
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)

    y_true = [id2label[int(x)] for x in labels]
    y_pred = [id2label[int(x)] for x in pred]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }


def main():
    set_seed(31)

    train_path = DATA_DIR / "operation_train_balanced_model_ready.jsonl"
    val_path = DATA_DIR / "operation_val_model_ready.jsonl"

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)

    print("train rows:", len(train_rows), dict(Counter(get_label(r) for r in train_rows)))
    print("val rows:", len(val_rows), dict(Counter(get_label(r) for r in val_rows)))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, fix_mistral_regex=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )

    train_ds = OpDataset(train_rows, tokenizer)
    val_ds = OpDataset(val_rows, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        run_name="deberta_opgov_v1_3_2",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    metrics = trainer.evaluate()

    pred_out = trainer.predict(val_ds)
    pred_ids = np.argmax(pred_out.predictions, axis=-1)

    y_true = [id2label[int(x)] for x in pred_out.label_ids]
    y_pred = [id2label[int(x)] for x in pred_ids]

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )

    final_dir = OUT_DIR / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_name": MODEL_NAME,
        "final_model_dir": str(final_dir),
        "label_space": LABELS,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_seconds": train_seconds,
        "eval_metrics": metrics,
        "classification_report": report,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "class_f1": {
            lab: report.get(lab, {}).get("f1-score", 0.0)
            for lab in LABELS
        },
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (OUT_DIR / "val_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, gold, pred in zip(val_rows, y_true, y_pred):
            f.write(json.dumps({
                "gold": gold,
                "pred": pred,
                "correct": gold == pred,
                "text": get_text(row),
            }, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
