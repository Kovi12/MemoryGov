import json
import os
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception as e:
    raise SystemExit(
        "PEFT is required for this job. Install/check peft in the agent env. "
        f"Import error: {repr(e)}"
    )


DATA_DIR = Path(os.environ.get(
    "OPGOV_DATA_DIR",
    "processed/memorygovbench_operation_v1_3_2_no_confirmation_semantic_model_ready_noleak"
))

MODEL_NAME = os.environ.get("OPGOV_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
OUT_DIR = Path(os.environ.get(
    "OPGOV_OUT_DIR",
    "processed/operation_classifier_results/qwen2_5_7b_lora_opgov_v1_3_2_no_confirmation"
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

LABEL_DEFS = """Labels:
WRITE - Store a new durable memory that is not already represented.
IGNORE - Do not store the candidate because it is temporary, trivial, procedural, or not useful as long-term memory.
UPDATE - Replace or correct an existing memory because the new information contradicts or supersedes it.
MERGE - Combine compatible new information with an existing memory while preserving the old information.
FORGET - Delete or deactivate an existing memory because the user requests removal or retention is no longer appropriate.
QUARANTINE - Isolate high-risk sensitive information such as secrets, API keys, passwords, bank accounts, private identifiers, or highly sensitive personal data.
"""


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


def build_prompt(text):
    return (
        "You are an operation classifier for a long-term memory governance system.\n\n"
        f"{LABEL_DEFS}\n"
        "Classify the following input into exactly one label.\n\n"
        f"Input:\n{text}\n\n"
        "Label:"
    )


def parse_label(text):
    upper = str(text).upper()
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", upper):
            return lab
    return "PARSE_ERROR"


class CausalLabelDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        label = get_label(row)
        prompt = build_prompt(get_text(row))
        target = " " + label + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        max_prompt_len = self.max_length - len(target_ids)
        if max_prompt_len < 32:
            max_prompt_len = 32

        prompt_ids = prompt_ids[-max_prompt_len:]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for x in features:
            pad = max_len - len(x["input_ids"])
            batch["input_ids"].append(x["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(x["attention_mask"] + [0] * pad)
            batch["labels"].append(x["labels"] + [-100] * pad)

        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }


@torch.no_grad()
def evaluate_generation(model, tokenizer, rows, limit=None):
    model.eval()

    if limit:
        rows = rows[:limit]

    y_true = []
    y_pred = []
    latencies = []

    device = next(model.parameters()).device

    for i, row in enumerate(rows, 1):
        gold = get_label(row)
        prompt = build_prompt(get_text(row))

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t0 = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        latencies.append(time.time() - t0)

        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_label(gen)

        y_true.append(gold)
        y_pred.append(pred)

        if i <= 20 or i % 100 == 0:
            print(f"eval {i}/{len(rows)} gold={gold} pred={pred} raw={gen!r}")

    valid = [(a, b) for a, b in zip(y_true, y_pred) if b in LABELS]
    y_true_valid = [a for a, b in valid]
    y_pred_valid = [b for a, b in valid]

    return {
        "n": len(rows),
        "valid_predictions": len(valid),
        "parse_errors": sum(b not in LABELS for b in y_pred),
        "accuracy_all": accuracy_score(y_true, y_pred),
        "accuracy_valid_only": accuracy_score(y_true_valid, y_pred_valid) if valid else None,
        "macro_f1_valid_only": f1_score(y_true_valid, y_pred_valid, labels=LABELS, average="macro", zero_division=0) if valid else None,
        "mean_latency_sec": float(np.mean(latencies)) if latencies else None,
        "median_latency_sec": float(np.median(latencies)) if latencies else None,
        "classification_report_valid_only": classification_report(
            y_true_valid,
            y_pred_valid,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        ) if valid else None,
        "label_counts_gold": dict(Counter(y_true)),
        "label_counts_pred": dict(Counter(y_pred)),
    }


def main():
    set_seed(31)

    train_path = DATA_DIR / "operation_train_balanced_model_ready.jsonl"
    val_path = DATA_DIR / "operation_val_model_ready.jsonl"

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)

    max_train = int(os.environ.get("OPGOV_MAX_TRAIN_ROWS", "0"))
    if max_train > 0:
        train_rows = train_rows[:max_train]

    print("model:", MODEL_NAME)
    print("out:", OUT_DIR)
    print("train rows:", len(train_rows), dict(Counter(get_label(r) for r in train_rows)))
    print("val rows:", len(val_rows), dict(Counter(get_label(r) for r in val_rows)))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(os.environ.get("OPGOV_LORA_R", "16")),
        lora_alpha=int(os.environ.get("OPGOV_LORA_ALPHA", "32")),
        lora_dropout=float(os.environ.get("OPGOV_LORA_DROPOUT", "0.05")),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = CausalLabelDataset(train_rows, tokenizer)
    collator = CausalCollator(tokenizer)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        run_name="qwen7b_lora_opgov_v1_3_2",
        learning_rate=float(os.environ.get("OPGOV_LR", "2e-4")),
        per_device_train_batch_size=int(os.environ.get("OPGOV_BATCH", "1")),
        gradient_accumulation_steps=int(os.environ.get("OPGOV_GRAD_ACCUM", "16")),
        num_train_epochs=float(os.environ.get("OPGOV_EPOCHS", "1")),
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    final_dir = OUT_DIR / "final_lora_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    eval_limit = int(os.environ.get("OPGOV_EVAL_LIMIT", "0"))
    eval_summary = evaluate_generation(model, tokenizer, val_rows, limit=eval_limit if eval_limit > 0 else None)

    summary = {
        "model_name": MODEL_NAME,
        "adapter_dir": str(final_dir),
        "label_space": LABELS,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_seconds": train_seconds,
        "train_config": {
            "epochs": os.environ.get("OPGOV_EPOCHS", "1"),
            "lr": os.environ.get("OPGOV_LR", "2e-4"),
            "batch": os.environ.get("OPGOV_BATCH", "1"),
            "grad_accum": os.environ.get("OPGOV_GRAD_ACCUM", "16"),
            "lora_r": os.environ.get("OPGOV_LORA_R", "16"),
        },
        "eval": eval_summary,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
