import json
import re
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, f1_score

TEST_PATH = Path("processed/memorygovbench_operation_hard_ood_v1/operation_hard_ood_200.jsonl")
OUT_DIR = Path("processed/operation_policy_override_eval/hard_ood_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

SENSITIVE_PATTERNS = [
    r"\bghp_[A-Za-z0-9_]{10,}\b",
    r"\bsk-[A-Za-z0-9_-]{8,}\b",
    r"password",
    r"private key",
    r"BEGIN OPENSSH PRIVATE KEY",
    r"\b\d{4} \d{4} \d{4} \d{4}\b",
    r"\bRO\d{2}[A-Z0-9]{10,}\b",
    r"\bIBAN\b",
    r"passport number",
    r"recovery phrase",
    r"private_key_id",
]

IGNORE_PATTERNS = [
    r"for this run only",
    r"while this terminal is open",
    r"during this debugging session",
    r"for the current shell",
    r"one-off",
    r"quick check",
    r"next command.*failed attempt",
    r"only from my terminal paste",
    r"in this benchmark draft",
    r"for the next five minutes",
]

FORGET_PATTERNS = [
    r"\bremove\b",
    r"\bpurge\b",
    r"\berase\b",
    r"\bdiscard\b",
    r"\bdrop\b",
    r"\bclear\b",
    r"\bgone\b",
    r"not .* retained",
    r"should not be kept",
]

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def policy_predict(text):
    t = text.lower()

    if any(re.search(p, text, flags=re.I) for p in SENSITIVE_PATTERNS):
        return "QUARANTINE"

    if any(re.search(p, text, flags=re.I) for p in FORGET_PATTERNS):
        return "FORGET"

    if any(re.search(p, text, flags=re.I) for p in IGNORE_PATTERNS):
        return "IGNORE"

    return None

rows = read_jsonl(TEST_PATH)

y_true = []
y_pred = []
records = []

for r in rows:
    gold = r["label"]
    pred = policy_predict(r["text"]) or "NO_OVERRIDE"

    y_true.append(gold)
    y_pred.append(pred)

    records.append({
        "case_id": r["case_id"],
        "category": r["category"],
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "text": r["text"],
    })

valid_labels = LABELS + ["NO_OVERRIDE"]

summary = {
    "n": len(rows),
    "accuracy_if_override_only": accuracy_score(y_true, y_pred),
    "macro_f1_if_override_only": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    "prediction_counts": dict(Counter(y_pred)),
    "gold_counts": dict(Counter(y_true)),
    "classification_report": classification_report(y_true, y_pred, labels=valid_labels, zero_division=0, output_dict=True),
    "note": "This evaluates deterministic overrides alone, not classifier plus override fallback.",
}

(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

with (OUT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(json.dumps(summary, indent=2))
