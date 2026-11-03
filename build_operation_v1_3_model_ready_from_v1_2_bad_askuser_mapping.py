import json
import re
from collections import Counter
from pathlib import Path


SRC_DIR = Path("processed/memorygovbench_operation_v1_2_model_ready_noleak")
OUT_DIR = Path("processed/memorygovbench_operation_v1_3_model_ready_noleak")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASK_USER_OLD = "ASK_USER"

VALID_NEW_LABELS = {
    "WRITE",
    "IGNORE",
    "UPDATE",
    "MERGE",
    "FORGET",
    "QUARANTINE",
}

SENSITIVE_MARKERS = [
    "national id",
    "passport",
    "passport number",
    "bank account",
    "iban",
    "credit card",
    "api token",
    "access token",
    "secret token",
    "database password",
    "password",
    "private ssh key",
    "private key",
    "seed phrase",
    "recovery code",
    "tax identifier",
    "insurance number",
    "crypto seed",
    "home alarm code",
    "alarm code",
    "bank pin",
    "pin ",
    "cloud access key",
    "vpn secret",
    "exact home address",
    "home address",
    "private phone number",
    "medical diagnosis",
    "diagnosis",
    "medical prescription",
    "prescription",
    "hr complaint",
    "legal case number",
    "case file",
    "identity scan",
    "salary is",
    "salary amount",
    "child attends",
    "therapy",
    "medical leave",
    "doctor",
    "medication",
    "loan payment",
    "divorce",
]

DESTRUCTIVE_MARKERS = [
    "delete",
    "forget",
    "remove from memory",
    "erase",
]

UPDATE_MARKERS = [
    "no longer",
    "stopped",
    "instead",
    "now prefer",
    "changed",
    "switched",
]

MERGE_MARKERS = [
    "also includes",
    "now includes",
    "add this to",
    "same project",
    "in addition",
]


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_label(row):
    for key in ["label", "operation_label", "target", "class_label", "y"]:
        if key in row and isinstance(row[key], str):
            return row[key]
    raise KeyError(f"No label field found in row keys: {sorted(row.keys())}")


def set_label(row, new_label):
    out = dict(row)
    changed = False

    for key in ["label", "operation_label", "target", "class_label", "y"]:
        if key in out and isinstance(out[key], str):
            out[key] = new_label
            changed = True

    if not changed:
        out["label"] = new_label

    out["operation_label_v1_3"] = new_label
    return out


def row_text(row):
    parts = []
    for key in [
        "text",
        "input",
        "prompt",
        "user_message",
        "message",
        "candidate",
        "candidate_memory",
        "target_memory",
        "context",
    ]:
        val = row.get(key)
        if isinstance(val, str):
            parts.append(val)
    return "\n".join(parts).lower()


def has_any(text, markers):
    return any(m in text for m in markers)


def relabel_ask_user(row):
    text = row_text(row)

    if has_any(text, SENSITIVE_MARKERS):
        return "QUARANTINE", "uncertain_consent_sensitive_boundary"

    if has_any(text, DESTRUCTIVE_MARKERS):
        return "FORGET", "uncertain_consent_destructive"

    if has_any(text, UPDATE_MARKERS):
        return "UPDATE", "uncertain_consent_update"

    if has_any(text, MERGE_MARKERS):
        return "MERGE", "uncertain_consent_merge"

    return "WRITE", "uncertain_consent_write"


def convert_row(row):
    old_label = get_label(row)
    out = dict(row)
    out["old_operation_label"] = old_label

    if old_label == ASK_USER_OLD:
        new_label, reason = relabel_ask_user(row)
        out = set_label(out, new_label)
        out["requires_confirmation"] = True
        out["confirmation_reason"] = reason
        out["removed_ask_user_label"] = True
        return out

    if old_label not in VALID_NEW_LABELS:
        raise ValueError(f"Unexpected label after ASK_USER removal: {old_label}")

    out = set_label(out, old_label)

    # Confirmation metadata is policy metadata, not the classifier target.
    if old_label == "FORGET":
        out["requires_confirmation"] = True
        out["confirmation_reason"] = "destructive_action"
    elif old_label == "QUARANTINE":
        out["requires_confirmation"] = True
        out["confirmation_reason"] = "sensitive_boundary"
    else:
        out["requires_confirmation"] = False
        out["confirmation_reason"] = "none"

    out["removed_ask_user_label"] = False
    return out


def main():
    src_files = sorted(SRC_DIR.glob("*.jsonl"))
    if not src_files:
        raise SystemExit(f"No jsonl files found in {SRC_DIR}")

    report = {
        "source_dir": str(SRC_DIR),
        "output_dir": str(OUT_DIR),
        "files": {},
        "total_old_labels": Counter(),
        "total_new_labels": Counter(),
        "ask_user_mapped_to": Counter(),
    }

    for src in src_files:
        rows = read_jsonl(src)
        out_rows = []
        old_counts = Counter()
        new_counts = Counter()
        ask_map = Counter()

        for row in rows:
            old_label = get_label(row)
            old_counts[old_label] += 1

            out = convert_row(row)
            new_label = out["operation_label_v1_3"]
            new_counts[new_label] += 1

            if old_label == ASK_USER_OLD:
                ask_map[new_label] += 1

            out_rows.append(out)

        dst = OUT_DIR / src.name
        write_jsonl(dst, out_rows)

        report["files"][src.name] = {
            "rows": len(rows),
            "old_labels": dict(old_counts),
            "new_labels": dict(new_counts),
            "ask_user_mapped_to": dict(ask_map),
        }

        report["total_old_labels"].update(old_counts)
        report["total_new_labels"].update(new_counts)
        report["ask_user_mapped_to"].update(ask_map)

    report["total_old_labels"] = dict(report["total_old_labels"])
    report["total_new_labels"] = dict(report["total_new_labels"])
    report["ask_user_mapped_to"] = dict(report["ask_user_mapped_to"])
    report["new_label_space"] = sorted(VALID_NEW_LABELS)

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
