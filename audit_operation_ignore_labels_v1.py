import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent

INPUTS = {
    "train_full": BASE / "processed/memorygovbench_operation_v1/operation_train_full.jsonl",
    "train_balanced": BASE / "processed/memorygovbench_operation_v1/operation_train_balanced.jsonl",
    "val": BASE / "processed/memorygovbench_operation_v1/operation_val.jsonl",
    "test_benchmark": BASE / "processed/memorygovbench_operation_v1/operation_test_benchmark.jsonl",
    "test_synthetic": BASE / "processed/memorygovbench_operation_v1/operation_test_synthetic.jsonl",
    "test_combined": BASE / "processed/memorygovbench_operation_v1/operation_test_combined.jsonl",
}

OUT_DIR = BASE / "processed/operation_label_audit_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT_DIR / "ignore_audit_all.jsonl"
OUT_EXCLUDED = OUT_DIR / "ignore_audit_excluded.jsonl"
OUT_KEPT = OUT_DIR / "ignore_audit_kept.jsonl"
OUT_EXCLUDE_IDS = OUT_DIR / "exclude_example_ids.txt"
OUT_REPORT = OUT_DIR / "report.json"


STORE_WORTHY_MARKERS = [
    "user uses",
    "user reads",
    "user flew",
    "user redeemed",
    "user applied",
    "user had",
    "user received",
    "user is flying",
    "user attended",
    "user prefers",
    "user likes",
    "user dislikes",
    "user wants",
    "user plans",
    "user is interested",
]


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_of(row):
    return row.get("gold", {}).get("primary_decision")


def candidate_text(row):
    return (row.get("candidate_memory") or {}).get("text") or ""


def has_history(row):
    history = row.get("history_context")
    return bool(history)


def has_existing(row):
    existing = row.get("existing_memories")
    return bool(existing)


def audit_row(row, split):
    label = label_of(row)
    source = row.get("source_dataset")
    text = candidate_text(row)
    text_l = text.lower()

    flags = []

    if label == "IGNORE":
        if source == "LongMemEval":
            flags.append("longmemeval_ignore")

        if not has_history(row):
            flags.append("no_history_context")

        if not has_existing(row):
            flags.append("no_existing_memories")

        if any(marker in text_l for marker in STORE_WORTHY_MARKERS):
            flags.append("looks_store_worthy_without_context")

    exclude = (
        label == "IGNORE"
        and source == "LongMemEval"
        and "no_history_context" in flags
        and "no_existing_memories" in flags
        and "looks_store_worthy_without_context" in flags
    )

    if exclude:
        audit_status = "exclude_from_operation_v1_1"
        reason = (
            "LongMemEval IGNORE example lacks history/existing-memory evidence and "
            "looks store-worthy under operation-governance policy."
        )
    else:
        audit_status = "keep"
        reason = "No exclusion rule matched."

    return {
        "example_id": row.get("example_id"),
        "split": split,
        "audit_status": audit_status,
        "exclude": exclude,
        "reason": reason,
        "label": label,
        "source_dataset": source,
        "source_file": row.get("source_file"),
        "scenario_type": row.get("scenario_type"),
        "memory_type": (row.get("candidate_memory") or {}).get("memory_type"),
        "sensitivity": (row.get("candidate_memory") or {}).get("sensitivity"),
        "candidate_memory": text,
        "has_history_context": has_history(row),
        "has_existing_memories": has_existing(row),
        "flags": flags,
    }


def main():
    all_audits = []
    seen = set()

    for split, path in INPUTS.items():
        for row in read_jsonl(path):
            if label_of(row) != "IGNORE":
                continue

            audit = audit_row(row, split)
            key = (audit["example_id"], split)
            if key in seen:
                continue
            seen.add(key)
            all_audits.append(audit)

    excluded = [row for row in all_audits if row["exclude"]]
    kept = [row for row in all_audits if not row["exclude"]]

    unique_excluded_ids = sorted(set(row["example_id"] for row in excluded))

    write_jsonl(OUT_ALL, all_audits)
    write_jsonl(OUT_EXCLUDED, excluded)
    write_jsonl(OUT_KEPT, kept)

    OUT_EXCLUDE_IDS.write_text("\n".join(unique_excluded_ids) + "\n", encoding="utf-8")

    report = {
        "version": "v3.68",
        "task": "audit_operation_ignore_labels_v1",
        "purpose": "Identify LongMemEval IGNORE labels that are not justified under no-leak operation-governance input.",
        "outputs": {
            "all": str(OUT_ALL),
            "excluded": str(OUT_EXCLUDED),
            "kept": str(OUT_KEPT),
            "exclude_ids": str(OUT_EXCLUDE_IDS),
            "report": str(OUT_REPORT),
        },
        "total_ignore_audits": len(all_audits),
        "excluded_rows": len(excluded),
        "unique_excluded_ids": len(unique_excluded_ids),
        "kept_rows": len(kept),
        "status_counter": dict(Counter(row["audit_status"] for row in all_audits)),
        "split_counter_all": dict(Counter(row["split"] for row in all_audits)),
        "split_counter_excluded": dict(Counter(row["split"] for row in excluded)),
        "source_counter_all": dict(Counter(row["source_dataset"] for row in all_audits)),
        "source_counter_excluded": dict(Counter(row["source_dataset"] for row in excluded)),
        "flag_counter": dict(Counter(flag for row in all_audits for flag in row["flags"])),
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
