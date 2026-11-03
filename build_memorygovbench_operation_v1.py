import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent

OUT_DIR = BASE / "processed/memorygovbench_operation_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_TRAIN = BASE / "processed/memorygovbench_current/operation_train.jsonl"
CURRENT_VAL = BASE / "processed/memorygovbench_current/operation_val.jsonl"
CURRENT_TEST = BASE / "processed/memorygovbench_current/operation_test.jsonl"

STRICT_EXISTING = BASE / "processed/frozen_results/operation_expansion_existing_v1/accepted.jsonl"

SYNTHETIC_FILES = [
    BASE / "processed/frozen_results/operation_synthetic_pilot_v1/clean.jsonl",
    BASE / "processed/frozen_results/operation_synthetic_batch1_v1/clean.jsonl",
    BASE / "processed/frozen_results/operation_synthetic_remaining_v1/clean.jsonl",
]

OUT_TRAIN_FULL = OUT_DIR / "operation_train_full.jsonl"
OUT_TRAIN_BALANCED = OUT_DIR / "operation_train_balanced.jsonl"
OUT_VAL = OUT_DIR / "operation_val.jsonl"
OUT_TEST_BENCHMARK = OUT_DIR / "operation_test_benchmark.jsonl"
OUT_TEST_SYNTHETIC = OUT_DIR / "operation_test_synthetic.jsonl"
OUT_TEST_COMBINED = OUT_DIR / "operation_test_combined.jsonl"
OUT_REPORT = OUT_DIR / "report.json"

SEED = 20260520
SYNTHETIC_TRAIN_RATIO = 0.80
SYNTHETIC_VAL_RATIO = 0.10
SYNTHETIC_TEST_RATIO = 0.10
BALANCED_MAX_PER_LABEL = 1200


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_of(row):
    return row.get("gold", {}).get("primary_decision")


def stable_key(row):
    example_id = row.get("example_id")
    if example_id:
        return f"id::{example_id}"

    candidate = row.get("candidate_memory", {}).get("text", "")
    existing = " || ".join(m.get("text", "") for m in row.get("existing_memories", []) or [])
    query = row.get("current_query", "")
    label = label_of(row)
    raw = f"{label}||{candidate}||{existing}||{query}"
    return "hash::" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def mark_source(row, source_tag):
    row = dict(row)
    metadata = dict(row.get("annotation_metadata") or {})
    metadata["operation_v1_source"] = source_tag
    row["annotation_metadata"] = metadata
    return row


def dedupe_rows(rows):
    seen = set()
    out = []
    duplicates = 0

    for row in rows:
        key = stable_key(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        out.append(row)

    return out, duplicates


def split_synthetic(rows):
    rng = random.Random(SEED)
    by_label = defaultdict(list)

    for row in rows:
        by_label[label_of(row)].append(row)

    train = []
    val = []
    test = []

    split_stats = {}

    for label, label_rows in sorted(by_label.items()):
        rng.shuffle(label_rows)
        n = len(label_rows)

        n_train = int(n * SYNTHETIC_TRAIN_RATIO)
        n_val = int(n * SYNTHETIC_VAL_RATIO)

        label_train = label_rows[:n_train]
        label_val = label_rows[n_train:n_train + n_val]
        label_test = label_rows[n_train + n_val:]

        train.extend(label_train)
        val.extend(label_val)
        test.extend(label_test)

        split_stats[label] = {
            "total": n,
            "train": len(label_train),
            "val": len(label_val),
            "test": len(label_test),
        }

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test, split_stats


def make_balanced_train(rows):
    rng = random.Random(SEED + 1)
    by_label = defaultdict(list)

    for row in rows:
        by_label[label_of(row)].append(row)

    non_empty_counts = [len(v) for v in by_label.values() if v]
    if not non_empty_counts:
        return [], {}

    target = min(BALANCED_MAX_PER_LABEL, max(non_empty_counts))

    balanced = []
    stats = {}

    for label, label_rows in sorted(by_label.items()):
        label_rows = list(label_rows)
        rng.shuffle(label_rows)

        if len(label_rows) > target:
            selected = label_rows[:target]
            mode = "downsampled"
        else:
            selected = label_rows
            mode = "kept_all"

        balanced.extend(selected)
        stats[label] = {
            "available": len(label_rows),
            "selected": len(selected),
            "mode": mode,
        }

    rng.shuffle(balanced)
    return balanced, stats


def summarize(rows):
    return {
        "total": len(rows),
        "label_counter": dict(Counter(label_of(row) for row in rows)),
        "source_dataset_counter": dict(Counter(row.get("source_dataset") for row in rows)),
        "operation_v1_source_counter": dict(Counter((row.get("annotation_metadata") or {}).get("operation_v1_source") for row in rows)),
        "scenario_counter": dict(Counter(row.get("scenario_type") for row in rows)),
        "memory_type_counter": dict(Counter(row.get("candidate_memory", {}).get("memory_type") for row in rows)),
        "sensitivity_counter": dict(Counter(row.get("candidate_memory", {}).get("sensitivity") for row in rows)),
    }


def main():
    current_train = [mark_source(row, "benchmark_current_train") for row in read_jsonl(CURRENT_TRAIN)]
    current_val = [mark_source(row, "benchmark_current_val") for row in read_jsonl(CURRENT_VAL)]
    current_test = [mark_source(row, "benchmark_current_test") for row in read_jsonl(CURRENT_TEST)]

    strict_existing = [mark_source(row, "strict_existing_expansion_train_only") for row in read_jsonl(STRICT_EXISTING)]

    synthetic_all = []
    synthetic_file_counts = {}

    for path in SYNTHETIC_FILES:
        rows = [mark_source(row, f"synthetic_clean::{path.name}::{path.parent.name}") for row in read_jsonl(path)]
        synthetic_file_counts[str(path)] = summarize(rows)
        synthetic_all.extend(rows)

    synthetic_all, synthetic_duplicates = dedupe_rows(synthetic_all)
    synthetic_train, synthetic_val, synthetic_test, synthetic_split_stats = split_synthetic(synthetic_all)

    train_full = current_train + strict_existing + synthetic_train
    val = current_val + synthetic_val
    test_benchmark = current_test
    test_synthetic = synthetic_test
    test_combined = current_test + synthetic_test

    train_full, train_full_duplicates = dedupe_rows(train_full)
    val, val_duplicates = dedupe_rows(val)
    test_benchmark, test_benchmark_duplicates = dedupe_rows(test_benchmark)
    test_synthetic, test_synthetic_duplicates = dedupe_rows(test_synthetic)
    test_combined, test_combined_duplicates = dedupe_rows(test_combined)

    train_balanced, balanced_stats = make_balanced_train(train_full)

    write_jsonl(OUT_TRAIN_FULL, train_full)
    write_jsonl(OUT_TRAIN_BALANCED, train_balanced)
    write_jsonl(OUT_VAL, val)
    write_jsonl(OUT_TEST_BENCHMARK, test_benchmark)
    write_jsonl(OUT_TEST_SYNTHETIC, test_synthetic)
    write_jsonl(OUT_TEST_COMBINED, test_combined)

    report = {
        "version": "v3.61",
        "dataset_name": "MemoryGovBench-operation-v1",
        "purpose": "First complete operation-governance dataset combining benchmark-native WRITE/IGNORE data, strict benchmark-derived expansion, and validated synthetic lifecycle labels.",
        "seed": SEED,
        "split_policy": {
            "benchmark_current_train_val_test": "Preserved from existing MemoryGovBench-current splits.",
            "strict_existing_expansion": "Train-only auxiliary data.",
            "synthetic_clean": {
                "train_ratio": SYNTHETIC_TRAIN_RATIO,
                "val_ratio": SYNTHETIC_VAL_RATIO,
                "test_ratio": SYNTHETIC_TEST_RATIO,
                "stratified_by": "gold.primary_decision",
            },
            "benchmark_test": "Kept separate to evaluate benchmark-native WRITE/IGNORE behavior.",
            "synthetic_test": "Kept separate to evaluate expanded lifecycle labels.",
        },
        "inputs": {
            "current_train": str(CURRENT_TRAIN),
            "current_val": str(CURRENT_VAL),
            "current_test": str(CURRENT_TEST),
            "strict_existing": str(STRICT_EXISTING),
            "synthetic_files": [str(path) for path in SYNTHETIC_FILES],
        },
        "outputs": {
            "train_full": str(OUT_TRAIN_FULL),
            "train_balanced": str(OUT_TRAIN_BALANCED),
            "val": str(OUT_VAL),
            "test_benchmark": str(OUT_TEST_BENCHMARK),
            "test_synthetic": str(OUT_TEST_SYNTHETIC),
            "test_combined": str(OUT_TEST_COMBINED),
            "report": str(OUT_REPORT),
        },
        "synthetic_file_counts": synthetic_file_counts,
        "synthetic_deduplicated_total": len(synthetic_all),
        "synthetic_duplicates_removed": synthetic_duplicates,
        "synthetic_split_stats": synthetic_split_stats,
        "dedupe_after_merge": {
            "train_full_duplicates_removed": train_full_duplicates,
            "val_duplicates_removed": val_duplicates,
            "test_benchmark_duplicates_removed": test_benchmark_duplicates,
            "test_synthetic_duplicates_removed": test_synthetic_duplicates,
            "test_combined_duplicates_removed": test_combined_duplicates,
        },
        "splits": {
            "train_full": summarize(train_full),
            "train_balanced": summarize(train_balanced),
            "val": summarize(val),
            "test_benchmark": summarize(test_benchmark),
            "test_synthetic": summarize(test_synthetic),
            "test_combined": summarize(test_combined),
        },
        "balanced_train_policy": {
            "balanced_max_per_label": BALANCED_MAX_PER_LABEL,
            "stats": balanced_stats,
        },
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
