import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent

OPERATION_INPUT = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_qwen32b_clean.jsonl"
RETRIEVAL_INPUT = BASE_DIR / "processed/longmemeval_retrieval_governance_canonical.jsonl"
OUT_DIR = BASE_DIR / "processed/splits_longmemeval"
REPORT_PATH = OUT_DIR / "longmemeval_split_report.json"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
RETRIEVAL_NEGATIVE_RATIO = 4


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def open_writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", encoding="utf-8")


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_int(text):
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_float(text):
    return stable_int(text) / float(16 ** 16 - 1)


def get_split_from_group(group_key):
    x = stable_float(group_key)
    if x < TRAIN_RATIO:
        return "train"
    if x < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


def get_group_key(row):
    split_group = row.get("split_group") or {}
    source_file = split_group.get("source_file") or row.get("source_file") or "unknown_source"
    question_id = split_group.get("question_id") or row.get("question_id") or row.get("source_example_id") or row.get("example_id")
    return f"{source_file}|{question_id}"


def get_label(row):
    gold = row.get("gold") or {}
    return gold.get("primary_decision", "UNKNOWN")


def get_scenario(row):
    return row.get("scenario_type", "UNKNOWN")


def count_dataset(path):
    stats = {
        "total": 0,
        "split_counter": Counter(),
        "label_counter": Counter(),
        "label_by_split": defaultdict(Counter),
        "scenario_by_split": defaultdict(Counter),
        "group_counter": Counter(),
    }

    for row in read_jsonl(path):
        group_key = get_group_key(row)
        split = get_split_from_group(group_key)
        label = get_label(row)
        scenario = get_scenario(row)

        stats["total"] += 1
        stats["split_counter"][split] += 1
        stats["label_counter"][label] += 1
        stats["label_by_split"][split][label] += 1
        stats["scenario_by_split"][split][scenario] += 1
        stats["group_counter"][split] += int(stable_int(group_key) == stable_int(group_key))

    return stats


def counter_to_dict(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, defaultdict):
        return {k: counter_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: counter_to_dict(v) for k, v in obj.items()}
    return obj


def split_operation_dataset():
    writers = {
        "train": open_writer(OUT_DIR / "operation_train.jsonl"),
        "val": open_writer(OUT_DIR / "operation_val.jsonl"),
        "test": open_writer(OUT_DIR / "operation_test.jsonl"),
    }

    stats = {
        "input": str(OPERATION_INPUT),
        "outputs": {k: str(OUT_DIR / f"operation_{k}.jsonl") for k in writers},
        "total": 0,
        "split_counter": Counter(),
        "label_counter": Counter(),
        "label_by_split": defaultdict(Counter),
        "scenario_by_split": defaultdict(Counter),
        "unique_group_keys": set(),
        "group_by_split": defaultdict(set),
    }

    for row in read_jsonl(OPERATION_INPUT):
        group_key = get_group_key(row)
        split = get_split_from_group(group_key)
        label = get_label(row)
        scenario = get_scenario(row)

        write_row(writers[split], row)

        stats["total"] += 1
        stats["split_counter"][split] += 1
        stats["label_counter"][label] += 1
        stats["label_by_split"][split][label] += 1
        stats["scenario_by_split"][split][scenario] += 1
        stats["unique_group_keys"].add(group_key)
        stats["group_by_split"][split].add(group_key)

    for handle in writers.values():
        handle.close()

    stats["unique_group_keys"] = len(stats["unique_group_keys"])
    stats["group_by_split"] = {k: len(v) for k, v in stats["group_by_split"].items()}
    return stats


def get_retrieval_sampling_probabilities(first_pass_stats):
    probabilities = {}

    for split in ["train", "val", "test"]:
        retrieve_count = first_pass_stats["label_by_split"][split].get("RETRIEVE", 0)
        suppress_count = first_pass_stats["label_by_split"][split].get("SUPPRESS", 0)

        target_suppress = min(suppress_count, retrieve_count * RETRIEVAL_NEGATIVE_RATIO)

        if suppress_count == 0:
            probabilities[split] = 0.0
        else:
            probabilities[split] = target_suppress / suppress_count

    return probabilities


def split_retrieval_dataset():
    first_pass = {
        "total": 0,
        "split_counter": Counter(),
        "label_counter": Counter(),
        "label_by_split": defaultdict(Counter),
        "scenario_by_split": defaultdict(Counter),
        "unique_group_keys": set(),
        "group_by_split": defaultdict(set),
    }

    for row in read_jsonl(RETRIEVAL_INPUT):
        group_key = get_group_key(row)
        split = get_split_from_group(group_key)
        label = get_label(row)
        scenario = get_scenario(row)

        first_pass["total"] += 1
        first_pass["split_counter"][split] += 1
        first_pass["label_counter"][label] += 1
        first_pass["label_by_split"][split][label] += 1
        first_pass["scenario_by_split"][split][scenario] += 1
        first_pass["unique_group_keys"].add(group_key)
        first_pass["group_by_split"][split].add(group_key)

    probabilities = get_retrieval_sampling_probabilities(first_pass)

    natural_writers = {
        "train": open_writer(OUT_DIR / "retrieval_natural_train.jsonl"),
        "val": open_writer(OUT_DIR / "retrieval_natural_val.jsonl"),
        "test": open_writer(OUT_DIR / "retrieval_natural_test.jsonl"),
    }

    balanced_writers = {
        "train": open_writer(OUT_DIR / "retrieval_balanced_train.jsonl"),
        "val": open_writer(OUT_DIR / "retrieval_balanced_val.jsonl"),
        "test": open_writer(OUT_DIR / "retrieval_balanced_test.jsonl"),
    }

    balanced_stats = {
        "split_counter": Counter(),
        "label_counter": Counter(),
        "label_by_split": defaultdict(Counter),
        "scenario_by_split": defaultdict(Counter),
    }

    for row in read_jsonl(RETRIEVAL_INPUT):
        group_key = get_group_key(row)
        split = get_split_from_group(group_key)
        label = get_label(row)
        scenario = get_scenario(row)
        example_id = row.get("example_id", "")

        write_row(natural_writers[split], row)

        keep_balanced = False

        if label == "RETRIEVE":
            keep_balanced = True
        elif label == "SUPPRESS":
            keep_balanced = stable_float(f"{split}|{example_id}|balanced") < probabilities[split]

        if keep_balanced:
            write_row(balanced_writers[split], row)
            balanced_stats["split_counter"][split] += 1
            balanced_stats["label_counter"][label] += 1
            balanced_stats["label_by_split"][split][label] += 1
            balanced_stats["scenario_by_split"][split][scenario] += 1

    for handle in natural_writers.values():
        handle.close()

    for handle in balanced_writers.values():
        handle.close()

    first_pass["unique_group_keys"] = len(first_pass["unique_group_keys"])
    first_pass["group_by_split"] = {k: len(v) for k, v in first_pass["group_by_split"].items()}

    return {
        "input": str(RETRIEVAL_INPUT),
        "natural_outputs": {
            "train": str(OUT_DIR / "retrieval_natural_train.jsonl"),
            "val": str(OUT_DIR / "retrieval_natural_val.jsonl"),
            "test": str(OUT_DIR / "retrieval_natural_test.jsonl"),
        },
        "balanced_outputs": {
            "train": str(OUT_DIR / "retrieval_balanced_train.jsonl"),
            "val": str(OUT_DIR / "retrieval_balanced_val.jsonl"),
            "test": str(OUT_DIR / "retrieval_balanced_test.jsonl"),
        },
        "negative_ratio": RETRIEVAL_NEGATIVE_RATIO,
        "suppress_sampling_probabilities": probabilities,
        "natural_stats": first_pass,
        "balanced_stats": balanced_stats,
    }


def convert_sets_and_counters(obj):
    if isinstance(obj, set):
        return len(obj)
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, defaultdict):
        return {k: convert_sets_and_counters(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: convert_sets_and_counters(v) for k, v in obj.items()}
    return obj


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    operation_report = split_operation_dataset()
    retrieval_report = split_retrieval_dataset()

    report = {
        "version": "v3.24",
        "split_policy": {
            "group_key": "source_file + question_id",
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
            "reason": "All examples derived from the same benchmark question are kept in the same split to reduce leakage."
        },
        "operation_governance": operation_report,
        "retrieval_governance": retrieval_report,
    }

    report = convert_sets_and_counters(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
