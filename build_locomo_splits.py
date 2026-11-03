import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

OPERATION_INPUT = PROCESSED_DIR / "locomo_operation_write.jsonl"
RETRIEVAL_INPUT = PROCESSED_DIR / "locomo_retrieval_governance.jsonl"
OUTPUT_DIR = PROCESSED_DIR / "splits_locomo"
REPORT_OUTPUT = OUTPUT_DIR / "locomo_split_report.json"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_group_key(row):
    split_group = row.get("split_group") or {}
    return json.dumps(split_group, sort_keys=True, ensure_ascii=False)


def row_label(row):
    return row.get("gold", {}).get("primary_decision", "UNKNOWN")


def deterministic_group_split(rows, train_ratio=0.8, val_ratio=0.1, seed=42):
    group_to_rows = defaultdict(list)

    for row in rows:
        group_to_rows[stable_group_key(row)].append(row)

    groups = list(group_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    split_rows = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group, group_rows in group_to_rows.items():
        if group in train_groups:
            split_rows["train"].extend(group_rows)
        elif group in val_groups:
            split_rows["val"].extend(group_rows)
        elif group in test_groups:
            split_rows["test"].extend(group_rows)
        else:
            raise RuntimeError(f"Unassigned group: {group}")

    return split_rows, {
        "total_groups": n,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
    }


def sample_balanced_retrieval(split_rows, negative_ratio=4, seed=42):
    balanced = {}
    rng = random.Random(seed)

    for split_name, rows in split_rows.items():
        positives = [r for r in rows if row_label(r) == "RETRIEVE"]
        negatives = [r for r in rows if row_label(r) == "SUPPRESS"]

        max_negatives = min(len(negatives), len(positives) * negative_ratio)
        sampled_negatives = rng.sample(negatives, max_negatives) if max_negatives > 0 else []

        out = positives + sampled_negatives
        rng.shuffle(out)
        balanced[split_name] = out

    return balanced


def summarize(rows_by_split):
    total = 0
    split_counter = Counter()
    label_counter = Counter()
    label_by_split = {}
    scenario_by_split = {}
    memory_type_counter = Counter()
    sensitivity_counter = Counter()
    source_type_counter = Counter()
    group_by_split = {}

    for split_name, rows in rows_by_split.items():
        total += len(rows)
        split_counter[split_name] = len(rows)

        labels = Counter()
        scenarios = Counter()
        groups = set()

        for row in rows:
            label = row_label(row)
            labels[label] += 1
            label_counter[label] += 1

            scenario = row.get("scenario_type", "UNKNOWN")
            scenarios[scenario] += 1

            candidate = row.get("candidate_memory") or {}
            memory_type_counter[candidate.get("memory_type", "UNKNOWN")] += 1
            sensitivity_counter[candidate.get("sensitivity", "UNKNOWN")] += 1
            source_type_counter[candidate.get("source_type", "UNKNOWN")] += 1

            groups.add(stable_group_key(row))

        label_by_split[split_name] = dict(labels)
        scenario_by_split[split_name] = dict(scenarios)
        group_by_split[split_name] = len(groups)

    return {
        "total": total,
        "split_counter": dict(split_counter),
        "label_counter": dict(label_counter),
        "label_by_split": label_by_split,
        "scenario_by_split": scenario_by_split,
        "memory_type_counter": dict(memory_type_counter),
        "sensitivity_counter": dict(sensitivity_counter),
        "source_type_counter": dict(source_type_counter),
        "group_by_split": group_by_split,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation_input", default=str(OPERATION_INPUT))
    parser.add_argument("--retrieval_input", default=str(RETRIEVAL_INPUT))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--negative_ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    operation_rows = list(read_jsonl(Path(args.operation_input)))
    retrieval_rows = list(read_jsonl(Path(args.retrieval_input)))

    operation_splits, operation_group_stats = deterministic_group_split(
        operation_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    retrieval_natural_splits, retrieval_group_stats = deterministic_group_split(
        retrieval_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    retrieval_balanced_splits = sample_balanced_retrieval(
        retrieval_natural_splits,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )

    operation_outputs = {
        "train": output_dir / "operation_train.jsonl",
        "val": output_dir / "operation_val.jsonl",
        "test": output_dir / "operation_test.jsonl",
    }

    retrieval_natural_outputs = {
        "train": output_dir / "retrieval_natural_train.jsonl",
        "val": output_dir / "retrieval_natural_val.jsonl",
        "test": output_dir / "retrieval_natural_test.jsonl",
    }

    retrieval_balanced_outputs = {
        "train": output_dir / "retrieval_balanced_train.jsonl",
        "val": output_dir / "retrieval_balanced_val.jsonl",
        "test": output_dir / "retrieval_balanced_test.jsonl",
    }

    for split_name, path in operation_outputs.items():
        write_jsonl(path, operation_splits[split_name])

    for split_name, path in retrieval_natural_outputs.items():
        write_jsonl(path, retrieval_natural_splits[split_name])

    for split_name, path in retrieval_balanced_outputs.items():
        write_jsonl(path, retrieval_balanced_splits[split_name])

    report = {
        "version": "v3.27",
        "split_policy": {
            "group_key": "full split_group JSON",
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "reason": "All examples derived from the same LoCoMo memory source or QA item are kept in the same split.",
        },
        "operation_governance": {
            "input": str(Path(args.operation_input).resolve()),
            "outputs": {k: str(v.resolve()) for k, v in operation_outputs.items()},
            "group_stats": operation_group_stats,
            "stats": summarize(operation_splits),
        },
        "retrieval_governance": {
            "input": str(Path(args.retrieval_input).resolve()),
            "natural_outputs": {k: str(v.resolve()) for k, v in retrieval_natural_outputs.items()},
            "balanced_outputs": {k: str(v.resolve()) for k, v in retrieval_balanced_outputs.items()},
            "negative_ratio": args.negative_ratio,
            "group_stats": retrieval_group_stats,
            "natural_stats": summarize(retrieval_natural_splits),
            "balanced_stats": summarize(retrieval_balanced_splits),
        },
    }

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
