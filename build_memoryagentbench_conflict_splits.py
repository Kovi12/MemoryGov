import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_NATURAL = BASE_DIR / "processed/memoryagentbench_conflict_retrieval_governance_natural_qwen32b.jsonl"
DEFAULT_BALANCED = BASE_DIR / "processed/memoryagentbench_conflict_retrieval_governance_balanced_qwen32b.jsonl"
DEFAULT_OUT_DIR = BASE_DIR / "processed/splits_memoryagentbench_conflict"
DEFAULT_REPORT = DEFAULT_OUT_DIR / "memoryagentbench_conflict_split_report.json"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def group_key(row):
    return json.dumps(row.get("split_group", {}), sort_keys=True, ensure_ascii=False)


def assign_group_splits(groups, train_ratio=0.8, val_ratio=0.1):
    ordered = sorted(groups, key=stable_hash)
    n = len(ordered)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    split_map = {}
    for idx, group in enumerate(ordered):
        if idx < n_train:
            split_map[group] = "train"
        elif idx < n_train + n_val:
            split_map[group] = "val"
        else:
            split_map[group] = "test"

    return split_map


def collect_all_groups(paths):
    groups = set()

    for path in paths:
        for row in read_jsonl(path):
            groups.add(group_key(row))

    return groups


def init_stats(input_path):
    return {
        "input": str(input_path),
        "total_examples": 0,
        "split_counter": Counter(),
        "label_counter": Counter(),
        "label_by_split": defaultdict(Counter),
        "scenario_by_split": defaultdict(Counter),
        "subset_by_split": defaultdict(Counter),
        "group_by_split": defaultdict(set)
    }


def open_output_handles(out_dir, prefix):
    paths = {
        "train": out_dir / f"{prefix}_train.jsonl",
        "val": out_dir / f"{prefix}_val.jsonl",
        "test": out_dir / f"{prefix}_test.jsonl"
    }

    handles = {}
    for split, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        handles[split] = open(path, "w", encoding="utf-8")

    return paths, handles


def close_handles(handles):
    for handle in handles.values():
        handle.close()


def stream_split_file(input_path, out_dir, prefix, split_map):
    paths, handles = open_output_handles(out_dir, prefix)
    stats = init_stats(input_path)

    for idx, row in enumerate(read_jsonl(input_path), start=1):
        gkey = group_key(row)
        split = split_map[gkey]

        handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")

        label = row.get("gold", {}).get("primary_decision", "unknown")
        scenario = row.get("scenario_type", "unknown")
        subset = row.get("source_subset", "unknown")

        stats["total_examples"] += 1
        stats["split_counter"][split] += 1
        stats["label_counter"][label] += 1
        stats["label_by_split"][split][label] += 1
        stats["scenario_by_split"][split][scenario] += 1
        stats["subset_by_split"][split][subset] += 1
        stats["group_by_split"][split].add(gkey)

        if idx % 25000 == 0:
            print(f"[{prefix}] processed={idx}", flush=True)

    close_handles(handles)

    final_stats = {
        "input": stats["input"],
        "outputs": {split: str(path) for split, path in paths.items()},
        "total_examples": stats["total_examples"],
        "split_counter": dict(stats["split_counter"]),
        "label_counter": dict(stats["label_counter"]),
        "label_by_split": {k: dict(v) for k, v in stats["label_by_split"].items()},
        "scenario_by_split": {k: dict(v) for k, v in stats["scenario_by_split"].items()},
        "subset_by_split": {k: dict(v) for k, v in stats["subset_by_split"].items()},
        "group_by_split": {k: len(v) for k, v in stats["group_by_split"].items()}
    }

    return final_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-input", default=str(DEFAULT_NATURAL))
    parser.add_argument("--balanced-input", default=str(DEFAULT_BALANCED))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    natural_input = Path(args.natural_input)
    balanced_input = Path(args.balanced_input)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_groups = collect_all_groups([natural_input, balanced_input])
    split_map = assign_group_splits(all_groups)

    group_split_counter = Counter(split_map.values())

    natural_stats = stream_split_file(
        input_path=natural_input,
        out_dir=out_dir,
        prefix="retrieval_natural",
        split_map=split_map
    )

    balanced_stats = stream_split_file(
        input_path=balanced_input,
        out_dir=out_dir,
        prefix="retrieval_balanced",
        split_map=split_map
    )

    report = {
        "version": "v3.30",
        "split_policy": {
            "group_key": "full split_group JSON",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "reason": "All candidate facts derived from the same MemoryAgentBench QA item are kept in the same split. Natural and balanced files use the same split map."
        },
        "group_stats": {
            "total_groups": len(all_groups),
            "group_split_counter": dict(group_split_counter)
        },
        "retrieval_governance": {
            "natural_stats": natural_stats,
            "balanced_stats": balanced_stats
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
