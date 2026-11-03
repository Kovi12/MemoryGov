import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    rows_by_label = defaultdict(list)

    for row in read_jsonl(args.input):
        label = row.get("target_label", "UNKNOWN")
        rows_by_label[label].append(row)

    rng = random.Random(args.seed)
    shard_a = []
    shard_b = []

    for label, rows in sorted(rows_by_label.items()):
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                shard_a.append(row)
            else:
                shard_b.append(row)

    rng.shuffle(shard_a)
    rng.shuffle(shard_b)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_a = out_dir / "tasks_shard_a.jsonl"
    path_b = out_dir / "tasks_shard_b.jsonl"
    report_path = out_dir / "split_report.json"

    write_jsonl(path_a, shard_a)
    write_jsonl(path_b, shard_b)

    def summarize(rows):
        c = Counter(row.get("target_label", "UNKNOWN") for row in rows)
        return {
            "total": len(rows),
            "label_counter": dict(c),
        }

    report = {
        "version": "v3.58",
        "input": args.input,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "shard_a": {
            "path": str(path_a),
            **summarize(shard_a),
        },
        "shard_b": {
            "path": str(path_b),
            **summarize(shard_b),
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
