import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def text_len(example):
    candidate = example.get("candidate_memory") or {}
    return len(candidate.get("text", ""))


def validate_file(path):
    print("\n" + "=" * 100)
    print(f"VALIDATING: {path}")
    print("=" * 100)

    total = 0
    ids = set()
    duplicate_ids = 0
    empty_candidate = 0
    empty_query = 0

    by_label = Counter()
    by_source_file = Counter()
    by_task_level = Counter()
    by_scenario = Counter()
    by_failure = Counter()
    by_sensitivity = Counter()
    by_memory_type = Counter()

    lengths = []
    label_by_source = defaultdict(Counter)
    scenario_by_label = defaultdict(Counter)

    sample_by_label = {}

    for ex in read_jsonl(path):
        total += 1

        ex_id = ex.get("example_id")
        if ex_id in ids:
            duplicate_ids += 1
        ids.add(ex_id)

        candidate = ex.get("candidate_memory") or {}
        candidate_text = candidate.get("text", "")
        current_query = ex.get("current_query")

        if not candidate_text:
            empty_candidate += 1
        if not current_query:
            empty_query += 1

        gold = ex.get("gold", {})
        label = gold.get("primary_decision", "UNKNOWN")
        source_file = ex.get("source_file", "UNKNOWN")
        task_level = ex.get("task_level", "UNKNOWN")
        scenario = ex.get("scenario_type", "UNKNOWN")
        failure = ex.get("failure_type", "NONE")
        sensitivity = candidate.get("sensitivity", "UNKNOWN")
        memory_type = candidate.get("memory_type", "UNKNOWN")

        by_label[label] += 1
        by_source_file[source_file] += 1
        by_task_level[task_level] += 1
        by_scenario[scenario] += 1
        by_failure[str(failure)] += 1
        by_sensitivity[sensitivity] += 1
        by_memory_type[memory_type] += 1

        label_by_source[source_file][label] += 1
        scenario_by_label[label][scenario] += 1

        length = text_len(ex)
        lengths.append(length)

        if label not in sample_by_label:
            sample_by_label[label] = ex

    print(f"Total examples: {total}")
    print(f"Unique example IDs: {len(ids)}")
    print(f"Duplicate example IDs: {duplicate_ids}")
    print(f"Empty candidate memory: {empty_candidate}")
    print(f"Empty current query: {empty_query}")

    print("\nLabels:")
    for k, v in by_label.most_common():
        print(f"  {k}: {v}")

    print("\nTask levels:")
    for k, v in by_task_level.most_common():
        print(f"  {k}: {v}")

    print("\nSource files:")
    for k, v in by_source_file.most_common():
        print(f"  {k}: {v}")

    print("\nLabels by source file:")
    for source_file, counts in label_by_source.items():
        print(f"  {source_file}: {dict(counts)}")

    print("\nScenario types:")
    for k, v in by_scenario.most_common():
        print(f"  {k}: {v}")

    print("\nFailure types:")
    for k, v in by_failure.most_common():
        print(f"  {k}: {v}")

    print("\nSensitivity:")
    for k, v in by_sensitivity.most_common():
        print(f"  {k}: {v}")

    print("\nMemory types:")
    for k, v in by_memory_type.most_common():
        print(f"  {k}: {v}")

    if lengths:
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)

        def pct(p):
            idx = int((p / 100) * (n - 1))
            return lengths_sorted[idx]

        print("\nCandidate text length in characters:")
        print(f"  min: {lengths_sorted[0]}")
        print(f"  p50: {pct(50)}")
        print(f"  p90: {pct(90)}")
        print(f"  p95: {pct(95)}")
        print(f"  p99: {pct(99)}")
        print(f"  max: {lengths_sorted[-1]}")

    sample_dir = BASE_DIR / "processed" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for label, ex in sample_by_label.items():
        out_path = sample_dir / f"{path.stem}_sample_{label.lower()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ex, f, indent=2, ensure_ascii=False)
        print(f"\nSaved sample for {label}: {out_path}")


def main():
    files = [
        BASE_DIR / "processed" / "longmemeval_retrieval_governance.jsonl",
        BASE_DIR / "processed" / "longmemeval_operation_write.jsonl",
    ]

    for path in files:
        if not path.exists():
            print(f"Missing file: {path}")
            continue
        validate_file(path)


if __name__ == "__main__":
    main()
