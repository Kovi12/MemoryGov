import json
import hashlib
import sys
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_hash(*parts):
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def canonical_key_for_write(example):
    split = example.get("split_group", {})
    candidate = example.get("candidate_memory") or {}
    return (
        split.get("question_id"),
        split.get("session_id"),
        stable_hash(candidate.get("text", "")),
    )


def canonical_key_for_retrieval(example):
    split = example.get("split_group", {})
    candidate = example.get("candidate_memory") or {}
    gold = example.get("gold", {})
    return (
        split.get("question_id"),
        split.get("session_id"),
        stable_hash(candidate.get("text", "")),
        gold.get("primary_decision"),
    )


def add_canonicalization_metadata(example, duplicate_count=1):
    example["canonicalization_metadata"] = {
        "canonicalized": True,
        "duplicate_count_known_exactly": False,
        "duplicate_count": duplicate_count,
        "kept_source_file": example.get("source_file"),
        "canonicalization_rule": (
            "streaming first-seen deduplication by question/session/text hash/label; "
            "input order is raw conversion order, so oracle is encountered before small and medium variants"
        ),
    }
    return example


def canonicalize_streaming(input_path, output_path, key_fn, mode):
    seen_keys = set()

    input_examples = 0
    output_examples = 0
    duplicate_rows_removed = 0

    source_counter_before = Counter()
    source_counter_after = Counter()
    label_counter_before = Counter()
    label_counter_after = Counter()
    task_level_counter_before = Counter()
    task_level_counter_after = Counter()

    with open(output_path, "w", encoding="utf-8") as out:
        for ex in read_jsonl(input_path):
            input_examples += 1

            source_file = ex.get("source_file", "UNKNOWN")
            label = (ex.get("gold") or {}).get("primary_decision", "UNKNOWN")
            task_level = ex.get("task_level", "UNKNOWN")

            source_counter_before[source_file] += 1
            label_counter_before[label] += 1
            task_level_counter_before[task_level] += 1

            key = key_fn(ex)

            if key in seen_keys:
                duplicate_rows_removed += 1
                continue

            seen_keys.add(key)
            ex = add_canonicalization_metadata(ex)
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

            output_examples += 1
            source_counter_after[source_file] += 1
            label_counter_after[label] += 1
            task_level_counter_after[task_level] += 1

            if output_examples % 25000 == 0:
                print(
                    f"[{mode}] written={output_examples} read={input_examples} duplicates_removed={duplicate_rows_removed}",
                    file=sys.stderr,
                    flush=True,
                )

    return {
        "mode": mode,
        "input_examples": input_examples,
        "output_examples": output_examples,
        "duplicate_rows_removed": duplicate_rows_removed,
        "unique_keys": len(seen_keys),
        "source_counter_before": dict(source_counter_before),
        "source_counter_after": dict(source_counter_after),
        "label_counter_before": dict(label_counter_before),
        "label_counter_after": dict(label_counter_after),
        "task_level_counter_before": dict(task_level_counter_before),
        "task_level_counter_after": dict(task_level_counter_after),
    }


def save_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    processed_dir = BASE_DIR / "processed"

    write_input = processed_dir / "longmemeval_operation_write.jsonl"
    retrieval_input = processed_dir / "longmemeval_retrieval_governance.jsonl"

    write_output = processed_dir / "longmemeval_operation_write_canonical.jsonl"
    retrieval_output = processed_dir / "longmemeval_retrieval_governance_canonical.jsonl"

    report_path = processed_dir / "longmemeval_canonicalization_report.json"

    if not write_input.exists():
        raise FileNotFoundError(write_input)

    if not retrieval_input.exists():
        raise FileNotFoundError(retrieval_input)

    report = {}

    print("Canonicalizing WRITE examples...", file=sys.stderr, flush=True)
    report["write"] = canonicalize_streaming(
        input_path=write_input,
        output_path=write_output,
        key_fn=canonical_key_for_write,
        mode="write",
    )

    print("Canonicalizing RETRIEVAL examples...", file=sys.stderr, flush=True)
    report["retrieval"] = canonicalize_streaming(
        input_path=retrieval_input,
        output_path=retrieval_output,
        key_fn=canonical_key_for_retrieval,
        mode="retrieval",
    )

    save_report(report_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved canonical WRITE file: {write_output}")
    print(f"Saved canonical RETRIEVAL file: {retrieval_output}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
