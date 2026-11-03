import json
from collections import defaultdict
from pathlib import Path


BASE = Path("processed/operation_expansion_memoryagentbench_v2")
INPUTS = {
    "clean": BASE / "clean.jsonl",
    "review": BASE / "review.jsonl",
    "all": BASE / "all.jsonl",
}

MAX_PER_GROUP = 10


def read_jsonl(path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact(row):
    return {
        "example_id": row.get("example_id"),
        "status": row.get("annotation_metadata", {}).get("status"),
        "label": row.get("gold", {}).get("primary_decision"),
        "candidate": row.get("candidate_memory", {}).get("text"),
        "existing": [m.get("text") for m in row.get("existing_memories", [])],
        "target_memory_ids": row.get("gold", {}).get("target_memory_ids", []),
        "reason": row.get("gold", {}).get("reason"),
        "derivation": row.get("annotation_metadata", {}).get("derivation"),
        "confidence": row.get("annotation_metadata", {}).get("confidence"),
        "flags": row.get("annotation_metadata", {}).get("review_flags", []),
        "source_file": row.get("source_file"),
        "scenario_type": row.get("scenario_type"),
        "question": row.get("annotation_metadata", {}).get("question"),
        "expected_answer": row.get("expected_answer"),
    }


def main():
    output_dir = BASE / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, path in INPUTS.items():
        rows_by_key = defaultdict(list)

        for row in read_jsonl(path):
            label = row.get("gold", {}).get("primary_decision")
            derivation = row.get("annotation_metadata", {}).get("derivation")
            flags = row.get("annotation_metadata", {}).get("review_flags", [])
            key = f"{label}__{derivation}"
            if flags:
                key += "__flags_" + "_".join(flags[:3])
            if len(rows_by_key[key]) < MAX_PER_GROUP:
                rows_by_key[key].append(compact(row))

        out_path = output_dir / f"{split_name}_examples_by_label_derivation.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows_by_key, f, indent=2, ensure_ascii=False)

        print("=" * 100)
        print(split_name, path)
        print("groups:", len(rows_by_key))
        print("output:", out_path)

        for key, examples in rows_by_key.items():
            print("\n" + key)
            for ex in examples[:3]:
                print(json.dumps(ex, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
