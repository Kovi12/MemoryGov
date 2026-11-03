import json
import hashlib
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent


def stable_hash(*parts):
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_atomic_example(row, mem, mem_index):
    primary_decision = "WRITE" if mem.get("should_write") is True else "IGNORE"

    memory_id = "lme_atomic_mem_" + stable_hash(
        row.get("task_id"),
        row.get("question_id"),
        row.get("session_id"),
        mem_index,
        mem.get("memory_text", ""),
    )

    example_id = "lme_atomic_op_" + stable_hash(
        row.get("task_id"),
        mem_index,
        mem.get("memory_text", ""),
        primary_decision,
    )

    reason = mem.get("reason", "")

    return {
        "example_id": example_id,
        "source_dataset": "LongMemEval",
        "source_file": row.get("source_file"),
        "source_extraction_file": "longmemeval_candidate_extractions_raw_qwen32b.jsonl",
        "split_group": {
            "question_id": row.get("question_id"),
            "session_id": row.get("session_id"),
            "task_id": row.get("task_id"),
        },
        "task_level": "operation_governance",
        "scenario_type": row.get("scenario_type"),
        "failure_type": None if primary_decision == "WRITE" else "low_utility_or_temporary_memory",
        "history_context": [],
        "candidate_memory": {
            "text": mem.get("memory_text", ""),
            "memory_type": mem.get("memory_type", "unknown"),
            "source_type": "llm_extracted_from_conversation_session",
            "source_trust": "benchmark_plus_llm_extraction",
            "sensitivity": mem.get("sensitivity", "unknown"),
            "confidence": None,
            "valid_from": row.get("candidate_valid_from"),
            "valid_to": None,
            "stability": mem.get("stability", "unknown"),
            "evidence_span": mem.get("evidence_span", ""),
        },
        "existing_memories": [],
        "current_query": row.get("current_query"),
        "expected_answer": row.get("expected_answer"),
        "gold": {
            "primary_decision": primary_decision,
            "secondary_decision": None,
            "target_memory_ids": [],
            "new_status_for_targets": None,
            "should_retrieve_candidate": None,
            "should_retrieve_existing": None,
            "reason": reason,
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": row.get("scenario_type") == "temporal-reasoning",
            "requires_contradiction_resolution": row.get("scenario_type") == "knowledge-update",
            "requires_sensitivity_reasoning": mem.get("sensitivity") != "low",
            "requires_appropriateness_reasoning": primary_decision == "IGNORE",
            "requires_provenance_reasoning": True,
        },
        "annotation_metadata": {
            "label_source": "llm_extracted_from_benchmark_evidence",
            "extractor_model": row.get("model"),
            "valid_extraction_schema": row.get("valid_json_schema"),
            "human_audited": False,
            "source_task_id": row.get("task_id"),
            "memory_index": mem_index,
            "generation_seconds": row.get("generation_seconds"),
        },
        "memory_record_preview": {
            "memory_id": memory_id,
            "memory_text": mem.get("memory_text", ""),
            "memory_type": mem.get("memory_type", "unknown"),
            "source_dataset": "LongMemEval",
            "source_type": "llm_extracted_from_conversation_session",
            "source_turns": [row.get("session_id")],
            "source_trust": "benchmark_plus_llm_extraction",
            "subject": "user",
            "entities": [],
            "created_at": row.get("candidate_valid_from"),
            "last_confirmed_at": row.get("candidate_valid_from"),
            "valid_from": row.get("candidate_valid_from"),
            "valid_to": None,
            "sensitivity": mem.get("sensitivity", "unknown"),
            "confidence": None,
            "status": "active" if primary_decision == "WRITE" else "ignored",
            "version_id": 1,
            "supersedes": [],
            "superseded_by": [],
            "derived_from": [row.get("task_id"), row.get("session_id")],
            "principal_scope": "user_only",
            "retrieval_success_count": 0,
            "retrieval_failure_count": 0,
            "metadata": {
                "scenario_type": row.get("scenario_type"),
                "evidence_span": mem.get("evidence_span", ""),
                "stability": mem.get("stability", "unknown"),
            },
        },
    }


def main():
    input_path = BASE_DIR / "processed/longmemeval_candidate_extractions_raw_qwen32b.jsonl"
    output_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_qwen32b.jsonl"
    report_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_report.json"

    rows_out = []

    total_rows = 0
    skipped_invalid_rows = 0
    total_memories = 0

    decision_counter = Counter()
    memory_type_counter = Counter()
    sensitivity_counter = Counter()
    stability_counter = Counter()
    scenario_counter = Counter()

    seen_example_ids = set()
    duplicate_example_ids = 0

    for row in read_jsonl(input_path):
        total_rows += 1

        if not row.get("valid_json_schema", False):
            skipped_invalid_rows += 1
            continue

        memories = ((row.get("parsed_output") or {}).get("memories") or [])

        for mem_index, mem in enumerate(memories):
            example = build_atomic_example(row, mem, mem_index)

            ex_id = example["example_id"]
            if ex_id in seen_example_ids:
                duplicate_example_ids += 1
                continue
            seen_example_ids.add(ex_id)

            rows_out.append(example)

            total_memories += 1
            decision_counter[example["gold"]["primary_decision"]] += 1
            memory_type_counter[example["candidate_memory"]["memory_type"]] += 1
            sensitivity_counter[example["candidate_memory"]["sensitivity"]] += 1
            stability_counter[example["candidate_memory"]["stability"]] += 1
            scenario_counter[example["scenario_type"]] += 1

    write_jsonl(output_path, rows_out)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "total_extraction_rows": total_rows,
        "skipped_invalid_extraction_rows": skipped_invalid_rows,
        "output_atomic_examples": len(rows_out),
        "duplicate_example_ids_removed": duplicate_example_ids,
        "decision_counter": dict(decision_counter),
        "memory_type_counter": dict(memory_type_counter),
        "sensitivity_counter": dict(sensitivity_counter),
        "stability_counter": dict(stability_counter),
        "scenario_counter": dict(scenario_counter),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved atomic operation examples to: {output_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
