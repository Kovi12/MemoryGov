import json
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import ijson


BASE_DIR = Path(__file__).resolve().parent


def stable_id(*parts: str) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def format_session(session: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in session:
        role = normalize_text(msg.get("role", "unknown"))
        content = normalize_text(msg.get("content", ""))

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines).strip()


def infer_memory_type(question_type: str, session_text: str) -> str:
    qtype = (question_type or "").lower()

    if "preference" in qtype:
        return "preference"

    if "temporal" in qtype:
        return "episodic"

    if "knowledge-update" in qtype:
        return "semantic"

    if "assistant" in qtype:
        return "procedural"

    if "user" in qtype:
        return "semantic"

    if any(x in session_text.lower() for x in ["prefer", "favorite", "like", "dislike", "usually", "style"]):
        return "preference"

    return "semantic"


def infer_sensitivity(text: str) -> str:
    lowered = text.lower()

    sensitive_markers = [
        "medical",
        "doctor",
        "diagnosis",
        "anxiety",
        "depression",
        "therapy",
        "password",
        "credit card",
        "bank",
        "salary",
        "ssn",
        "passport",
        "religion",
        "transgender",
        "sexual",
        "political",
        "adoption",
        "insurance",
    ]

    if any(marker in lowered for marker in sensitive_markers):
        return "medium"

    return "low"


def make_memory_record(
    memory_id: str,
    session_id: str,
    session_date: Optional[str],
    session_text: str,
    source_file: str,
    question_type: str,
) -> Dict[str, Any]:
    memory_type = infer_memory_type(question_type, session_text)

    return {
        "memory_id": memory_id,
        "memory_text": session_text,
        "memory_type": memory_type,
        "source_dataset": "LongMemEval",
        "source_file": source_file,
        "source_type": "conversation_session",
        "source_turns": [session_id],
        "source_trust": "benchmark",
        "subject": "user",
        "entities": [],
        "created_at": session_date,
        "last_confirmed_at": session_date,
        "valid_from": session_date,
        "valid_to": None,
        "sensitivity": infer_sensitivity(session_text),
        "confidence": 1.0,
        "status": "active",
        "version_id": 1,
        "supersedes": [],
        "superseded_by": [],
        "derived_from": [session_id],
        "principal_scope": "user_only",
        "retrieval_success_count": 0,
        "retrieval_failure_count": 0,
        "metadata": {
            "question_type": question_type,
            "original_session_id": session_id,
        },
    }


def make_retrieval_example(
    source_file: str,
    item: Dict[str, Any],
    session_index: int,
    session_id: str,
    session_date: Optional[str],
    session_text: str,
    is_answer_session: bool,
) -> Dict[str, Any]:
    question_id = item["question_id"]
    question_type = item.get("question_type", "unknown")
    question = normalize_text(item.get("question"))
    answer = normalize_text(item.get("answer"))
    question_date = item.get("question_date")

    memory_id = "lme_mem_" + stable_id(source_file, session_id, session_text)
    memory_record = make_memory_record(
        memory_id=memory_id,
        session_id=session_id,
        session_date=session_date,
        session_text=session_text,
        source_file=source_file,
        question_type=question_type,
    )

    primary_decision = "RETRIEVE" if is_answer_session else "SUPPRESS"

    if is_answer_session:
        reason = "This session is listed in answer_session_ids and is required evidence for the question."
    else:
        reason = "This session is part of the haystack but is not listed as answer evidence for this question, so it should not enter context for this query."

    return {
        "example_id": "lme_ret_" + stable_id(source_file, question_id, session_id, str(session_index)),
        "source_dataset": "LongMemEval",
        "source_file": source_file,
        "split_group": {
            "question_id": question_id,
            "session_id": session_id,
            "source_file": source_file,
        },
        "task_level": "retrieval_governance",
        "scenario_type": question_type,
        "failure_type": None if is_answer_session else "irrelevant_or_non_evidence_memory",
        "history_context": [],
        "candidate_memory": {
            "text": session_text,
            "memory_type": memory_record["memory_type"],
            "source_type": "conversation_session",
            "source_trust": "benchmark",
            "sensitivity": memory_record["sensitivity"],
            "confidence": 1.0,
            "valid_from": session_date,
            "valid_to": None,
        },
        "existing_memories": [memory_record],
        "current_query": question,
        "expected_answer": answer,
        "gold": {
            "primary_decision": primary_decision,
            "secondary_decision": None,
            "target_memory_ids": [memory_id],
            "new_status_for_targets": None,
            "should_retrieve_candidate": bool(is_answer_session),
            "should_retrieve_existing": bool(is_answer_session),
            "reason": reason,
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": question_type == "temporal-reasoning",
            "requires_contradiction_resolution": question_type == "knowledge-update",
            "requires_sensitivity_reasoning": memory_record["sensitivity"] != "low",
            "requires_appropriateness_reasoning": True,
            "requires_provenance_reasoning": True,
        },
        "annotation_metadata": {
            "label_source": "benchmark_derived",
            "human_audited": False,
            "confidence": 1.0,
        },
    }


def make_write_example(
    source_file: str,
    item: Dict[str, Any],
    session_index: int,
    session_id: str,
    session_date: Optional[str],
    session_text: str,
) -> Dict[str, Any]:
    question_id = item["question_id"]
    question_type = item.get("question_type", "unknown")
    question = normalize_text(item.get("question"))
    answer = normalize_text(item.get("answer"))

    memory_id = "lme_mem_" + stable_id(source_file, session_id, session_text)
    memory_record = make_memory_record(
        memory_id=memory_id,
        session_id=session_id,
        session_date=session_date,
        session_text=session_text,
        source_file=source_file,
        question_type=question_type,
    )

    return {
        "example_id": "lme_write_" + stable_id(source_file, question_id, session_id, str(session_index)),
        "source_dataset": "LongMemEval",
        "source_file": source_file,
        "split_group": {
            "question_id": question_id,
            "session_id": session_id,
            "source_file": source_file,
        },
        "task_level": "operation_governance",
        "scenario_type": question_type,
        "failure_type": None,
        "history_context": [],
        "candidate_memory": {
            "text": session_text,
            "memory_type": memory_record["memory_type"],
            "source_type": "conversation_session",
            "source_trust": "benchmark",
            "sensitivity": memory_record["sensitivity"],
            "confidence": 1.0,
            "valid_from": session_date,
            "valid_to": None,
        },
        "existing_memories": [],
        "current_query": question,
        "expected_answer": answer,
        "gold": {
            "primary_decision": "WRITE",
            "secondary_decision": "RETRIEVE",
            "target_memory_ids": [],
            "new_status_for_targets": None,
            "should_retrieve_candidate": True,
            "should_retrieve_existing": None,
            "reason": "This session is later required as evidence for a benchmark question, so it represents useful long-term memory.",
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": question_type == "temporal-reasoning",
            "requires_contradiction_resolution": question_type == "knowledge-update",
            "requires_sensitivity_reasoning": memory_record["sensitivity"] != "low",
            "requires_appropriateness_reasoning": False,
            "requires_provenance_reasoning": True,
        },
        "annotation_metadata": {
            "label_source": "benchmark_derived",
            "human_audited": False,
            "confidence": 1.0,
        },
    }


def stream_longmemeval(path: Path):
    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


def convert_file(
    input_path: Path,
    retrieval_out,
    write_out,
    max_questions: Optional[int] = None,
    include_suppressed: bool = True,
):
    source_file = input_path.name

    stats = {
        "questions": 0,
        "retrieval_examples": 0,
        "write_examples": 0,
        "retrieve_labels": 0,
        "suppress_labels": 0,
        "question_types": {},
    }

    seen_write_memory_ids = set()

    for q_idx, item in enumerate(stream_longmemeval(input_path)):
        if max_questions is not None and q_idx >= max_questions:
            break

        question_id = item.get("question_id")
        question_type = item.get("question_type", "unknown")
        answer_session_ids = set(item.get("answer_session_ids", []))
        haystack_session_ids = item.get("haystack_session_ids", [])
        haystack_dates = item.get("haystack_dates", [])
        haystack_sessions = item.get("haystack_sessions", [])

        stats["questions"] += 1
        stats["question_types"][question_type] = stats["question_types"].get(question_type, 0) + 1

        if not question_id or not haystack_sessions:
            continue

        for s_idx, session in enumerate(haystack_sessions):
            if not isinstance(session, list):
                continue

            session_id = haystack_session_ids[s_idx] if s_idx < len(haystack_session_ids) else f"{question_id}_session_{s_idx}"
            session_date = haystack_dates[s_idx] if s_idx < len(haystack_dates) else None
            session_text = format_session(session)

            if not session_text:
                continue

            is_answer_session = session_id in answer_session_ids

            if include_suppressed or is_answer_session:
                retrieval_example = make_retrieval_example(
                    source_file=source_file,
                    item=item,
                    session_index=s_idx,
                    session_id=session_id,
                    session_date=session_date,
                    session_text=session_text,
                    is_answer_session=is_answer_session,
                )
                retrieval_out.write(json.dumps(retrieval_example, ensure_ascii=False) + "\n")
                stats["retrieval_examples"] += 1

                if is_answer_session:
                    stats["retrieve_labels"] += 1
                else:
                    stats["suppress_labels"] += 1

            if is_answer_session:
                memory_key = stable_id(source_file, session_id, session_text)
                if memory_key not in seen_write_memory_ids:
                    write_example = make_write_example(
                        source_file=source_file,
                        item=item,
                        session_index=s_idx,
                        session_id=session_id,
                        session_date=session_date,
                        session_text=session_text,
                    )
                    write_out.write(json.dumps(write_example, ensure_ascii=False) + "\n")
                    seen_write_memory_ids.add(memory_key)
                    stats["write_examples"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "raw/longmemeval/longmemeval_oracle.json",
            "raw/longmemeval/longmemeval_s_cleaned.json",
            "raw/longmemeval/longmemeval_m_cleaned.json",
        ],
    )
    parser.add_argument(
        "--output_dir",
        default="processed",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--no_suppressed",
        action="store_true",
    )
    args = parser.parse_args()

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_path = output_dir / "longmemeval_retrieval_governance.jsonl"
    write_path = output_dir / "longmemeval_operation_write.jsonl"
    stats_path = output_dir / "longmemeval_conversion_stats.json"

    all_stats = {}

    with open(retrieval_path, "w", encoding="utf-8") as retrieval_out, open(write_path, "w", encoding="utf-8") as write_out:
        for input_rel in args.inputs:
            input_path = BASE_DIR / input_rel
            if not input_path.exists():
                print(f"Missing input file: {input_path}")
                continue

            print(f"Converting {input_path}")
            stats = convert_file(
                input_path=input_path,
                retrieval_out=retrieval_out,
                write_out=write_out,
                max_questions=args.max_questions,
                include_suppressed=not args.no_suppressed,
            )
            all_stats[input_path.name] = stats
            print(json.dumps(stats, indent=2))

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved retrieval examples to: {retrieval_path}")
    print(f"Saved write examples to: {write_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
