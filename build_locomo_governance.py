import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "raw/locomo/data/locomo10.json"
PROCESSED_DIR = BASE_DIR / "processed"

OPERATION_OUTPUT = PROCESSED_DIR / "locomo_operation_write.jsonl"
RETRIEVAL_OUTPUT = PROCESSED_DIR / "locomo_retrieval_governance.jsonl"
REPORT_OUTPUT = PROCESSED_DIR / "locomo_governance_report.json"

DECISIONS = {
    "WRITE",
    "IGNORE",
    "UPDATE",
    "MERGE",
    "SUPPRESS",
    "FORGET",
    "RETRIEVE",
    "QUARANTINE",
    "ASK_USER",
}

MEMORY_TYPES = {
    "episodic",
    "semantic",
    "preference",
    "procedural",
    "sensitive",
    "authorization",
    "project_state",
}


def stable_hash(text, n=16):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def classify_memory_type(text, source_kind):
    t = text.lower()

    sensitive_keywords = [
        "transgender", "lgbtq", "religious", "religion", "mental health",
        "accident", "injury", "doctor", "medical", "therapy", "adoption",
        "family", "son", "daughter", "kids", "children"
    ]

    preference_keywords = [
        "likes", "loves", "enjoys", "prefers", "favorite", "finds it",
        "is interested in", "wants", "hopes", "dream", "goal"
    ]

    project_keywords = [
        "plans to", "is planning", "is preparing", "is working on",
        "researching", "applied", "application", "process"
    ]

    procedural_keywords = [
        "usually", "always", "routine", "habit", "uses", "takes time to",
        "continues to", "works on"
    ]

    if any(k in t for k in sensitive_keywords):
        return "sensitive"

    if source_kind == "event":
        return "episodic"

    if any(k in t for k in project_keywords):
        return "project_state"

    if any(k in t for k in preference_keywords):
        return "preference"

    if any(k in t for k in procedural_keywords):
        return "procedural"

    return "semantic"


def classify_sensitivity(text, memory_type):
    t = text.lower()

    high_keywords = [
        "transition", "transgender", "medical", "doctor", "injury",
        "accident", "traumatic", "religious conservatives"
    ]

    medium_keywords = [
        "lgbtq", "mental health", "adoption", "family", "kids", "son",
        "daughter", "religion", "support group", "mentor"
    ]

    if memory_type == "sensitive":
        if any(k in t for k in high_keywords):
            return "high"
        return "medium"

    if any(k in t for k in high_keywords):
        return "high"

    if any(k in t for k in medium_keywords):
        return "medium"

    return "low"


def classify_stability(text, source_kind):
    t = text.lower()

    temporary_keywords = [
        "yesterday", "last night", "recently", "this week", "last week",
        "today", "tomorrow", "soon", "currently"
    ]

    if source_kind == "event":
        return "long_term"

    if any(k in t for k in temporary_keywords):
        return "session_relevant"

    return "long_term"


def get_sessions(conversation):
    sessions = []

    for key, value in conversation.items():
        if not re.fullmatch(r"session_\d+", key):
            continue

        session_num = int(key.split("_")[1])
        date_key = f"session_{session_num}_date_time"
        date_value = conversation.get(date_key)

        if isinstance(value, list):
            sessions.append({
                "session_num": session_num,
                "session_key": key,
                "date": date_value,
                "turns": value,
            })

    sessions.sort(key=lambda x: x["session_num"])
    return sessions


def build_turn_index(sample):
    conversation = sample.get("conversation", {})
    sessions = get_sessions(conversation)

    turns = []
    by_dia_id = {}

    for session in sessions:
        for turn_idx, turn in enumerate(session["turns"]):
            dia_id = normalize_text(turn.get("dia_id"))
            speaker = normalize_text(turn.get("speaker"))
            text = normalize_text(turn.get("text"))

            if not dia_id or not text:
                continue

            row = {
                "dia_id": dia_id,
                "session_num": session["session_num"],
                "session_key": session["session_key"],
                "date": session["date"],
                "turn_index": turn_idx,
                "speaker": speaker,
                "text": text,
                "has_image": bool(turn.get("img_url")),
                "blip_caption": normalize_text(turn.get("blip_caption")),
                "query": normalize_text(turn.get("query")),
            }

            turns.append(row)
            by_dia_id[dia_id] = row

    return turns, by_dia_id


def parse_observation_item(item):
    if isinstance(item, str):
        return item, []

    if isinstance(item, list):
        if not item:
            return "", []

        text = normalize_text(item[0])
        refs = []

        for part in item[1:]:
            if isinstance(part, str):
                refs.append(part)
            elif isinstance(part, list):
                refs.extend(str(x) for x in part)

        return text, refs

    return "", []


def build_operation_example(sample_id, source_kind, memory_text, subject, session_key, session_date, evidence_refs):
    memory_text = normalize_text(memory_text)
    memory_type = classify_memory_type(memory_text, source_kind)
    sensitivity = classify_sensitivity(memory_text, memory_type)
    stability = classify_stability(memory_text, source_kind)

    key = f"locomo|operation|{sample_id}|{source_kind}|{session_key}|{subject}|{memory_text}"
    example_id = f"locomo_op_{stable_hash(key)}"

    return {
        "example_id": example_id,
        "source_dataset": "LoCoMo",
        "source_file": "locomo10.json",
        "split_group": {
            "sample_id": sample_id,
            "session_key": session_key,
            "source_file": "locomo10.json",
        },
        "task_level": "operation_governance",
        "scenario_type": f"locomo_{source_kind}",
        "failure_type": None,
        "history_context": [],
        "candidate_memory": {
            "text": memory_text,
            "memory_type": memory_type,
            "source_type": source_kind,
            "source_trust": "benchmark",
            "sensitivity": sensitivity,
            "confidence": 1.0,
            "valid_from": session_date,
            "valid_to": None,
        },
        "existing_memories": [],
        "current_query": "Should this candidate be written to long-term memory?",
        "expected_answer": None,
        "gold": {
            "primary_decision": "WRITE",
            "secondary_decision": None,
            "target_memory_ids": [],
            "new_status_for_targets": None,
            "should_retrieve_candidate": None,
            "should_retrieve_existing": None,
            "reason": "LoCoMo provides this item as a structured observation or event memory.",
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": False,
            "requires_contradiction_resolution": False,
            "requires_sensitivity_reasoning": sensitivity in {"medium", "high"},
            "requires_appropriateness_reasoning": False,
            "requires_provenance_reasoning": True,
        },
        "annotation_metadata": {
            "label_source": "benchmark_derived",
            "human_audited": False,
            "confidence": 1.0,
            "subject": subject,
            "session_key": session_key,
            "session_date": session_date,
            "evidence_refs": evidence_refs,
        },
    }


def build_retrieval_example(sample_id, qa_idx, qa, turn, decision):
    question = normalize_text(qa.get("question"))
    answer = qa.get("answer")
    evidence = qa.get("evidence") or []
    category = qa.get("category")

    candidate_text = f"{turn['speaker']}: {turn['text']}"
    if turn.get("has_image"):
        extra = []
        if turn.get("blip_caption"):
            extra.append(f"image_caption: {turn['blip_caption']}")
        if turn.get("query"):
            extra.append(f"image_query: {turn['query']}")
        if extra:
            candidate_text += "\n" + "\n".join(extra)

    key = f"locomo|retrieval|{sample_id}|qa{qa_idx}|{turn['dia_id']}|{decision}"
    example_id = f"locomo_ret_{stable_hash(key)}"

    return {
        "example_id": example_id,
        "source_dataset": "LoCoMo",
        "source_file": "locomo10.json",
        "split_group": {
            "sample_id": sample_id,
            "qa_idx": qa_idx,
            "source_file": "locomo10.json",
        },
        "task_level": "retrieval_governance",
        "scenario_type": f"locomo_qa_category_{category}",
        "failure_type": None if decision == "RETRIEVE" else "irrelevant_or_non_evidence_memory",
        "history_context": [],
        "candidate_memory": {
            "text": candidate_text,
            "memory_type": "episodic",
            "source_type": "conversation_turn",
            "source_trust": "benchmark",
            "sensitivity": classify_sensitivity(candidate_text, classify_memory_type(candidate_text, "turn")),
            "confidence": 1.0,
            "valid_from": turn.get("date"),
            "valid_to": None,
        },
        "existing_memories": [],
        "current_query": question,
        "expected_answer": answer,
        "gold": {
            "primary_decision": decision,
            "secondary_decision": None,
            "target_memory_ids": [],
            "new_status_for_targets": None,
            "should_retrieve_candidate": decision == "RETRIEVE",
            "should_retrieve_existing": None,
            "reason": "Dialogue turn is listed as evidence for the LoCoMo QA item." if decision == "RETRIEVE" else "Dialogue turn is not listed as evidence for the LoCoMo QA item.",
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": category in {2, 3, 4},
            "requires_contradiction_resolution": False,
            "requires_sensitivity_reasoning": classify_sensitivity(candidate_text, "semantic") in {"medium", "high"},
            "requires_appropriateness_reasoning": decision == "SUPPRESS",
            "requires_provenance_reasoning": True,
        },
        "annotation_metadata": {
            "label_source": "benchmark_derived",
            "human_audited": False,
            "confidence": 1.0,
            "sample_id": sample_id,
            "qa_idx": qa_idx,
            "dia_id": turn["dia_id"],
            "evidence": evidence,
            "qa_category": category,
        },
    }


def extract_operation_examples(sample):
    sample_id = sample.get("sample_id", "unknown_sample")
    rows = []

    observation = sample.get("observation") or {}
    for session_key, session_obs in observation.items():
        match = re.search(r"session_(\d+)_observation", session_key)
        session_num = match.group(1) if match else None
        session_date = None

        if session_num:
            session_date = sample.get("conversation", {}).get(f"session_{session_num}_date_time")

        if not isinstance(session_obs, dict):
            continue

        for subject, items in session_obs.items():
            if not isinstance(items, list):
                continue

            for item in items:
                text, refs = parse_observation_item(item)
                if not text:
                    continue

                rows.append(build_operation_example(
                    sample_id=sample_id,
                    source_kind="observation",
                    memory_text=text,
                    subject=subject,
                    session_key=session_key,
                    session_date=session_date,
                    evidence_refs=refs,
                ))

    event_summary = sample.get("event_summary") or {}
    for session_key, session_events in event_summary.items():
        match = re.search(r"events_session_(\d+)", session_key)
        session_num = match.group(1) if match else None
        session_date = None

        if isinstance(session_events, dict):
            session_date = session_events.get("date")

        if session_num and not session_date:
            session_date = sample.get("conversation", {}).get(f"session_{session_num}_date_time")

        if not isinstance(session_events, dict):
            continue

        for subject, items in session_events.items():
            if subject == "date":
                continue
            if not isinstance(items, list):
                continue

            for text in items:
                text = normalize_text(text)
                if not text:
                    continue

                rows.append(build_operation_example(
                    sample_id=sample_id,
                    source_kind="event",
                    memory_text=text,
                    subject=subject,
                    session_key=session_key,
                    session_date=session_date,
                    evidence_refs=[],
                ))

    return rows


def extract_retrieval_examples(sample, negative_ratio, seed):
    rng = random.Random(seed)
    sample_id = sample.get("sample_id", "unknown_sample")
    turns, by_dia_id = build_turn_index(sample)
    qa_items = sample.get("qa") or []

    rows = []

    all_dia_ids = [t["dia_id"] for t in turns]

    for qa_idx, qa in enumerate(qa_items):
        evidence = qa.get("evidence") or []
        evidence = [normalize_text(x) for x in evidence if normalize_text(x)]

        positive_dia_ids = [d for d in evidence if d in by_dia_id]
        if not positive_dia_ids:
            continue

        for dia_id in positive_dia_ids:
            rows.append(build_retrieval_example(
                sample_id=sample_id,
                qa_idx=qa_idx,
                qa=qa,
                turn=by_dia_id[dia_id],
                decision="RETRIEVE",
            ))

        negative_pool = [d for d in all_dia_ids if d not in set(positive_dia_ids)]
        rng.shuffle(negative_pool)

        n_negatives = min(len(negative_pool), max(1, len(positive_dia_ids) * negative_ratio))
        for dia_id in negative_pool[:n_negatives]:
            rows.append(build_retrieval_example(
                sample_id=sample_id,
                qa_idx=qa_idx,
                qa=qa,
                turn=by_dia_id[dia_id],
                decision="SUPPRESS",
            ))

    return rows


def validate_decision(row):
    decision = row.get("gold", {}).get("primary_decision")
    if decision not in DECISIONS:
        return False
    memory_type = row.get("candidate_memory", {}).get("memory_type")
    if memory_type not in MEMORY_TYPES:
        return False
    if not row.get("candidate_memory", {}).get("text"):
        return False
    if not row.get("current_query"):
        return False
    return True


def write_outputs(operation_rows, retrieval_rows):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for path in [OPERATION_OUTPUT, RETRIEVAL_OUTPUT]:
        if path.exists():
            path.unlink()

    op_seen = set()
    ret_seen = set()

    op_written = 0
    ret_written = 0

    for row in operation_rows:
        if not validate_decision(row):
            continue
        example_id = row["example_id"]
        if example_id in op_seen:
            continue
        op_seen.add(example_id)
        append_jsonl(OPERATION_OUTPUT, row)
        op_written += 1

    for row in retrieval_rows:
        if not validate_decision(row):
            continue
        example_id = row["example_id"]
        if example_id in ret_seen:
            continue
        ret_seen.add(example_id)
        append_jsonl(RETRIEVAL_OUTPUT, row)
        ret_written += 1

    return op_written, ret_written


def build_report(operation_rows, retrieval_rows, op_written, ret_written, negative_ratio):
    def summarize(rows):
        label_counter = Counter()
        scenario_counter = Counter()
        memory_type_counter = Counter()
        sensitivity_counter = Counter()
        source_type_counter = Counter()
        group_counter = Counter()

        for row in rows:
            label_counter[row.get("gold", {}).get("primary_decision", "UNKNOWN")] += 1
            scenario_counter[row.get("scenario_type", "UNKNOWN")] += 1
            candidate = row.get("candidate_memory") or {}
            memory_type_counter[candidate.get("memory_type", "UNKNOWN")] += 1
            sensitivity_counter[candidate.get("sensitivity", "UNKNOWN")] += 1
            source_type_counter[candidate.get("source_type", "UNKNOWN")] += 1
            group_counter[str(row.get("split_group", {}))] += 1

        return {
            "total_raw": len(rows),
            "label_counter": dict(label_counter),
            "scenario_counter": dict(scenario_counter),
            "memory_type_counter": dict(memory_type_counter),
            "sensitivity_counter": dict(sensitivity_counter),
            "source_type_counter": dict(source_type_counter),
            "unique_split_groups": len(group_counter),
        }

    report = {
        "version": "v3.26",
        "input": str(RAW_PATH),
        "operation_output": str(OPERATION_OUTPUT),
        "retrieval_output": str(RETRIEVAL_OUTPUT),
        "negative_ratio": negative_ratio,
        "operation_written": op_written,
        "retrieval_written": ret_written,
        "operation_summary_raw": summarize(operation_rows),
        "retrieval_summary_raw": summarize(retrieval_rows),
    }

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_PATH))
    parser.add_argument("--negative_ratio", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    data = read_json(input_path)

    operation_rows = []
    retrieval_rows = []

    for idx, sample in enumerate(data):
        operation_rows.extend(extract_operation_examples(sample))
        retrieval_rows.extend(extract_retrieval_examples(
            sample=sample,
            negative_ratio=args.negative_ratio,
            seed=args.seed + idx,
        ))

    op_written, ret_written = write_outputs(operation_rows, retrieval_rows)
    report = build_report(operation_rows, retrieval_rows, op_written, ret_written, args.negative_ratio)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
