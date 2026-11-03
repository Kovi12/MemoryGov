import json
import random
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "processed/operation_forget_targeted_aug_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(20260527)


TOPICS = [
    ("old phone number", "User's old phone number is stored in memory.", "sensitive", "medium"),
    ("old address", "User's old address is stored in memory.", "sensitive", "high"),
    ("salary", "User's salary is stored in memory.", "sensitive", "high"),
    ("medical appointment", "User had a medical appointment.", "sensitive", "high"),
    ("diagnosis", "User disclosed a medical diagnosis.", "sensitive", "high"),
    ("therapy plan", "User is considering therapy.", "sensitive", "high"),
    ("political discussion", "User mentioned a political discussion.", "sensitive", "high"),
    ("religious belief", "User discussed a private religious belief.", "sensitive", "high"),
    ("relationship breakup", "User talked about a breakup.", "sensitive", "medium"),
    ("family conflict", "User had a conflict with a family member.", "sensitive", "medium"),
    ("job search", "User is job hunting.", "sensitive", "medium"),
    ("financial stress", "User is experiencing financial stress.", "sensitive", "medium"),
    ("home address", "User's home address is stored in memory.", "sensitive", "high"),
    ("credit card detail", "User shared a credit card detail.", "sensitive", "high"),
    ("national ID detail", "User shared a national ID detail.", "sensitive", "high"),
    ("immigration status", "User disclosed immigration status.", "sensitive", "high"),
    ("sexual orientation", "User disclosed sexual orientation.", "sensitive", "high"),
    ("substance use", "User disclosed substance use.", "sensitive", "high"),
    ("mental health issue", "User disclosed a mental health issue.", "sensitive", "high"),
    ("private location routine", "User shared a private location routine.", "sensitive", "medium"),
    ("old preference for dark mode", "User prefers dark mode.", "preference", "low"),
    ("old preference for long answers", "User prefers long detailed answers.", "preference", "low"),
    ("old project deadline", "User's project deadline is in June.", "project_state", "low"),
    ("old travel preference", "User prefers window seats.", "preference", "low"),
    ("old coding preference", "User prefers code with many comments.", "preference", "low"),
]

REQUEST_TEMPLATES = [
    "Forget my {topic}.",
    "Please forget my {topic}.",
    "Delete the memory about my {topic}.",
    "Remove the memory about my {topic}.",
    "Stop remembering my {topic}.",
    "Do not remember my {topic}.",
    "Do not keep my {topic} in memory.",
    "Erase the stored memory about my {topic}.",
    "I want you to forget my {topic}.",
    "Please remove anything you saved about my {topic}.",
    "That memory about my {topic} should be deleted.",
    "I no longer want my {topic} stored.",
]

CANDIDATE_TEMPLATES = [
    "User asks to forget their {topic}.",
    "User asks to delete the stored memory about their {topic}.",
    "User requests removal of the memory about their {topic}.",
    "User asks the assistant not to remember their {topic}.",
    "User wants the stored memory about their {topic} erased.",
]

CONTEXT_PREFIXES = [
    "",
    "I shared this before, but now I changed my mind. ",
    "This is private, so please listen carefully. ",
    "I do not want this kept anymore. ",
    "For privacy reasons, ",
]


def make_input_text(history_context, candidate_memory, existing_memories, memory_type, sensitivity):
    history = " | ".join(f"{x['speaker']}: {x['text']}" for x in history_context) if history_context else "None"

    existing_parts = []
    for i, mem in enumerate(existing_memories, 1):
        existing_parts.append(
            f"{mem['memory_id']}: {mem['text']} "
            f"[type={mem['memory_type']}; sensitivity={mem['sensitivity']}; status={mem['status']}]"
        )

    existing_text = " | ".join(existing_parts) if existing_parts else "None"

    return f"""Task: operation_governance
Question: Decide what the memory system should do with the candidate memory.

Allowed labels:
WRITE, IGNORE, UPDATE, MERGE, FORGET, ASK_USER, QUARANTINE

History context:
{history}

Candidate memory:
{candidate_memory}

Candidate metadata:
type={memory_type}
sensitivity={sensitivity}

Existing memories:
{existing_text}

Decision target:
Choose the correct operation label for the candidate memory.""".strip()


def build_examples(n):
    rows = []

    seen = set()
    attempts = 0

    while len(rows) < n and attempts < n * 20:
        attempts += 1

        topic, existing_text, memory_type, sensitivity = RNG.choice(TOPICS)
        req_template = RNG.choice(REQUEST_TEMPLATES)
        cand_template = RNG.choice(CANDIDATE_TEMPLATES)
        prefix = RNG.choice(CONTEXT_PREFIXES)

        user_text = prefix + req_template.format(topic=topic)
        candidate_text = cand_template.format(topic=topic)

        key = (user_text, candidate_text, existing_text)
        if key in seen:
            continue
        seen.add(key)

        idx = len(rows) + 1
        example_id = f"forget_targeted_aug_v1_{idx:04d}"

        existing = [{
            "memory_id": "m1",
            "text": existing_text,
            "memory_type": memory_type,
            "sensitivity": sensitivity,
            "status": "active",
        }]

        history = [{"speaker": "user", "text": user_text}]

        text = make_input_text(history, candidate_text, existing, memory_type, sensitivity)

        row = {
            "example_id": example_id,
            "label": "FORGET",
            "target": "FORGET",
            "text": text,
            "input_text": text,
            "task_level": "operation_governance",
            "source_dataset": "TargetedForgetAugmentation",
            "source_file": "operation_forget_targeted_aug_v1",
            "scenario_type": "targeted_forget_explicit_deletion",
            "failure_type": "sensitivity_overrides_deletion_intent",
            "memory_type": memory_type,
            "sensitivity": sensitivity,
            "operation_v1_source": "targeted_forget_aug_v1_train_only",
            "raw": {
                "example_id": example_id,
                "source_dataset": "TargetedForgetAugmentation",
                "source_file": "operation_forget_targeted_aug_v1",
                "task_level": "operation_governance",
                "scenario_type": "targeted_forget_explicit_deletion",
                "failure_type": "sensitivity_overrides_deletion_intent",
                "history_context": history,
                "candidate_memory": {
                    "text": candidate_text,
                    "memory_type": memory_type,
                    "source_type": "targeted_synthetic",
                    "source_trust": "synthetic",
                    "sensitivity": sensitivity,
                    "confidence": 0.95,
                    "valid_from": None,
                    "valid_to": None,
                },
                "existing_memories": existing,
                "current_query": None,
                "expected_answer": None,
                "gold": {
                    "primary_decision": "FORGET",
                    "secondary_decision": None,
                    "target_memory_ids": ["m1"],
                    "new_status_for_targets": "deleted",
                    "should_retrieve_candidate": None,
                    "should_retrieve_existing": None,
                    "reason": "The user explicitly asks to delete or forget an existing memory. Deletion intent overrides sensitivity routing.",
                },
                "diagnostic_tags": {
                    "requires_temporal_reasoning": False,
                    "requires_contradiction_resolution": False,
                    "requires_sensitivity_reasoning": True,
                    "requires_appropriateness_reasoning": True,
                    "requires_provenance_reasoning": False,
                    "requires_deletion_intent_priority": True,
                },
                "annotation_metadata": {
                    "label_source": "targeted_forget_aug_v1",
                    "human_audited": False,
                    "confidence": 0.95,
                    "policy_rule": "explicit_deletion_request_overrides_sensitivity_routing",
                    "train_only": True,
                },
            },
        }

        rows.append(row)

    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rows = build_examples(300)

    write_jsonl(OUT_DIR / "forget_targeted_aug_train_only.jsonl", rows)

    report = {
        "version": "v3.84",
        "task": "build_operation_forget_targeted_aug_v1",
        "purpose": "Train-only targeted augmentation for explicit deletion/forgetting requests, especially sensitive content where FORGET should override ASK_USER/QUARANTINE.",
        "total": len(rows),
        "label_counter": dict(Counter(row["label"] for row in rows)),
        "memory_type_counter": dict(Counter(row["memory_type"] for row in rows)),
        "sensitivity_counter": dict(Counter(row["sensitivity"] for row in rows)),
        "outputs": {
            "train_only": str(OUT_DIR / "forget_targeted_aug_train_only.jsonl"),
            "report": str(OUT_DIR / "report.json"),
        },
        "policy_rule": "explicit_deletion_request_overrides_sensitivity_routing",
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
