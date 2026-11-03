import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_LABELS = ["IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]


SCENARIOS = {
    "IGNORE": [
        "transient small talk with no long-term value",
        "assistant recommendation not about the user's durable preference",
        "generic world fact not centered on the user",
        "temporary one-off event that should not be stored",
        "unsupported benchmark-answer leakage",
        "low-value observation that does not help future personalization"
    ],
    "UPDATE": [
        "user changed a previous preference",
        "user corrected a previously stored fact",
        "a previous task/project state became obsolete",
        "a planned event was completed or cancelled",
        "a user's recurring schedule changed",
        "a previous health, job, travel, or study status was superseded"
    ],
    "MERGE": [
        "candidate is a paraphrase of an existing memory",
        "candidate adds a small complementary detail to an existing memory",
        "candidate and existing memory describe the same preference",
        "candidate and existing memory describe the same project",
        "candidate and existing memory describe the same recurring habit"
    ],
    "FORGET": [
        "user explicitly asks to delete a stored memory",
        "user says not to remember a specific fact anymore",
        "user asks the assistant to remove a sensitive memory",
        "user revokes permission to store a previously saved preference",
        "user asks to clear an outdated personal detail"
    ],
    "ASK_USER": [
        "candidate is medium-sensitive and should require consent",
        "candidate concerns third-party personal information",
        "candidate may be useful but privacy implications are unclear",
        "candidate involves finances, relationships, health, or identity but is not obviously high-risk",
        "candidate contains an ambiguous instruction about whether to remember it"
    ],
    "QUARANTINE": [
        "candidate contains high-sensitivity personal information",
        "candidate contains protected identity or sexuality information",
        "candidate contains precise private address, credential, or authorization information",
        "candidate contains high-risk medical or mental-health information",
        "candidate contains legal, political, religious, or minor-related sensitive information"
    ]
}


POLICY = {
    "IGNORE": "The candidate should not be stored. It is not useful, not user-centered, too transient, unsupported, or merely world knowledge.",
    "UPDATE": "The candidate should replace or supersede one or more existing memories. Existing memories must be provided and target_memory_ids must identify them.",
    "MERGE": "The candidate should be merged with one or more semantically overlapping existing memories. Existing memories must be provided and target_memory_ids must identify them.",
    "FORGET": "The user explicitly requests deletion or non-retention of one or more existing memories. Existing memories and target_memory_ids are mandatory.",
    "ASK_USER": "The memory may be useful but should not be stored without confirmation because consent, sensitivity, or ownership is ambiguous.",
    "QUARANTINE": "The memory is high-sensitivity and should be routed to protected/quarantined memory rather than ordinary memory."
}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def stable_id(*parts):
    text = "||".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_label(row):
    return row.get("gold", {}).get("primary_decision")


def compact_anchor(row):
    candidate = row.get("candidate_memory", {}) or {}
    existing = row.get("existing_memories", []) or []
    gold = row.get("gold", {}) or {}
    metadata = row.get("annotation_metadata", {}) or {}

    return {
        "label": gold.get("primary_decision"),
        "candidate_text": candidate.get("text", ""),
        "memory_type": candidate.get("memory_type", "semantic"),
        "sensitivity": candidate.get("sensitivity", "low"),
        "existing_memories": [
            {
                "memory_id": m.get("memory_id") or f"m{i + 1}",
                "text": m.get("text", ""),
                "memory_type": m.get("memory_type", "semantic"),
                "sensitivity": m.get("sensitivity", "low")
            }
            for i, m in enumerate(existing[:3])
        ],
        "reason": gold.get("reason", ""),
        "scenario_type": row.get("scenario_type", ""),
        "source_dataset": row.get("source_dataset", ""),
        "derivation": metadata.get("derivation", "")
    }


def load_anchors(paths):
    by_label = defaultdict(list)
    all_rows = []

    for path in paths:
        for row in read_jsonl(path):
            label = get_label(row)
            if not label:
                continue
            anchor = compact_anchor(row)
            by_label[label].append(anchor)
            all_rows.append(anchor)

    return by_label, all_rows


def build_prompt(label, scenario, examples_requested, anchors):
    schema = {
        "examples": [
            {
                "source_dataset": "SyntheticAnchored",
                "source_file": "operation_synthetic_generation",
                "task_level": "operation_governance",
                "scenario_type": f"synthetic_{label.lower()}",
                "failure_type": None,
                "history_context": [
                    {
                        "speaker": "user",
                        "text": "A short user message or previous conversation fragment that justifies the candidate."
                    }
                ],
                "candidate_memory": {
                    "text": "User-centered memory candidate written as a concise declarative sentence.",
                    "memory_type": "episodic|semantic|procedural|preference|project_state|sensitive|authorization",
                    "source_type": "synthetic",
                    "source_trust": "synthetic",
                    "sensitivity": "low|medium|high",
                    "confidence": 0.85,
                    "valid_from": None,
                    "valid_to": None
                },
                "existing_memories": [
                    {
                        "memory_id": "m1",
                        "text": "Existing memory text, required for UPDATE, MERGE, and FORGET.",
                        "memory_type": "episodic|semantic|procedural|preference|project_state|sensitive|authorization",
                        "sensitivity": "low|medium|high",
                        "status": "active"
                    }
                ],
                "current_query": "Should this candidate be written to long-term memory, ignored, updated, merged, forgotten, confirmed with the user, or quarantined?",
                "expected_answer": None,
                "gold": {
                    "primary_decision": label,
                    "secondary_decision": None,
                    "target_memory_ids": ["m1"],
                    "new_status_for_targets": None,
                    "should_retrieve_candidate": None,
                    "should_retrieve_existing": None,
                    "reason": "Short explanation of why this operation is correct."
                },
                "diagnostic_tags": {
                    "requires_temporal_reasoning": False,
                    "requires_contradiction_resolution": False,
                    "requires_sensitivity_reasoning": False,
                    "requires_appropriateness_reasoning": False,
                    "requires_provenance_reasoning": False
                },
                "annotation_metadata": {
                    "label_source": "synthetic_anchored",
                    "human_audited": False,
                    "confidence": 0.8,
                    "synthetic_policy_case": scenario
                }
            }
        ]
    }

    instructions = {
        "label": label,
        "policy": POLICY[label],
        "scenario": scenario,
        "examples_requested": examples_requested,
        "hard_constraints": [
            f"Every example must have gold.primary_decision exactly equal to {label}.",
            "Do not copy anchor examples verbatim.",
            "Use realistic user-centered memory situations.",
            "candidate_memory.text must be concise and user-centered.",
            "Do not use benchmark QA wording.",
            "Do not create world-fact memories unless they are explicitly tied to the user.",
            "Return valid JSON only.",
            "Return one top-level object with the key examples."
        ],
        "label_specific_constraints": {
            "IGNORE": [
                "existing_memories may be empty.",
                "target_memory_ids must be empty.",
                "The reason must explain why the memory should not be stored."
            ],
            "UPDATE": [
                "existing_memories must contain at least one active memory.",
                "target_memory_ids must identify the memory being superseded.",
                "new_status_for_targets should be superseded.",
                "The candidate must clearly revise, correct, or replace the existing memory."
            ],
            "MERGE": [
                "existing_memories must contain at least one related active memory.",
                "target_memory_ids must identify the memory being merged with.",
                "new_status_for_targets should be merged.",
                "The candidate must be a duplicate, paraphrase, or complementary detail."
            ],
            "FORGET": [
                "history_context must contain an explicit user deletion or non-retention request.",
                "existing_memories must contain the memory to delete.",
                "target_memory_ids must identify the memory to delete.",
                "new_status_for_targets should be deleted.",
                "Do not create examples where forget means forgetting in ordinary speech."
            ],
            "ASK_USER": [
                "existing_memories may be empty.",
                "target_memory_ids may be empty.",
                "The reason must explain why confirmation is required before storage."
            ],
            "QUARANTINE": [
                "candidate_memory.sensitivity should usually be high.",
                "candidate_memory.memory_type should usually be sensitive or authorization.",
                "The reason must explain why ordinary memory storage is inappropriate."
            ]
        }[label],
        "anchors": anchors,
        "required_schema": schema
    }

    return json.dumps(instructions, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="processed/operation_synthetic_v1/tasks_pilot.jsonl")
    parser.add_argument("--report", default="processed/operation_synthetic_v1/tasks_pilot_report.json")
    parser.add_argument("--examples_per_label", type=int, default=120)
    parser.add_argument("--examples_per_task", type=int, default=5)
    parser.add_argument("--labels", default=",".join(ALLOWED_LABELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor_paths", nargs="*", default=[
        "processed/memorygovbench_current/operation_train.jsonl",
        "processed/memorygovbench_current/operation_val.jsonl",
        "processed/operation_expansion_existing/operation_existing_expansion_all.jsonl"
    ])
    args = parser.parse_args()

    random.seed(args.seed)

    labels = [x.strip().upper() for x in args.labels.split(",") if x.strip()]
    labels = [x for x in labels if x in ALLOWED_LABELS]

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    by_label, all_anchors = load_anchors(args.anchor_paths)

    tasks = []
    label_counter = Counter()

    for label in labels:
        task_count = math.ceil(args.examples_per_label / args.examples_per_task)
        label_anchors = by_label.get(label, [])
        fallback_anchors = by_label.get("WRITE", []) + by_label.get("IGNORE", []) + all_anchors

        for task_idx in range(task_count):
            scenario = random.choice(SCENARIOS[label])
            pool = label_anchors if len(label_anchors) >= 3 else fallback_anchors
            anchors = random.sample(pool, k=min(4, len(pool))) if pool else []

            task_id = "synthetic_op_task_" + stable_id(label, task_idx, scenario, args.seed)
            task = {
                "task_id": task_id,
                "task_type": "synthetic_operation_generation",
                "target_label": label,
                "examples_requested": args.examples_per_task,
                "seed_scenario": scenario,
                "anchor_count": len(anchors),
                "anchors": anchors,
                "prompt": build_prompt(label, scenario, args.examples_per_task, anchors),
                "annotation_metadata": {
                    "version": "v3.50",
                    "label_source": "synthetic_anchored_task",
                    "examples_per_label": args.examples_per_label,
                    "examples_per_task": args.examples_per_task,
                    "seed": args.seed
                }
            }
            tasks.append(task)
            label_counter[label] += args.examples_per_task

    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    report = {
        "version": "v3.50",
        "task": "build_operation_synthetic_generation_tasks",
        "output": str(output_path.resolve()),
        "total_tasks": len(tasks),
        "requested_examples_by_label": dict(label_counter),
        "labels": labels,
        "examples_per_label": args.examples_per_label,
        "examples_per_task": args.examples_per_task,
        "anchor_paths": args.anchor_paths,
        "anchor_counter_by_label": {k: len(v) for k, v in sorted(by_label.items())},
        "note": "These are generation tasks only. Generated examples must be validated before any merge."
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
