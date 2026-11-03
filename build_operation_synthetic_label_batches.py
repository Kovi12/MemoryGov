import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_COUNTS = {
    "IGNORE": 40,
    "UPDATE": 40,
    "MERGE": 40,
    "FORGET": 80,
    "ASK_USER": 60,
    "QUARANTINE": 40,
}

LABELS = ["IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]

LABEL_CONSTRAINTS = {
    "IGNORE": """
Additional constraints for IGNORE:
- The candidate must be user-centered but low-value, transient, generic, already obsolete, or not useful for future personalization.
- candidate_memory.text must start with "User" or "User's".
- The history_context must include a user message that supports the candidate.
- Do not create non-user-centered world facts.
- Do not create a fact about a random third party unless it is clearly not worth storing for the user.
- Do not include sensitive personal data.
- existing_memories should usually be empty.
- gold.primary_decision must be "IGNORE".
- gold.new_status_for_targets must be null.
- gold.target_memory_ids must be an empty list.
""",
    "UPDATE": """
Additional constraints for UPDATE:
- existing_memories must contain at least one target memory.
- The candidate must clearly revise, correct, replace, or supersede the existing memory.
- Use clear state-change language such as: "now", "no longer", "changed from", "changed to", "switched from", "switched to", "resolved", "fixed", "replaced", "cancelled", or "moved".
- Do not generate merely related information.
- Do not generate a separate event that only shares a topic with the existing memory.
- gold.primary_decision must be "UPDATE".
- gold.target_memory_ids must include the memory_id of the superseded memory.
- gold.new_status_for_targets must be "superseded".
""",
    "MERGE": """
Additional constraints for MERGE:
- existing_memories must contain at least one target memory.
- The candidate must be a near-duplicate, paraphrase, or complementary detail that should be combined with the existing memory.
- The candidate should preserve the old memory and add a small useful detail.
- Do not generate a separate event that only shares a topic.
- Do not generate contradictions.
- gold.primary_decision must be "MERGE".
- gold.target_memory_ids must include the memory_id of the memory being merged.
- gold.new_status_for_targets must be "merged".
""",
    "FORGET": """
Additional constraints for FORGET:
- The history_context must contain an explicit user deletion request.
- Use one of these exact deletion intents in the user message: "forget this", "delete this", "remove this from memory", "do not remember this", "erase this", "stop storing this", or "remove that memory".
- existing_memories must contain the memory being deleted.
- The candidate_memory.text should describe the memory deletion request or the targeted memory.
- Do not use "forget" in the ordinary sense, such as "I forgot my keys" or "I will never forget this trip".
- Do not generate vague deletion requests without a clear target.
- gold.primary_decision must be "FORGET".
- gold.target_memory_ids must include the memory_id of the deleted memory.
- gold.new_status_for_targets must be "deleted".
""",
    "ASK_USER": """
Additional constraints for ASK_USER:
- The candidate must be user-centered.
- candidate_memory.text must start with "User" or "User's".
- The history_context must include a user message that makes the candidate plausible but not clearly safe to store without confirmation.
- The candidate should involve mild privacy, ambiguous consent, uncertainty, third-party relation to the user, or unclear long-term usefulness.
- Good ASK_USER cases:
  - User mentions a friend's personal detail but does not ask the assistant to remember it.
  - User gives an unstable preference that may be temporary.
  - User mentions a mildly private life situation where consent to store is unclear.
  - User gives a detail that could help personalization but is not clearly intended as memory.
- Do not use high-sensitivity medical, financial, legal, political, religious, sexual orientation, gender identity, credentials, trauma, or addiction information. Those should be QUARANTINE, not ASK_USER.
- Do not generate world facts.
- Do not generate candidate memories about third parties unless the candidate explicitly links the detail to the user, for example "User's friend..." or "User mentioned that their brother...".
- existing_memories may be empty.
- gold.primary_decision must be "ASK_USER".
- gold.new_status_for_targets must be null.
""",
    "QUARANTINE": """
Additional constraints for QUARANTINE:
- The candidate must contain explicitly high-sensitivity personal information.
- Valid high-sensitivity categories include medical/health, financial account/transaction, legal issue, immigration status, religious belief, political belief, sexual orientation, gender identity, trauma, addiction, mental health, or private credentials.
- Do not generate low-sensitivity personal preferences.
- Do not generate vague private information without a clear sensitive category.
- gold.primary_decision must be "QUARANTINE".
- gold.new_status_for_targets must be null.
""",
}

DIVERSITY_HINTS = {
    "IGNORE": [
        "short-lived errand",
        "generic weather/news observation",
        "one-time casual comment",
        "temporary mood with no future relevance",
        "irrelevant world fact",
        "already expired appointment mention",
    ],
    "UPDATE": [
        "changed preference",
        "corrected personal fact",
        "resolved technical issue",
        "changed schedule",
        "moved address or location",
        "cancelled plan",
        "switched tools or services",
        "old health routine replaced by a new one",
    ],
    "MERGE": [
        "same restaurant with extra detail",
        "same project with added deadline",
        "same hobby with added frequency",
        "same travel plan with extra location detail",
        "same preference with clarification",
        "same recurring event with added time detail",
    ],
    "FORGET": [
        "delete an old address",
        "forget an old password",
        "remove a private relationship detail",
        "erase a memory about a stressful event",
        "stop storing an outdated phone number",
        "remove a third-party private detail",
        "delete an old financial detail",
        "forget a no-longer-relevant preference",
    ],
    "ASK_USER": [
        "third-party personal detail",
        "uncertain user preference",
        "mildly private social event",
        "ambiguous ownership of information",
        "unclear whether future personalization benefits",
        "personal but not highly sensitive information",
    ],
    "QUARANTINE": [
        "medical diagnosis",
        "therapy or mental health detail",
        "financial account or transaction",
        "legal problem",
        "religious belief",
        "political affiliation",
        "sexual orientation or gender identity",
        "immigration or residency detail",
    ],
}


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


def stable_id(*parts):
    text = "||".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rewrite_to_one_example(prompt):
    if not isinstance(prompt, str):
        return prompt

    replacements = [
        (r"exactly\s+\d+\s+examples", "exactly 1 example"),
        (r"exactly\s+five\s+examples", "exactly 1 example"),
        (r"exactly\s+two\s+examples", "exactly 1 example"),
        (r"generate\s+\d+\s+examples", "generate 1 example"),
        (r"generate\s+five\s+examples", "generate 1 example"),
        (r"generate\s+two\s+examples", "generate 1 example"),
        (r"create\s+\d+\s+examples", "create 1 example"),
        (r"create\s+five\s+examples", "create 1 example"),
        (r"create\s+two\s+examples", "create 1 example"),
        (r"produce\s+\d+\s+examples", "produce 1 example"),
        (r"produce\s+five\s+examples", "produce 1 example"),
        (r"produce\s+two\s+examples", "produce 1 example"),
        (r"\"examples\"\s+array\s+must\s+contain\s+exactly\s+\d+", "\"examples\" array must contain exactly 1"),
        (r"examples_requested\s*[:=]\s*\d+", "examples_requested: 1"),
    ]

    out = prompt
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    return out


def strengthen_prompt(prompt, label, hint, batch_name, variant_idx):
    prompt = rewrite_to_one_example(prompt)

    addition = f"""

CRITICAL OUTPUT COUNT RULE:
Generate exactly 1 example.
The top-level JSON object must be:
{{"examples": [ ... ]}}
The "examples" array must contain exactly 1 item.
After the single example is complete, close the JSON object immediately.
Do not generate a second example.

TARGET LABEL:
{label}

DIVERSITY HINT:
Use this scenario family, but do not copy it literally: {hint}

BATCH METADATA:
batch_name: {batch_name}
variant_idx: {variant_idx}

{LABEL_CONSTRAINTS[label]}

QUALITY RULES:
- The candidate must be user-centered unless the correct label is IGNORE for a non-user-centered item.
- Avoid duplicate wording across examples.
- Avoid benchmark-answer leakage.
- Do not mention that the example is synthetic inside the memory text.
- Keep the example realistic and concise.
- The candidate_memory.text must be user-centered and should start with "User" or "User's".
- The candidate must be user-centered unless the correct label is IGNORE for a non-user-centered item.
- Return only valid JSON.
"""
    return prompt.rstrip() + "\n" + addition.strip() + "\n"


def parse_counts(text):
    counts = dict(DEFAULT_COUNTS)

    if not text:
        return counts

    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid count item: {item}")
        label, value = item.split("=", 1)
        label = label.strip().upper()
        if label not in LABELS:
            raise ValueError(f"Unknown label in counts: {label}")
        counts[label] = int(value.strip())

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed/operation_synthetic_v1/tasks_pilot_2examples.jsonl")
    parser.add_argument("--output_dir", default="processed/operation_synthetic_v1/label_batches/batch1")
    parser.add_argument("--batch_name", default="batch1")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--counts", default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts = parse_counts(args.counts)

    by_label = defaultdict(list)
    for row in read_jsonl(args.input):
        label = row.get("target_label")
        if label in LABELS:
            by_label[label].append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    report = {
        "version": "v3.57",
        "task": "build_operation_synthetic_label_batches",
        "input": args.input,
        "output_dir": str(out_dir),
        "batch_name": args.batch_name,
        "seed": args.seed,
        "requested_counts": counts,
        "available_seed_tasks_by_label": {label: len(by_label[label]) for label in LABELS},
        "outputs": {},
    }

    for label in LABELS:
        seeds = by_label[label]
        if not seeds:
            raise ValueError(f"No seed tasks found for label {label}")

        rows = []
        requested = counts[label]
        hints = DIVERSITY_HINTS[label]

        for i in range(requested):
            base = dict(rng.choice(seeds))
            hint = hints[i % len(hints)]

            new_id = "synthetic_op_full_task_" + stable_id(args.batch_name, label, i, base.get("task_id"), hint)
            base["task_id"] = new_id
            base["target_label"] = label
            base["examples_requested"] = 1
            base["synthetic_batch"] = args.batch_name
            base["synthetic_batch_index"] = i
            base["synthetic_diversity_hint"] = hint
            base["prompt"] = strengthen_prompt(
                base.get("prompt", ""),
                label=label,
                hint=hint,
                batch_name=args.batch_name,
                variant_idx=i,
            )

            rows.append(base)
            all_rows.append(base)

        label_path = out_dir / f"tasks_{label.lower()}.jsonl"
        write_jsonl(label_path, rows)

        report["outputs"][label] = {
            "path": str(label_path),
            "count": len(rows),
        }

    all_path = out_dir / "tasks_all.jsonl"
    write_jsonl(all_path, all_rows)

    report["outputs"]["ALL"] = {
        "path": str(all_path),
        "count": len(all_rows),
    }

    report_path = out_dir / "task_batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
