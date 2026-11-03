import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_LABELS = {"IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"}
TARGET_REQUIRED = {"UPDATE", "MERGE", "FORGET"}


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


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts):
    text = "||".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_label(label):
    return clean_text(label).upper()


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def has_user_centering(text):
    t = f" {text.lower()} "
    markers = [" user ", " user's ", " the user ", " i ", " my ", " me ", " mine "]
    return any(m in t for m in markers)


def has_forget_intent(text, history):
    joined = (text + " " + " ".join(clean_text(x.get("text", "")) for x in history if isinstance(x, dict))).lower()
    patterns = [
        "delete",
        "remove",
        "forget that",
        "forget this",
        "do not remember",
        "don't remember",
        "stop storing",
        "clear this",
        "erase",
        "no longer remember",
        "not keep this"
    ]
    return any(p in joined for p in patterns)


def normalize_existing(existing):
    output = []
    for i, item in enumerate(as_list(existing)):
        if not isinstance(item, dict):
            continue
        text = clean_text(item.get("text"))
        if not text:
            continue
        memory_id = clean_text(item.get("memory_id")) or f"m{i + 1}"
        output.append({
            "memory_id": memory_id,
            "text": text,
            "memory_type": clean_text(item.get("memory_type")) or "semantic",
            "sensitivity": clean_text(item.get("sensitivity")) or "low",
            "status": clean_text(item.get("status")) or "active"
        })
    return output


def normalize_example(raw_task, ex, idx):
    if not isinstance(ex, dict):
        return None, ["example_not_object"]

    candidate = ex.get("candidate_memory", {}) or {}
    gold = ex.get("gold", {}) or {}
    metadata = ex.get("annotation_metadata", {}) or {}

    label = normalize_label(gold.get("primary_decision"))
    candidate_text = clean_text(candidate.get("text"))
    existing = normalize_existing(ex.get("existing_memories", []))
    target_ids = [clean_text(x) for x in as_list(gold.get("target_memory_ids")) if clean_text(x)]
    history = as_list(ex.get("history_context", []))

    row_id = "synthetic_op_" + stable_id(raw_task.get("task_id"), idx, label, candidate_text)

    norm = {
        "example_id": row_id,
        "source_dataset": "SyntheticAnchored",
        "source_file": "operation_synthetic_generation",
        "split_group": {
            "synthetic_task_id": raw_task.get("task_id"),
            "synthetic_example_idx": idx,
            "target_label": raw_task.get("target_label")
        },
        "task_level": "operation_governance",
        "scenario_type": clean_text(ex.get("scenario_type")) or f"synthetic_{label.lower()}",
        "failure_type": ex.get("failure_type"),
        "history_context": history if isinstance(history, list) else [],
        "candidate_memory": {
            "text": candidate_text,
            "memory_type": clean_text(candidate.get("memory_type")) or "semantic",
            "source_type": clean_text(candidate.get("source_type")) or "synthetic",
            "source_trust": clean_text(candidate.get("source_trust")) or "synthetic",
            "sensitivity": clean_text(candidate.get("sensitivity")) or "low",
            "confidence": float(candidate.get("confidence", 0.85) or 0.85),
            "valid_from": candidate.get("valid_from"),
            "valid_to": candidate.get("valid_to")
        },
        "existing_memories": existing,
        "current_query": clean_text(ex.get("current_query")) or "Should this candidate be written to long-term memory, ignored, updated, merged, forgotten, confirmed with the user, or quarantined?",
        "expected_answer": ex.get("expected_answer"),
        "gold": {
            "primary_decision": label,
            "secondary_decision": gold.get("secondary_decision"),
            "target_memory_ids": target_ids,
            "new_status_for_targets": gold.get("new_status_for_targets"),
            "should_retrieve_candidate": gold.get("should_retrieve_candidate"),
            "should_retrieve_existing": gold.get("should_retrieve_existing"),
            "reason": clean_text(gold.get("reason"))
        },
        "diagnostic_tags": ex.get("diagnostic_tags") if isinstance(ex.get("diagnostic_tags"), dict) else {
            "requires_temporal_reasoning": label == "UPDATE",
            "requires_contradiction_resolution": label in {"UPDATE", "MERGE"},
            "requires_sensitivity_reasoning": label in {"ASK_USER", "QUARANTINE"},
            "requires_appropriateness_reasoning": label in {"IGNORE", "ASK_USER", "QUARANTINE"},
            "requires_provenance_reasoning": False
        },
        "annotation_metadata": {
            "label_source": "synthetic_anchored",
            "human_audited": False,
            "confidence": float(metadata.get("confidence", 0.8) or 0.8),
            "synthetic_policy_case": clean_text(metadata.get("synthetic_policy_case")) or raw_task.get("seed_scenario"),
            "synthetic_task_id": raw_task.get("task_id"),
            "target_label": raw_task.get("target_label"),
            "model": raw_task.get("model"),
            "validation_flags": []
        }
    }

    return norm, []


def validate(row):
    flags = []
    label = row["gold"]["primary_decision"]
    candidate_text = row["candidate_memory"]["text"]
    existing = row["existing_memories"]
    target_ids = row["gold"]["target_memory_ids"]
    sensitivity = row["candidate_memory"]["sensitivity"]
    memory_type = row["candidate_memory"]["memory_type"]
    history = row.get("history_context", [])

    if label not in ALLOWED_LABELS:
        flags.append("invalid_label")

    if label != row["annotation_metadata"].get("target_label"):
        flags.append("label_mismatch_with_task")

    if not candidate_text:
        flags.append("missing_candidate_text")

    if not row["gold"].get("reason"):
        flags.append("missing_reason")

    if not has_user_centering(candidate_text):
        flags.append("weak_user_centering")

    if label in TARGET_REQUIRED:
        if not existing:
            flags.append("missing_existing_memory")
        if not target_ids:
            flags.append("missing_target_memory_ids")

    if label == "FORGET":
        if not has_forget_intent(candidate_text, history):
            flags.append("forget_without_explicit_delete_intent")
        if row["gold"].get("new_status_for_targets") not in {"deleted", "forgotten", "removed", "inactive"}:
            flags.append("forget_status_not_deleted")

    if label == "UPDATE":
        if row["gold"].get("new_status_for_targets") not in {"superseded", "updated", "inactive"}:
            flags.append("update_status_not_superseded")

    if label == "MERGE":
        if row["gold"].get("new_status_for_targets") not in {"merged", "deduplicated", "inactive"}:
            flags.append("merge_status_not_merged")

    if label == "QUARANTINE":
        if sensitivity != "high":
            flags.append("quarantine_not_high_sensitivity")
        if memory_type not in {"sensitive", "authorization", "semantic", "episodic"}:
            flags.append("unexpected_quarantine_memory_type")

    if label == "ASK_USER":
        if sensitivity == "high":
            flags.append("ask_user_high_sensitivity_should_probably_quarantine")

    confidence = row["annotation_metadata"].get("confidence", 0.0)
    if confidence < 0.7:
        flags.append("low_generation_confidence")

    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="processed/operation_synthetic_v1/pilot_validated")
    parser.add_argument("--max_clean_per_label", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    clean_rows = []
    review_rows = []
    rejected_rows = []

    seen = set()
    clean_per_label = Counter()
    raw_counter = Counter()
    flag_counter = Counter()
    label_counter_all = Counter()
    label_counter_clean = Counter()
    label_counter_review = Counter()

    for raw in read_jsonl(args.input):
        raw_counter["raw_rows"] += 1

        if not raw.get("valid_json_schema"):
            rejected_rows.append({
                "task_id": raw.get("task_id"),
                "reason": "invalid_raw_generation",
                "validation_error": raw.get("validation_error"),
                "raw_output": raw.get("raw_output", "")[:4000]
            })
            raw_counter["invalid_raw_generation"] += 1
            continue

        parsed = raw.get("parsed_output") or {}
        examples = parsed.get("examples", [])
        if not isinstance(examples, list):
            rejected_rows.append({
                "task_id": raw.get("task_id"),
                "reason": "missing_examples_list",
                "raw_output": raw.get("raw_output", "")[:4000]
            })
            raw_counter["missing_examples_list"] += 1
            continue

        for idx, ex in enumerate(examples):
            row, norm_errors = normalize_example(raw, ex, idx)
            if row is None:
                rejected_rows.append({
                    "task_id": raw.get("task_id"),
                    "reason": "normalization_failed",
                    "errors": norm_errors,
                    "example": ex
                })
                raw_counter["normalization_failed"] += 1
                continue

            key = (
                row["gold"]["primary_decision"],
                row["candidate_memory"]["text"].lower(),
                " || ".join(m["text"].lower() for m in row["existing_memories"])
            )

            if key in seen:
                row["annotation_metadata"]["validation_flags"] = ["duplicate_generated_example"]
                review_rows.append(row)
                flag_counter["duplicate_generated_example"] += 1
                label_counter_review[row["gold"]["primary_decision"]] += 1
                continue

            seen.add(key)
            flags = validate(row)
            row["annotation_metadata"]["validation_flags"] = flags

            all_rows.append(row)
            label = row["gold"]["primary_decision"]
            label_counter_all[label] += 1

            if flags:
                review_rows.append(row)
                label_counter_review[label] += 1
                for flag in flags:
                    flag_counter[flag] += 1
            else:
                if args.max_clean_per_label is not None and clean_per_label[label] >= args.max_clean_per_label:
                    row["annotation_metadata"]["validation_flags"] = ["max_clean_per_label_overflow"]
                    review_rows.append(row)
                    label_counter_review[label] += 1
                    flag_counter["max_clean_per_label_overflow"] += 1
                else:
                    clean_rows.append(row)
                    clean_per_label[label] += 1
                    label_counter_clean[label] += 1

    write_jsonl(output_dir / "all.jsonl", all_rows)
    write_jsonl(output_dir / "clean.jsonl", clean_rows)
    write_jsonl(output_dir / "review.jsonl", review_rows)
    write_jsonl(output_dir / "rejected.jsonl", rejected_rows)

    report = {
        "version": "v3.50",
        "task": "validate_operation_synthetic_generation",
        "input": str(Path(args.input).resolve()),
        "output_dir": str(output_dir.resolve()),
        "raw_counter": dict(raw_counter),
        "total_all": len(all_rows),
        "total_clean": len(clean_rows),
        "total_review": len(review_rows),
        "total_rejected": len(rejected_rows),
        "label_counter_all": dict(label_counter_all),
        "label_counter_clean": dict(label_counter_clean),
        "label_counter_review": dict(label_counter_review),
        "flag_counter": dict(flag_counter),
        "note": "Only clean examples are candidates for training. Review examples require inspection or stricter filtering."
    }

    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
