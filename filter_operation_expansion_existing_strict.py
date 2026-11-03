import json
import re
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent

INPUT_CLEAN = BASE / "processed/operation_expansion_existing/operation_existing_expansion_clean.jsonl"
INPUT_REVIEW = BASE / "processed/operation_expansion_existing/operation_existing_expansion_review.jsonl"

OUT_DIR = BASE / "processed/operation_expansion_existing_strict"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ACCEPTED = OUT_DIR / "accepted.jsonl"
OUT_REVIEW = OUT_DIR / "review_remaining.jsonl"
OUT_REJECTED = OUT_DIR / "rejected.jsonl"
OUT_REPORT = OUT_DIR / "report.json"


SENSITIVE_TERMS = {
    "lgbtq",
    "transgender",
    "gay",
    "lesbian",
    "bisexual",
    "sexuality",
    "sexual orientation",
    "gender identity",
    "mental health",
    "therapy",
    "therapist",
    "counseling",
    "counselling",
    "depression",
    "anxiety",
    "trauma",
    "addiction",
    "rehab",
    "doctor",
    "medical",
    "diagnosis",
    "diagnosed",
    "illness",
    "disease",
    "medication",
    "pregnancy",
    "pregnant",
    "fertility",
    "religion",
    "religious",
    "political",
    "immigration",
    "legal issue",
    "lawsuit",
    "criminal",
    "bankruptcy",
    "debt",
    "salary",
    "income",
}

STRONG_UPDATE_MARKERS = {
    "replaced",
    "fixed",
    "resolved",
    "repaired",
    "working flawlessly",
    "working now",
    "now working",
    "no longer",
    "stopped",
    "switched from",
    "switched to",
    "changed from",
    "changed to",
    "moved from",
    "moved to",
    "cancelled",
    "canceled",
    "is now fixed",
    "has been fixed",
    "was fixed",
    "was replaced",
    "has been replaced",
    "is now working",
}

WEAK_UPDATE_ONLY_MARKERS = {
    "currently",
    "interested",
    "planning",
    "thinking",
    "considering",
    "has led",
    "has experience",
    "has not picked up",
    "not picked up",
}


def read_jsonl(path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text):
    return set(norm(text).split())


def label_of(row):
    return row.get("gold", {}).get("primary_decision")


def confidence_of(row):
    return float(row.get("annotation_metadata", {}).get("confidence", 0.0) or 0.0)


def derivation_of(row):
    return row.get("annotation_metadata", {}).get("derivation")


def candidate_text(row):
    return (row.get("candidate_memory", {}).get("text") or "").strip()


def existing_texts(row):
    return [
        (m.get("text") or "").strip()
        for m in row.get("existing_memories", []) or []
        if (m.get("text") or "").strip()
    ]


def sensitivity(row):
    return (row.get("candidate_memory", {}).get("sensitivity") or "").lower()


def memory_type(row):
    return (row.get("candidate_memory", {}).get("memory_type") or "").lower()


def has_existing_target(row):
    return bool(row.get("existing_memories")) and bool(row.get("gold", {}).get("target_memory_ids"))


def shared_count(a, b):
    return len(tokens(a) & tokens(b))


def contains_sensitive_term(text):
    t = norm(text)
    for term in SENSITIVE_TERMS:
        if norm(term) in t:
            return True
    return False


def has_strong_update_marker(text):
    t = norm(text)
    return any(norm(marker) in t for marker in STRONG_UPDATE_MARKERS)


def only_has_weak_update_marker(text):
    t = norm(text)
    return any(norm(marker) in t for marker in WEAK_UPDATE_ONLY_MARKERS)


def is_strict_quarantine(row):
    if label_of(row) != "QUARANTINE":
        return False, "not_quarantine"

    cand = candidate_text(row)

    if not cand:
        return False, "missing_candidate_text"

    if sensitivity(row) != "high" and memory_type(row) != "sensitive":
        return False, "not_high_or_sensitive_type"

    if not contains_sensitive_term(cand):
        return False, "no_explicit_sensitive_term"

    return True, "accepted_strict_quarantine"


def is_strict_merge(row):
    if label_of(row) != "MERGE":
        return False, "not_merge"

    if derivation_of(row) != "longmemeval_near_duplicate_memory_pair":
        return False, "bad_derivation"

    if confidence_of(row) < 0.84:
        return False, "low_confidence"

    if not has_existing_target(row):
        return False, "missing_existing_target"

    cand = candidate_text(row)
    exs = existing_texts(row)

    if not cand or not exs:
        return False, "missing_text"

    cand_norm = norm(cand)

    for ex in exs:
        ex_norm = norm(ex)

        if not ex_norm or not cand_norm:
            continue

        if ex_norm in cand_norm and len(cand_norm.split()) > len(ex_norm.split()):
            return True, "accepted_strict_merge_existing_contained_in_candidate"

    return False, "not_true_containment_merge"


def has_problem_resolution_pattern(candidate, existing):
    cand = norm(candidate)
    ex = norm(existing)

    existing_problem_markers = {
        "issue",
        "problem",
        "not functioning",
        "not working",
        "broken",
        "flat tire",
        "error",
        "malfunction",
        "failed",
        "failing",
        "trouble",
        "difficulty",
    }

    candidate_resolution_markers = {
        "fixed",
        "resolved",
        "repaired",
        "replaced",
        "working flawlessly",
        "working now",
        "now working",
        "is now working",
        "has been fixed",
        "was fixed",
        "has been replaced",
        "was replaced",
    }

    return (
        any(marker in ex for marker in existing_problem_markers)
        and any(marker in cand for marker in candidate_resolution_markers)
    )


def has_direct_state_change_pattern(candidate, existing):
    cand = norm(candidate)
    ex = norm(existing)

    direct_markers = {
        "no longer",
        "stopped",
        "switched from",
        "changed from",
        "moved from",
        "cancelled",
        "canceled",
    }

    return any(marker in cand for marker in direct_markers) and shared_count(candidate, existing) >= 3


def is_user_centered(row):
    cand = candidate_text(row).lower()
    exs = " ".join(existing_texts(row)).lower()

    if cand.startswith("user ") or cand.startswith("user's "):
        return True

    if "user " in cand or "user's " in cand:
        return True

    if exs.startswith("user ") or exs.startswith("user's "):
        return True

    return False


def is_strict_update(row):
    if label_of(row) != "UPDATE":
        return False, "not_update"

    if derivation_of(row) != "longmemeval_temporal_or_change_marker_pair":
        return False, "bad_derivation"

    if confidence_of(row) < 0.76:
        return False, "low_confidence"

    if not has_existing_target(row):
        return False, "missing_existing_target"

    cand = candidate_text(row)
    exs = existing_texts(row)

    if not cand or not exs:
        return False, "missing_text"

    if not is_user_centered(row):
        return False, "not_user_centered"

    if len(norm(cand).split()) > 24:
        return False, "candidate_too_broad_or_aggregate"

    for ex in exs:
        if shared_count(cand, ex) < 3:
            continue

        if has_problem_resolution_pattern(cand, ex):
            return True, "accepted_strict_update_problem_resolution"

        if has_direct_state_change_pattern(cand, ex):
            return True, "accepted_strict_update_direct_state_change"

    return False, "not_clear_supersession_update"

def classify(row):
    label = label_of(row)

    if label == "QUARANTINE":
        return is_strict_quarantine(row)

    if label == "MERGE":
        return is_strict_merge(row)

    if label == "UPDATE":
        return is_strict_update(row)

    if label == "ASK_USER":
        return False, "ask_user_held_for_policy_definition"

    if label == "FORGET":
        return False, "forget_held_for_manual_or_synthetic"

    return False, f"unsupported_label::{label}"


def summarize(rows):
    label_counter = Counter()
    source_counter = Counter()
    derivation_counter = Counter()
    reason_counter = Counter()
    sensitivity_counter = Counter()
    memory_type_counter = Counter()

    for row in rows:
        label_counter[label_of(row)] += 1
        source_counter[row.get("source_dataset")] += 1
        derivation_counter[derivation_of(row)] += 1
        reason_counter[row.get("filter_reason")] += 1
        sensitivity_counter[sensitivity(row)] += 1
        memory_type_counter[memory_type(row)] += 1

    return {
        "total": len(rows),
        "label_counter": dict(label_counter),
        "source_dataset_counter": dict(source_counter),
        "derivation_counter": dict(derivation_counter),
        "filter_reason_counter": dict(reason_counter),
        "sensitivity_counter": dict(sensitivity_counter),
        "memory_type_counter": dict(memory_type_counter),
    }


def main():
    accepted = []
    review = []
    rejected = []
    seen = set()

    input_rows = []
    input_rows.extend(read_jsonl(INPUT_CLEAN) or [])
    input_rows.extend(read_jsonl(INPUT_REVIEW) or [])

    for row in input_rows:
        ok, reason = classify(row)

        key = (
            label_of(row),
            norm(candidate_text(row)),
            tuple(sorted(norm(x) for x in existing_texts(row))),
        )

        if key in seen:
            row["filter_status"] = "rejected"
            row["filter_reason"] = "duplicate_after_filtering"
            rejected.append(row)
            continue

        seen.add(key)

        row["filter_status"] = "accepted" if ok else "review_remaining"
        row["filter_reason"] = reason

        if ok:
            accepted.append(row)
        else:
            review.append(row)

    write_jsonl(OUT_ACCEPTED, accepted)
    write_jsonl(OUT_REVIEW, review)
    write_jsonl(OUT_REJECTED, rejected)

    report = {
        "version": "v3.51",
        "task": "filter_operation_expansion_existing_strict",
        "purpose": "Strictly filter existing benchmark-derived operation expansion after v3.50 produced false-positive MERGE and weak UPDATE examples.",
        "inputs": {
            "clean": str(INPUT_CLEAN),
            "review": str(INPUT_REVIEW),
        },
        "outputs": {
            "accepted": str(OUT_ACCEPTED),
            "review_remaining": str(OUT_REVIEW),
            "rejected": str(OUT_REJECTED),
            "report": str(OUT_REPORT),
        },
        "policy": {
            "quarantine": "Accept only explicit sensitive content, not all LoCoMo sensitive-type memories.",
            "merge": "Accept only cases where the existing memory is contained in the candidate and the candidate adds extra detail.",
            "update": "Accept only cases with strong update/supersession markers and enough lexical overlap with the target memory.",
            "ask_user": "Held out until final consent policy is defined.",
            "forget": "Held out because ambiguous ordinary uses of 'forget' are unsafe as deletion labels.",
        },
        "accepted": summarize(accepted),
        "review_remaining": summarize(review),
        "rejected": summarize(rejected),
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
