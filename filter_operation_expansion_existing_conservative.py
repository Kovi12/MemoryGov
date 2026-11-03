import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent

INPUT_CLEAN = BASE / "processed/operation_expansion_existing/operation_existing_expansion_clean.jsonl"
INPUT_REVIEW = BASE / "processed/operation_expansion_existing/operation_existing_expansion_review.jsonl"

OUT_DIR = BASE / "processed/operation_expansion_existing_conservative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ACCEPTED = OUT_DIR / "accepted.jsonl"
OUT_REVIEW_REMAINING = OUT_DIR / "review_remaining.jsonl"
OUT_REJECTED = OUT_DIR / "rejected.jsonl"
OUT_REPORT = OUT_DIR / "report.json"


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


def label_of(row):
    return row.get("gold", {}).get("primary_decision")


def confidence_of(row):
    return float(row.get("annotation_metadata", {}).get("confidence", 0.0) or 0.0)


def derivation_of(row):
    return row.get("annotation_metadata", {}).get("derivation")


def flags_of(row):
    return set(row.get("annotation_metadata", {}).get("review_flags", []) or [])


def candidate_text(row):
    return (row.get("candidate_memory", {}).get("text") or "").strip()


def existing_texts(row):
    return [
        (m.get("text") or "").strip()
        for m in row.get("existing_memories", []) or []
        if (m.get("text") or "").strip()
    ]


def has_existing_target(row):
    return bool(row.get("existing_memories")) and bool(row.get("gold", {}).get("target_memory_ids"))


def token_set(text):
    cleaned = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return set("".join(cleaned).split())


def shared_token_count(a, b):
    return len(token_set(a) & token_set(b))


def has_update_marker(text):
    t = text.lower()
    markers = [
        "now",
        "no longer",
        "replaced",
        "changed",
        "switched",
        "moved",
        "started",
        "stopped",
        "updated",
        "resolved",
        "fixed",
        "working",
        "instead",
        "currently",
    ]
    return any(m in t for m in markers)


def is_conservative_merge(row):
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

    max_shared = max(shared_token_count(cand, ex) for ex in exs)
    if max_shared < 4:
        return False, "too_few_shared_tokens"

    return True, "accepted_conservative_merge"


def is_conservative_update(row):
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

    if not has_update_marker(cand):
        return False, "missing_update_marker"

    max_shared = max(shared_token_count(cand, ex) for ex in exs)
    if max_shared < 2:
        return False, "too_few_shared_tokens"

    return True, "accepted_conservative_update"


def is_accepted_clean(row):
    label = label_of(row)

    if label == "QUARANTINE":
        return True, "accepted_clean_quarantine"

    if label == "MERGE":
        ok, reason = is_conservative_merge(row)
        if ok:
            return True, reason
        return False, reason

    return False, f"clean_label_not_auto_accepted::{label}"


def classify_review(row):
    label = label_of(row)

    if label == "MERGE":
        return is_conservative_merge(row)

    if label == "UPDATE":
        return is_conservative_update(row)

    if label == "ASK_USER":
        return False, "ask_user_policy_relabel_keep_for_later"

    if label == "FORGET":
        return False, "forget_ambiguous_keep_for_synthetic_or_manual"

    return False, f"unsupported_review_label::{label}"


def summarize(rows):
    c = Counter()
    by_label = Counter()
    by_source = Counter()
    by_derivation = Counter()
    by_reason = Counter()

    for row in rows:
        c["total"] += 1
        by_label[label_of(row)] += 1
        by_source[row.get("source_dataset")] += 1
        by_derivation[derivation_of(row)] += 1
        by_reason[row.get("filter_reason")] += 1

    return {
        "total": c["total"],
        "label_counter": dict(by_label),
        "source_dataset_counter": dict(by_source),
        "derivation_counter": dict(by_derivation),
        "filter_reason_counter": dict(by_reason),
    }


def main():
    accepted = []
    review_remaining = []
    rejected = []

    seen_keys = set()

    for row in read_jsonl(INPUT_CLEAN):
        ok, reason = is_accepted_clean(row)
        row["filter_status"] = "accepted" if ok else "review_remaining"
        row["filter_reason"] = reason

        key = (
            label_of(row),
            candidate_text(row).lower(),
            tuple(sorted(x.lower() for x in existing_texts(row))),
        )

        if key in seen_keys:
            row["filter_status"] = "rejected"
            row["filter_reason"] = "duplicate_after_filtering"
            rejected.append(row)
            continue

        seen_keys.add(key)

        if ok:
            accepted.append(row)
        else:
            review_remaining.append(row)

    for row in read_jsonl(INPUT_REVIEW):
        ok, reason = classify_review(row)
        row["filter_status"] = "accepted" if ok else "review_remaining"
        row["filter_reason"] = reason

        key = (
            label_of(row),
            candidate_text(row).lower(),
            tuple(sorted(x.lower() for x in existing_texts(row))),
        )

        if key in seen_keys:
            row["filter_status"] = "rejected"
            row["filter_reason"] = "duplicate_after_filtering"
            rejected.append(row)
            continue

        seen_keys.add(key)

        if ok:
            accepted.append(row)
        else:
            review_remaining.append(row)

    write_jsonl(OUT_ACCEPTED, accepted)
    write_jsonl(OUT_REVIEW_REMAINING, review_remaining)
    write_jsonl(OUT_REJECTED, rejected)

    report = {
        "version": "v3.50",
        "task": "filter_operation_expansion_existing_conservative",
        "purpose": "Build a conservative accepted set from existing benchmark-derived operation expansion before synthetic data is merged.",
        "inputs": {
            "clean": str(INPUT_CLEAN),
            "review": str(INPUT_REVIEW),
        },
        "outputs": {
            "accepted": str(OUT_ACCEPTED),
            "review_remaining": str(OUT_REVIEW_REMAINING),
            "rejected": str(OUT_REJECTED),
            "report": str(OUT_REPORT),
        },
        "policy": {
            "accepted_clean_quarantine": "Accept high/medium sensitivity policy-routed memories as QUARANTINE.",
            "accepted_merge": "Accept only near-duplicate LongMemEval pairs with an existing target and enough shared tokens.",
            "accepted_update": "Accept only temporal/change-marker LongMemEval pairs with an existing target and update-like lexical evidence.",
            "ask_user": "Kept for later because it depends on the final consent policy.",
            "forget": "Kept for later because ambiguous uses of the word forget are unsafe as FORGET labels.",
        },
        "accepted": summarize(accepted),
        "review_remaining": summarize(review_remaining),
        "rejected": summarize(rejected),
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
