import argparse
import json
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_OUT_DIR = BASE_DIR / "processed" / "operation_expansion_existing"

LONGMEMEVAL_REVIEW = BASE_DIR / "processed" / "longmemeval_atomic_operation_candidates_qwen32b_review.jsonl"
LONGMEMEVAL_NORMALIZED = BASE_DIR / "processed" / "longmemeval_atomic_operation_candidates_qwen32b_normalized_all.jsonl"
LONGMEMEVAL_CLEAN = BASE_DIR / "processed" / "longmemeval_atomic_operation_candidates_qwen32b_clean.jsonl"
LOCOMO_OPERATION = BASE_DIR / "processed" / "locomo_operation_write.jsonl"
MEMORYAGENTBENCH_RAW_DIR = BASE_DIR / "raw" / "memoryagentbench"

CURRENT_OPERATION_FILES = [
    BASE_DIR / "processed" / "memorygovbench_current" / "operation_train.jsonl",
    BASE_DIR / "processed" / "memorygovbench_current" / "operation_val.jsonl",
    BASE_DIR / "processed" / "memorygovbench_current" / "operation_test.jsonl",
]

HARD_IGNORE_FLAGS = {
    "answer_leakage_risk",
    "missing_evidence",
    "benchmark_inference_reason",
    "non_user_centered_or_world_memory",
    "artificial_or_query_evidence",
}

SOFT_IGNORE_FLAGS = {
    "evidence_not_found_in_session",
}

UPDATE_MARKERS = [
    "now",
    "currently",
    "no longer",
    "not anymore",
    "changed",
    "updated",
    "replaced",
    "switched",
    "moved",
    "renamed",
    "started using",
    "stopped using",
    "instead",
    "new",
    "latest",
    "recently",
    "from now on",
    "these days",
    "used to",
]

FORGET_MARKERS = [
    "forget",
    "delete",
    "remove from memory",
    "remove this memory",
    "do not remember",
    "don't remember",
    "dont remember",
    "do not store",
    "don't store",
    "dont store",
    "do not save",
    "don't save",
    "dont save",
    "erase",
]

SENSITIVE_HIGH_PATTERNS = [
    r"\blgbtq\b",
    r"\btransgender\b",
    r"\bgay\b",
    r"\blesbian\b",
    r"\bbisexual\b",
    r"\bsexual orientation\b",
    r"\breligion\b",
    r"\bmuslim\b",
    r"\bchristian\b",
    r"\bcatholic\b",
    r"\bjewish\b",
    r"\btherapy\b",
    r"\btherapist\b",
    r"\bdoctor\b",
    r"\bdiagnosed\b",
    r"\bdiagnosis\b",
    r"\bdepression\b",
    r"\banxiety\b",
    r"\bmedical\b",
    r"\bmedication\b",
    r"\bpassword\b",
    r"\btoken\b",
    r"\bapi key\b",
    r"\bsecret\b",
    r"\bssn\b",
    r"\bsocial security\b",
    r"\bpassport\b",
    r"\bcredit card\b",
    r"\bbank account\b",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "user",
    "their", "his", "her", "my", "i", "me", "we", "our", "they", "them", "it",
    "this", "that", "these", "those", "as", "about", "into", "after", "before",
}


def read_jsonl(path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                yield {
                    "_parse_error": True,
                    "_path": str(path),
                    "_line_no": line_no,
                    "_raw": line[:1000],
                }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stable_hash(obj, prefix):
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9$@._:/\- ]+", "", text)
    return text.strip()


def compact_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_tokens(text):
    toks = re.findall(r"[a-z0-9$@._:/\-]+", normalize_text(text))
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def jaccard(a, b):
    sa = set(word_tokens(a))
    sb = set(word_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def seq_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def text_similarity(a, b):
    return max(seq_similarity(a, b), jaccard(a, b))


def has_any_marker(text, markers):
    low = normalize_text(text)
    return any(marker in low for marker in markers)


def is_high_sensitive_text(text):
    low = normalize_text(text)
    return any(re.search(pattern, low) for pattern in SENSITIVE_HIGH_PATTERNS)


def get_nested(row, *keys, default=None):
    cur = row
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def get_candidate_text(row):
    return compact_text(
        get_nested(row, "candidate_memory", "text")
        or row.get("memory_text")
        or row.get("candidate_memory_text")
        or row.get("text")
        or ""
    )


def get_memory_type(row):
    return (
        get_nested(row, "candidate_memory", "memory_type")
        or row.get("memory_type")
        or "semantic"
    )


def get_sensitivity(row):
    return (
        get_nested(row, "candidate_memory", "sensitivity")
        or row.get("sensitivity")
        or "low"
    )


def get_stability(row):
    return (
        get_nested(row, "candidate_memory", "stability")
        or row.get("stability")
        or "unknown"
    )


def get_source_dataset(row):
    return row.get("source_dataset") or get_nested(row, "annotation_metadata", "source_dataset") or "unknown"


def get_source_file(row):
    return row.get("source_file") or get_nested(row, "annotation_metadata", "source_file") or "unknown"


def get_query(row):
    return row.get("current_query") or row.get("query") or ""


def get_expected_answer(row):
    return row.get("expected_answer") or row.get("expected_answers")


def get_flags(row):
    flags = row.get("flags") or row.get("review_flags") or row.get("suspicious_flags")
    if not flags:
        flags = get_nested(row, "annotation_metadata", "flags", default=[])
    if isinstance(flags, str):
        flags = [flags]
    return list(flags or [])


def get_reason(row):
    return row.get("reason") or get_nested(row, "gold", "reason") or ""


def get_group_key(row):
    split_group = row.get("split_group")
    if split_group:
        return json.dumps(split_group, sort_keys=True, ensure_ascii=False)

    fields = {
        "source_dataset": get_source_dataset(row),
        "source_file": get_source_file(row),
        "question_id": row.get("question_id"),
        "source_example_id": row.get("source_example_id"),
        "qa_pair_id": row.get("qa_pair_id"),
        "query": get_query(row),
    }
    return json.dumps(fields, sort_keys=True, ensure_ascii=False)


def get_label(row):
    return get_nested(row, "gold", "primary_decision") or row.get("decision")


def make_candidate_memory(row, override_text=None, override_sensitivity=None, override_memory_type=None):
    text = compact_text(override_text if override_text is not None else get_candidate_text(row))
    memory_type = override_memory_type or get_memory_type(row)
    sensitivity = override_sensitivity or get_sensitivity(row)

    if sensitivity == "low" and is_high_sensitive_text(text):
        sensitivity = "high"

    return {
        "text": text,
        "memory_type": memory_type,
        "source_type": get_nested(row, "candidate_memory", "source_type", default=row.get("source_type") or "benchmark_derived"),
        "source_trust": get_nested(row, "candidate_memory", "source_trust", default="benchmark"),
        "sensitivity": sensitivity,
        "confidence": get_nested(row, "candidate_memory", "confidence", default=row.get("confidence") or 1.0),
        "valid_from": get_nested(row, "candidate_memory", "valid_from", default=row.get("candidate_valid_from")),
        "valid_to": get_nested(row, "candidate_memory", "valid_to", default=None),
    }


def make_existing_memory(text, row, idx=0):
    memory_id = stable_hash(
        {
            "source": get_source_dataset(row),
            "source_file": get_source_file(row),
            "text": text,
            "idx": idx,
        },
        "existing_mem",
    )
    return {
        "memory_id": memory_id,
        "text": compact_text(text),
        "memory_type": get_memory_type(row),
        "sensitivity": get_sensitivity(row),
        "status": "active",
        "source_type": "benchmark_derived",
    }


def make_operation_example(row, label, reason, derivation, confidence, existing_memories=None, review_flags=None):
    candidate = make_candidate_memory(row)
    text = candidate["text"]

    if not text:
        return None

    existing_memories = existing_memories or []
    review_flags = review_flags or []

    target_ids = [m["memory_id"] for m in existing_memories if m.get("memory_id")]

    status_map = {
        "UPDATE": "superseded",
        "MERGE": "merged",
        "FORGET": "forgotten",
        "QUARANTINE": "quarantined",
        "ASK_USER": "pending_user_confirmation",
        "IGNORE": None,
        "WRITE": None,
    }

    payload_for_id = {
        "label": label,
        "text": text,
        "existing": [m.get("text") for m in existing_memories],
        "source_dataset": get_source_dataset(row),
        "source_file": get_source_file(row),
        "group": get_group_key(row),
        "derivation": derivation,
    }

    example_id = stable_hash(payload_for_id, "op_expand")

    return {
        "example_id": example_id,
        "source_dataset": get_source_dataset(row),
        "source_file": get_source_file(row),
        "split_group": {
            "source_dataset": get_source_dataset(row),
            "source_file": get_source_file(row),
            "group_key": get_group_key(row),
            "derivation": derivation,
        },
        "task_level": "operation_governance",
        "scenario_type": row.get("scenario_type") or f"existing_mined_{derivation}",
        "failure_type": None,
        "history_context": row.get("history_context") or [],
        "candidate_memory": candidate,
        "existing_memories": existing_memories,
        "current_query": "What operation should be applied to this candidate memory?",
        "expected_answer": None,
        "gold": {
            "primary_decision": label,
            "secondary_decision": None,
            "target_memory_ids": target_ids,
            "new_status_for_targets": status_map.get(label),
            "should_retrieve_candidate": None,
            "should_retrieve_existing": None,
            "reason": reason,
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": label in {"UPDATE", "FORGET"},
            "requires_contradiction_resolution": label in {"UPDATE", "ASK_USER"},
            "requires_sensitivity_reasoning": label in {"QUARANTINE", "ASK_USER"},
            "requires_appropriateness_reasoning": label in {"IGNORE", "ASK_USER", "QUARANTINE"},
            "requires_provenance_reasoning": label in {"IGNORE", "ASK_USER", "QUARANTINE"},
        },
        "annotation_metadata": {
            "label_source": "heuristic_existing_benchmark_mined",
            "human_audited": False,
            "confidence": confidence,
            "derivation": derivation,
            "review_flags": review_flags,
            "original_example_id": row.get("example_id") or row.get("task_id"),
            "original_label": get_label(row),
            "original_reason": get_reason(row),
            "original_expected_answer": get_expected_answer(row),
        },
    }


def add_example(rows, seen, row, label, reason, derivation, confidence, existing_memories=None, review_flags=None):
    ex = make_operation_example(
        row=row,
        label=label,
        reason=reason,
        derivation=derivation,
        confidence=confidence,
        existing_memories=existing_memories,
        review_flags=review_flags,
    )
    if not ex:
        return False

    dedup_key = stable_hash(
        {
            "label": label,
            "candidate": ex["candidate_memory"]["text"],
            "existing": [m["text"] for m in ex["existing_memories"]],
        },
        "dedup",
    )

    if dedup_key in seen:
        return False

    seen.add(dedup_key)
    rows.append(ex)
    return True


def build_existing_label_index():
    index = defaultdict(Counter)

    for path in CURRENT_OPERATION_FILES:
        for row in read_jsonl(path):
            text = get_candidate_text(row)
            label = get_label(row)
            if text and label:
                index[normalize_text(text)][label] += 1

    return index


def mine_ignore_from_longmemeval_review(rows, seen, max_per_label):
    count = 0

    for row in read_jsonl(LONGMEMEVAL_REVIEW):
        if count >= max_per_label:
            break

        flags = set(get_flags(row))
        if not flags:
            continue

        hard = flags & HARD_IGNORE_FLAGS
        soft = flags & SOFT_IGNORE_FLAGS

        if not hard and not soft:
            continue

        text = get_candidate_text(row)
        if not text:
            continue

        if hard:
            confidence = 0.9
            selected_flags = sorted(hard)
            reason = "Candidate should not be written because it is unsupported, answer-leaking, benchmark-inferred, or not user-centered."
        else:
            confidence = 0.72
            selected_flags = sorted(soft)
            reason = "Candidate has weak or missing evidence in the source session and should be treated as an operation-level ignore candidate."

        if add_example(
            rows,
            seen,
            row,
            "IGNORE",
            reason,
            "longmemeval_review_flags",
            confidence,
            review_flags=selected_flags,
        ):
            count += 1

    return count


def mine_quarantine_from_locomo(rows, seen, max_per_label):
    count = 0

    for row in read_jsonl(LOCOMO_OPERATION):
        if count >= max_per_label:
            break

        text = get_candidate_text(row)
        memory_type = get_memory_type(row)
        sensitivity = get_sensitivity(row)

        should_quarantine = (
            sensitivity == "high"
            or memory_type == "sensitive"
            or is_high_sensitive_text(text)
        )

        if not should_quarantine:
            continue

        reason = "Candidate contains high-sensitivity personal information and should be routed to protected memory rather than ordinary long-term memory."

        if add_example(
            rows,
            seen,
            row,
            "QUARANTINE",
            reason,
            "locomo_sensitive_memory_policy_relabel",
            0.86,
            review_flags=["policy_relabel_from_write_to_quarantine"],
        ):
            count += 1

    return count


def mine_ask_user_from_locomo_sensitive_medium(rows, seen, max_per_label):
    count = 0

    for row in read_jsonl(LOCOMO_OPERATION):
        if count >= max_per_label:
            break

        text = get_candidate_text(row)
        memory_type = get_memory_type(row)
        sensitivity = get_sensitivity(row)
        source_type = get_nested(row, "candidate_memory", "source_type", default=row.get("source_type") or "")

        if sensitivity != "medium":
            continue

        if memory_type not in {"sensitive", "semantic", "episodic", "preference"}:
            continue

        if source_type not in {"observation", "event", "benchmark_derived", ""}:
            continue

        if is_high_sensitive_text(text):
            continue

        reason = "Candidate is potentially personal or sensitive enough that the memory system should ask the user before storing it."

        if add_example(
            rows,
            seen,
            row,
            "ASK_USER",
            reason,
            "locomo_medium_sensitivity_confirmation_policy",
            0.74,
            review_flags=["policy_requires_user_confirmation"],
        ):
            count += 1

    return count


def mine_forget_requests(rows, seen, max_per_label):
    count = 0

    source_paths = [
        LONGMEMEVAL_NORMALIZED,
        LONGMEMEVAL_CLEAN,
        LONGMEMEVAL_REVIEW,
        LOCOMO_OPERATION,
    ]

    for path in source_paths:
        for row in read_jsonl(path):
            if count >= max_per_label:
                return count

            text_blob = " ".join([
                get_candidate_text(row),
                get_query(row),
                get_reason(row),
                json.dumps(row.get("history_context") or [], ensure_ascii=False),
            ])

            if not has_any_marker(text_blob, FORGET_MARKERS):
                continue

            candidate_text = get_candidate_text(row)
            if not candidate_text:
                continue

            existing = [make_existing_memory(candidate_text, row, idx=0)]

            reason = "Source text contains an explicit or near-explicit deletion, forgetting, or do-not-store request."

            if add_example(
                rows,
                seen,
                row,
                "FORGET",
                reason,
                "explicit_forget_or_deletion_marker",
                0.82,
                existing_memories=existing,
                review_flags=["forget_marker_detected"],
            ):
                count += 1

    return count


def group_rows_for_pair_mining(path):
    groups = defaultdict(list)

    for row in read_jsonl(path):
        text = get_candidate_text(row)
        if not text:
            continue

        label = get_label(row)
        if label and label not in {"WRITE", "IGNORE"}:
            continue

        groups[get_group_key(row)].append(row)

    return groups


def mine_merge_update_from_longmemeval(rows, seen, max_per_label, merge_similarity, update_min_similarity, update_max_similarity):
    merge_count = 0
    update_count = 0
    ask_count = 0

    groups = group_rows_for_pair_mining(LONGMEMEVAL_NORMALIZED)

    for _, group in groups.items():
        if len(group) < 2:
            continue

        group = group[:60]
        local_merge = 0
        local_update = 0
        local_ask = 0

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]

                text_a = get_candidate_text(a)
                text_b = get_candidate_text(b)
                if not text_a or not text_b:
                    continue

                sim = text_similarity(text_a, text_b)
                shared = len(set(word_tokens(text_a)) & set(word_tokens(text_b)))

                if merge_count < max_per_label and local_merge < 3:
                    if merge_similarity <= sim < 0.995:
                        existing = [make_existing_memory(text_a, a, idx=i)]
                        reason = f"Candidate is a near-duplicate or complementary paraphrase of an existing memory. Similarity={sim:.3f}."

                        if add_example(
                            rows,
                            seen,
                            b,
                            "MERGE",
                            reason,
                            "longmemeval_near_duplicate_memory_pair",
                            0.84,
                            existing_memories=existing,
                            review_flags=["near_duplicate_pair"],
                        ):
                            merge_count += 1
                            local_merge += 1

                scenario = str(a.get("scenario_type") or b.get("scenario_type") or "")
                pair_blob = " ".join([text_a, text_b, get_reason(a), get_reason(b), scenario])
                has_update = has_any_marker(pair_blob, UPDATE_MARKERS)
                same_topic = shared >= 2

                if update_count < max_per_label and local_update < 4:
                    if same_topic and has_update and update_min_similarity <= sim <= update_max_similarity:
                        existing = [make_existing_memory(text_a, a, idx=i)]
                        reason = f"Candidate appears to revise or supersede a related existing memory. Similarity={sim:.3f}, shared_content_tokens={shared}."

                        if add_example(
                            rows,
                            seen,
                            b,
                            "UPDATE",
                            reason,
                            "longmemeval_temporal_or_change_marker_pair",
                            0.76,
                            existing_memories=existing,
                            review_flags=["change_marker_pair"],
                        ):
                            update_count += 1
                            local_update += 1

                if ask_count < max_per_label and local_ask < 2:
                    contradiction_like = (
                        (" no " in f" {normalize_text(text_a)} " and " no " not in f" {normalize_text(text_b)} ")
                        or (" no " in f" {normalize_text(text_b)} " and " no " not in f" {normalize_text(text_a)} ")
                        or ("not" in word_tokens(text_a) and "not" not in word_tokens(text_b))
                        or ("not" in word_tokens(text_b) and "not" not in word_tokens(text_a))
                        or ("changed" in normalize_text(pair_blob))
                        or ("instead" in normalize_text(pair_blob))
                        or ("no longer" in normalize_text(pair_blob))
                    )

                    if same_topic and contradiction_like and 0.25 <= sim <= 0.82:
                        existing = [make_existing_memory(text_a, a, idx=i)]
                        reason = f"Candidate may conflict with or change an existing memory, but the correct operation is ambiguous without confirmation. Similarity={sim:.3f}."

                        if add_example(
                            rows,
                            seen,
                            b,
                            "ASK_USER",
                            reason,
                            "longmemeval_ambiguous_conflict_pair",
                            0.68,
                            existing_memories=existing,
                            review_flags=["ambiguous_conflict_pair"],
                        ):
                            ask_count += 1
                            local_ask += 1

                if (
                    merge_count >= max_per_label
                    and update_count >= max_per_label
                    and ask_count >= max_per_label
                ):
                    return merge_count, update_count, ask_count

    return merge_count, update_count, ask_count


def profile_memoryagentbench_raw(out_dir):
    report = {}

    if not MEMORYAGENTBENCH_RAW_DIR.exists():
        return report

    for path in sorted(MEMORYAGENTBENCH_RAW_DIR.glob("*.jsonl")):
        key_counter = Counter()
        row_count = 0
        samples = []

        for row in read_jsonl(path):
            row_count += 1
            if isinstance(row, dict):
                key_counter.update(row.keys())
            if len(samples) < 3:
                samples.append(row)

        report[path.name] = {
            "path": str(path),
            "rows": row_count,
            "top_level_keys": dict(key_counter),
            "samples": samples,
        }

    write_json(out_dir / "memoryagentbench_raw_schema_report.json", report)
    return report


def attach_current_label_conflicts(rows, existing_index):
    conflict_counter = Counter()

    for row in rows:
        text = normalize_text(get_candidate_text(row))
        existing = existing_index.get(text)
        if not existing:
            continue

        new_label = get_label(row)
        existing_labels = sorted(existing.keys())

        row["annotation_metadata"]["existing_current_operation_labels"] = existing_labels

        if new_label not in existing:
            row["annotation_metadata"]["conflicts_with_current_operation_dataset"] = True
            conflict_counter[f"{','.join(existing_labels)}->{new_label}"] += 1
        else:
            row["annotation_metadata"]["conflicts_with_current_operation_dataset"] = False

    return conflict_counter


def split_clean_review(rows):
    clean = []
    review = []

    for row in rows:
        label = get_label(row)
        confidence = float(get_nested(row, "annotation_metadata", "confidence", default=0.0))
        conflict = bool(get_nested(row, "annotation_metadata", "conflicts_with_current_operation_dataset", default=False))
        flags = set(get_nested(row, "annotation_metadata", "review_flags", default=[]))

        clean_ok = (
            confidence >= 0.82
            and label in {"IGNORE", "MERGE", "QUARANTINE", "FORGET"}
            and not ("ambiguous_conflict_pair" in flags)
        )

        if conflict and label not in {"QUARANTINE"}:
            clean_ok = False

        if clean_ok:
            clean.append(row)
        else:
            review.append(row)

    return clean, review


def build_report(rows, clean, review, conflict_counter, mab_schema):
    def summarize(subset):
        return {
            "total": len(subset),
            "label_counter": dict(Counter(get_label(r) for r in subset)),
            "source_dataset_counter": dict(Counter(get_source_dataset(r) for r in subset)),
            "source_file_counter": dict(Counter(get_source_file(r) for r in subset)),
            "scenario_counter": dict(Counter(r.get("scenario_type") for r in subset)),
            "derivation_counter": dict(Counter(get_nested(r, "annotation_metadata", "derivation") for r in subset)),
            "memory_type_counter": dict(Counter(get_memory_type(r) for r in subset)),
            "sensitivity_counter": dict(Counter(get_sensitivity(r) for r in subset)),
        }

    return {
        "version": "v3.48",
        "task": "operation_expansion_from_existing_benchmarks",
        "purpose": "Mine additional operation-governance labels from existing benchmark-derived data before synthetic generation.",
        "important_note": "These outputs are candidates. Do not merge blindly into the training dataset before reviewing label policy conflicts.",
        "outputs": {
            "all": str(DEFAULT_OUT_DIR / "operation_existing_expansion_all.jsonl"),
            "clean": str(DEFAULT_OUT_DIR / "operation_existing_expansion_clean.jsonl"),
            "review": str(DEFAULT_OUT_DIR / "operation_existing_expansion_review.jsonl"),
            "memoryagentbench_schema": str(DEFAULT_OUT_DIR / "memoryagentbench_raw_schema_report.json"),
        },
        "all": summarize(rows),
        "clean": summarize(clean),
        "review": summarize(review),
        "conflict_with_current_operation_dataset_counter": dict(conflict_counter),
        "memoryagentbench_raw_files_profiled": {
            name: {
                "rows": data.get("rows"),
                "top_level_keys": data.get("top_level_keys"),
            }
            for name, data in mab_schema.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max_per_label", type=int, default=2000)
    parser.add_argument("--merge_similarity", type=float, default=0.82)
    parser.add_argument("--update_min_similarity", type=float, default=0.30)
    parser.add_argument("--update_max_similarity", type=float, default=0.82)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()

    mined = {}

    mined["ignore_from_longmemeval_review"] = mine_ignore_from_longmemeval_review(
        rows, seen, args.max_per_label
    )

    mined["quarantine_from_locomo"] = mine_quarantine_from_locomo(
        rows, seen, args.max_per_label
    )

    mined["ask_user_from_locomo_medium_sensitive"] = mine_ask_user_from_locomo_sensitive_medium(
        rows, seen, args.max_per_label
    )

    mined["forget_requests"] = mine_forget_requests(
        rows, seen, args.max_per_label
    )

    merge_count, update_count, ask_count = mine_merge_update_from_longmemeval(
        rows,
        seen,
        args.max_per_label,
        args.merge_similarity,
        args.update_min_similarity,
        args.update_max_similarity,
    )

    mined["merge_from_longmemeval_pairs"] = merge_count
    mined["update_from_longmemeval_pairs"] = update_count
    mined["ask_user_from_longmemeval_pairs"] = ask_count

    mab_schema = profile_memoryagentbench_raw(out_dir)

    existing_index = build_existing_label_index()
    conflict_counter = attach_current_label_conflicts(rows, existing_index)

    clean, review = split_clean_review(rows)

    all_path = out_dir / "operation_existing_expansion_all.jsonl"
    clean_path = out_dir / "operation_existing_expansion_clean.jsonl"
    review_path = out_dir / "operation_existing_expansion_review.jsonl"
    report_path = out_dir / "operation_existing_expansion_report.json"

    write_jsonl(all_path, rows)
    write_jsonl(clean_path, clean)
    write_jsonl(review_path, review)

    report = build_report(rows, clean, review, conflict_counter, mab_schema)
    report["mining_counts"] = mined
    report["parameters"] = {
        "max_per_label": args.max_per_label,
        "merge_similarity": args.merge_similarity,
        "update_min_similarity": args.update_min_similarity,
        "update_max_similarity": args.update_max_similarity,
    }

    write_json(report_path, report)

    print(json.dumps({
        "version": report["version"],
        "all": len(rows),
        "clean": len(clean),
        "review": len(review),
        "label_counter_all": report["all"]["label_counter"],
        "label_counter_clean": report["clean"]["label_counter"],
        "label_counter_review": report["review"]["label_counter"],
        "report": str(report_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
