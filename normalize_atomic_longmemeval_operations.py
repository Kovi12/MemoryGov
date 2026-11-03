import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_space(text):
    text = str(text or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def compact_text(text):
    return re.sub(r"[^a-z0-9]+", "", normalize_space(text))


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_space(text))


STOPWORDS = {
    "i", "me", "my", "mine", "you", "your", "yours", "the", "a", "an",
    "and", "or", "to", "of", "in", "on", "for", "with", "that", "this",
    "it", "is", "was", "were", "am", "are", "be", "been", "being", "do",
    "did", "does", "have", "has", "had", "can", "could", "would", "should",
    "will", "just", "also", "really", "about", "from", "as", "at", "by",
    "if", "so", "but", "not", "then", "there", "here", "when", "where",
    "what", "which", "who", "how"
}


def content_tokens(text):
    toks = tokenize(text)
    filtered = [tok for tok in toks if tok not in STOPWORDS]
    return filtered if filtered else toks


def load_task_sessions(path):
    task_sessions = {}

    for task in read_jsonl(path):
        task_id = task.get("task_id")
        prompt = task.get("prompt", "")
        marker = "Conversation session:\n"

        if marker in prompt:
            session_text = prompt.split(marker, 1)[1]
        else:
            session_text = ""

        task_sessions[task_id] = {
            "text": session_text,
            "norm": normalize_space(session_text),
            "compact": compact_text(session_text),
            "tokens": tokenize(session_text),
            "token_set": set(tokenize(session_text)),
        }

    return task_sessions


def contains_any(text, terms):
    lowered = text.lower()
    return any(term in lowered for term in terms)


def infer_sensitivity(text, current):
    lowered = text.lower()

    high_terms = [
        "password", "api key", "access token", "secret key", "private key",
        "ssn", "social security number", "passport number", "bank account number",
        "credit card number"
    ]

    medium_terms = [
        "credit card", "mortgage", "insurance", "premium", "bank", "loan",
        "salary", "tax", "rent", "$", "medical", "doctor", "therapy",
        "diagnosis", "cancer", "injury", "knee", "prescription", "fitbit",
        "religion", "religious", "church", "bible", "prayer", "mass",
        "transgender", "lgbtq", "adoption", "accident", "workplace",
        "works in", "home", "address", "keycard", "security panel"
    ]

    if contains_any(lowered, high_terms):
        return "high"

    if contains_any(lowered, medium_terms):
        if current == "high":
            return current
        return "medium"

    return current


def infer_memory_type(text, current_type):
    lowered = text.lower()

    if current_type == "sensitive":
        return "semantic"

    if current_type == "procedural":
        if contains_any(lowered, [
            "prefers", "likes", "enjoys", "values", "is interested in",
            "is concerned about", "wants", "would like"
        ]):
            return "preference"

        if contains_any(lowered, [
            "plans to", "intends to", "is planning", "is considering",
            "is thinking of", "is preparing", "is working on", "is evaluating",
            "is researching"
        ]):
            return "project_state"

        if contains_any(lowered, [
            "uses", "tracks", "owns", "drives", "has", "attended",
            "received", "saved", "paid", "gets around", "works in"
        ]):
            return "semantic"

    return current_type


def evidence_is_artificial(evidence, current_query):
    ev = normalize_space(evidence)
    query = normalize_space(current_query)

    if not ev:
        return True

    artificial_phrases = [
        "none directly stated",
        "none provided",
        "inferred from context",
        "inferred from the context",
        "not explicitly mentioned",
        "benchmark question",
        "benchmark answer"
    ]

    if any(phrase in ev for phrase in artificial_phrases):
        return True

    if query and (ev == query or ev in query or query in ev):
        return True

    return False


def ordered_content_match(evidence_tokens, session_tokens):
    if len(evidence_tokens) < 4:
        return False, 0.0

    positions = []
    start_search = 0

    for tok in evidence_tokens:
        found = False
        for idx in range(start_search, len(session_tokens)):
            if session_tokens[idx] == tok:
                positions.append(idx)
                start_search = idx + 1
                found = True
                break
        if not found:
            return False, len(positions) / len(evidence_tokens)

    span = positions[-1] - positions[0] + 1
    max_span = max(40, len(evidence_tokens) * 6)

    if span <= max_span:
        return True, 1.0

    return False, 1.0


def token_overlap_match(evidence_tokens, session_token_set):
    if len(evidence_tokens) < 5:
        return False, 0.0

    overlap = sum(1 for tok in evidence_tokens if tok in session_token_set) / len(evidence_tokens)
    has_distinctive = any(len(tok) >= 6 for tok in evidence_tokens)

    if has_distinctive and overlap >= 0.9:
        return True, overlap

    return False, overlap


def evidence_in_session(evidence, session_info, current_query):
    if evidence_is_artificial(evidence, current_query):
        return False, "artificial_or_query_evidence", 0.0

    evidence_norm = normalize_space(evidence)

    if evidence_norm in session_info["norm"]:
        return True, "exact", 1.0

    evidence_compact = compact_text(evidence_norm)
    if len(evidence_compact) >= 30 and evidence_compact in session_info["compact"]:
        return True, "compact", 1.0

    if "..." in evidence_norm:
        parts = [part.strip() for part in evidence_norm.split("...") if len(part.strip()) >= 12]
        if parts:
            part_results = [evidence_in_session(part, session_info, current_query) for part in parts]
            if all(result[0] for result in part_results):
                avg_score = sum(result[2] for result in part_results) / len(part_results)
                return True, "ellipsis_all_parts", avg_score
            if any(result[0] for result in part_results):
                avg_score = sum(result[2] for result in part_results) / len(part_results)
                return True, "ellipsis_partial", avg_score

    ev_tokens = content_tokens(evidence_norm)

    ordered_ok, ordered_score = ordered_content_match(ev_tokens, session_info["tokens"])
    if ordered_ok:
        return True, "ordered_content_tokens", ordered_score

    overlap_ok, overlap_score = token_overlap_match(ev_tokens, session_info["token_set"])
    if overlap_ok:
        return True, "content_token_overlap", overlap_score

    return False, "not_found", max(ordered_score, overlap_score)


def is_answer_leakage_risk(memory_text, evidence_span, expected_answer, evidence_found):
    answer_norm = normalize_space(expected_answer)
    text_norm = normalize_space(memory_text)
    evidence_norm = normalize_space(evidence_span)

    if not answer_norm:
        return False

    if evidence_found:
        return False

    direct_answer_in_memory = answer_norm in text_norm
    evidence_is_answer = evidence_norm == answer_norm
    answer_in_evidence = answer_norm in evidence_norm

    return direct_answer_in_memory or evidence_is_answer or answer_in_evidence


def update_nested_fields(example, new_type, new_sensitivity):
    candidate = example.get("candidate_memory") or {}
    candidate["memory_type"] = new_type
    candidate["sensitivity"] = new_sensitivity
    example["candidate_memory"] = candidate

    preview = example.get("memory_record_preview") or {}
    preview["memory_type"] = new_type
    preview["sensitivity"] = new_sensitivity
    example["memory_record_preview"] = preview

    diagnostic = example.get("diagnostic_tags") or {}
    diagnostic["requires_sensitivity_reasoning"] = new_sensitivity != "low"
    example["diagnostic_tags"] = diagnostic

    return example


def main():
    atomic_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_qwen32b.jsonl"
    tasks_path = BASE_DIR / "processed/longmemeval_candidate_extraction_tasks.jsonl"

    normalized_all_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_qwen32b_normalized_all.jsonl"
    clean_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_qwen32b_clean.jsonl"
    review_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_qwen32b_review.jsonl"
    report_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_normalization_report.json"

    task_sessions = load_task_sessions(tasks_path)

    normalized_all = []
    clean = []
    review = []

    total = 0
    clean_count = 0
    review_count = 0

    before_type = Counter()
    after_type = Counter()
    before_sensitivity = Counter()
    after_sensitivity = Counter()
    decision_counter_all = Counter()
    decision_counter_clean = Counter()
    review_flag_counter = Counter()
    type_change_counter = Counter()
    sensitivity_change_counter = Counter()
    scenario_counter_clean = Counter()
    evidence_match_method_counter = Counter()

    review_examples_by_flag = defaultdict(list)

    for example in read_jsonl(atomic_path):
        total += 1

        if total % 500 == 0:
            print(f"processed={total}", file=sys.stderr, flush=True)

        candidate = example.get("candidate_memory") or {}
        gold = example.get("gold") or {}
        split = example.get("split_group") or {}

        task_id = split.get("task_id")
        session_info = task_sessions.get(task_id)

        if session_info is None:
            session_info = {
                "text": "",
                "norm": "",
                "compact": "",
                "tokens": [],
                "token_set": set(),
            }

        memory_text = candidate.get("text", "")
        evidence_span = candidate.get("evidence_span", "")
        expected_answer = example.get("expected_answer", "")
        current_query = example.get("current_query", "")
        reason = gold.get("reason", "")

        original_type = candidate.get("memory_type", "unknown")
        original_sensitivity = candidate.get("sensitivity", "unknown")

        before_type[original_type] += 1
        before_sensitivity[original_sensitivity] += 1

        new_type = infer_memory_type(memory_text, original_type)
        new_sensitivity = infer_sensitivity(memory_text + " " + evidence_span, original_sensitivity)

        flags = []

        if new_type != original_type:
            flags.append("memory_type_normalized")
            type_change_counter[f"{original_type}->{new_type}"] += 1

        if new_sensitivity != original_sensitivity:
            flags.append("sensitivity_normalized")
            sensitivity_change_counter[f"{original_sensitivity}->{new_sensitivity}"] += 1

        evidence_found, evidence_match_method, evidence_match_score = evidence_in_session(
            evidence_span,
            session_info,
            current_query,
        )

        evidence_match_method_counter[evidence_match_method] += 1

        if not evidence_span.strip():
            flags.append("missing_evidence")

        if evidence_span.strip() and not evidence_found:
            flags.append("evidence_not_found_in_session")

        if is_answer_leakage_risk(memory_text, evidence_span, expected_answer, evidence_found):
            flags.append("answer_leakage_risk")

        if "benchmark" in reason.lower() or "expected answer" in reason.lower() or "inferred from benchmark" in reason.lower():
            flags.append("benchmark_inference_reason")

        if len(memory_text.strip()) < 15:
            flags.append("very_short_memory_text")

        if len(memory_text.strip()) > 250:
            flags.append("possibly_not_atomic")

        if gold.get("primary_decision") == "WRITE" and candidate.get("stability") == "temporary":
            flags.append("temporary_write_review")

        if "user" not in memory_text.lower():
            flags.append("non_user_centered_or_world_memory")

        example = update_nested_fields(example, new_type, new_sensitivity)

        example["normalization_metadata"] = {
            "normalized": True,
            "original_memory_type": original_type,
            "normalized_memory_type": new_type,
            "original_sensitivity": original_sensitivity,
            "normalized_sensitivity": new_sensitivity,
            "evidence_found_in_session": evidence_found,
            "evidence_match_method": evidence_match_method,
            "evidence_match_score": evidence_match_score,
            "review_flags": flags,
        }

        normalized_all.append(example)

        decision = gold.get("primary_decision", "UNKNOWN")
        decision_counter_all[decision] += 1
        after_type[new_type] += 1
        after_sensitivity[new_sensitivity] += 1

        severe_flags = {
            "missing_evidence",
            "evidence_not_found_in_session",
            "answer_leakage_risk",
            "very_short_memory_text",
            "possibly_not_atomic",
        }

        should_review = any(flag in severe_flags for flag in flags)

        if should_review:
            review.append(example)
            review_count += 1
            for flag in flags:
                review_flag_counter[flag] += 1
                if len(review_examples_by_flag[flag]) < 10:
                    review_examples_by_flag[flag].append({
                        "example_id": example.get("example_id"),
                        "decision": decision,
                        "memory_text": memory_text,
                        "evidence_span": evidence_span,
                        "expected_answer": expected_answer,
                        "reason": reason,
                        "evidence_match_method": evidence_match_method,
                        "evidence_match_score": evidence_match_score,
                        "flags": flags,
                    })
        else:
            clean.append(example)
            clean_count += 1
            decision_counter_clean[decision] += 1
            scenario_counter_clean[example.get("scenario_type", "UNKNOWN")] += 1

    write_jsonl(normalized_all_path, normalized_all)
    write_jsonl(clean_path, clean)
    write_jsonl(review_path, review)

    report = {
        "input": str(atomic_path),
        "tasks_input": str(tasks_path),
        "normalized_all_output": str(normalized_all_path),
        "clean_output": str(clean_path),
        "review_output": str(review_path),
        "total_examples": total,
        "clean_examples": clean_count,
        "review_examples": review_count,
        "before_memory_type_counter": dict(before_type),
        "after_memory_type_counter": dict(after_type),
        "before_sensitivity_counter": dict(before_sensitivity),
        "after_sensitivity_counter": dict(after_sensitivity),
        "decision_counter_all": dict(decision_counter_all),
        "decision_counter_clean": dict(decision_counter_clean),
        "scenario_counter_clean": dict(scenario_counter_clean),
        "evidence_match_method_counter": dict(evidence_match_method_counter),
        "review_flag_counter": dict(review_flag_counter),
        "type_change_counter": dict(type_change_counter),
        "sensitivity_change_counter": dict(sensitivity_change_counter),
        "review_examples_by_flag": dict(review_examples_by_flag),
    }

    write_json(report_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved normalized all examples to: {normalized_all_path}")
    print(f"Saved clean examples to: {clean_path}")
    print(f"Saved review examples to: {review_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
