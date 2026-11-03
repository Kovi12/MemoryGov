import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


VERSION = "v3.49"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "memoryagentbench"
OUT_DIR = BASE_DIR / "processed" / "operation_expansion_memoryagentbench_v2"

DEFAULT_FILES = [
    "Accurate_Retrieval.jsonl",
    "Long_Range_Understanding.jsonl",
    "Test_Time_Learning.jsonl",
]

OPTIONAL_CONFLICT_FILE = "Conflict_Resolution.jsonl"

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "with", "and", "or", "is", "are", "was", "were",
    "be", "been", "being", "has", "have", "had", "by", "from", "as", "at", "that", "this", "it", "its",
    "his", "her", "their", "they", "he", "she", "who", "what", "which", "where", "when", "how", "why",
}

RELATION_MARKERS = [
    " is ", " was ", " are ", " were ", " has ", " have ", " had ", " plays ", " played ",
    " works ", " worked ", " lives ", " lived ", " resides ", " resided ", " located ",
    " born ", " died ", " married ", " spouse ", " citizenship ", " country ", " capital ",
    " associated ", " affiliated ", " created ", " founded ", " directed ", " managed ",
]

STRONG_UPDATE_RE = re.compile(
    r"\b(no longer|now|currently|changed|updated|replaced|renamed|moved from|moved to|switched from|switched to|instead of|corrected|superseded)\b",
    re.IGNORECASE,
)

WEAK_UPDATE_RE = re.compile(
    r"\b(previously|formerly|used to|used-to|before|after|later|recently|new|latest)\b",
    re.IGNORECASE,
)

STRONG_FORGET_RE = re.compile(
    r"("
    r"\b(forget|delete|remove|erase|purge)\b.{0,80}\b(memory|remembered|stored|record|profile|personal data|data|this information)\b"
    r"|"
    r"\b(stop|do not|don't)\b.{0,40}\b(remembering|storing|retaining|keeping)\b"
    r"|"
    r"\b(user|person|speaker)\b.{0,80}\b(asked|requested|told)\b.{0,80}\b(forget|delete|remove|erase|purge)\b"
    r")",
    re.IGNORECASE,
)

AMBIGUOUS_FORGET_RE = re.compile(r"\bforget|forgot|forgotten\b", re.IGNORECASE)

BAD_FORGET_RE = re.compile(
    r"\b(never forget|don't forget to|do not forget to|not forget to|forgot to|forgetting to)\b",
    re.IGNORECASE,
)

SENSITIVE_HIGH_RE = re.compile(
    r"\b("
    r"lgbtq|transgender|sexual orientation|gay|lesbian|bisexual|hiv|aids|cancer|depression|anxiety|therapy|therapist|"
    r"mental health|diagnosed|diagnosis|pregnant|pregnancy|abortion|religion|religious|political belief|salary|mortgage|"
    r"bank account|credit card|ssn|social security|passport|medical|doctor|medication|addiction"
    r")\b",
    re.IGNORECASE,
)

SENSITIVE_MEDIUM_RE = re.compile(
    r"\b("
    r"address|phone number|email|workplace|location|school|income|insurance|family|relationship|divorce|citizenship|"
    r"nationality|ethnicity|race"
    r")\b",
    re.IGNORECASE,
)


def stable_hash(*parts, length=16):
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as exc:
                yield line_no, {"__parse_error__": str(exc), "__raw_line__": line}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_text(text):
    text = normalize_ws(text).lower()
    text = re.sub(r"^\s*(fact\s*)?\d+\s*[:.)-]\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_tokens(text):
    toks = re.findall(r"[a-z0-9]+", normalize_text(text))
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def token_jaccard(a, b):
    sa = set(content_tokens(a))
    sb = set(content_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_ratio(a, b):
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def compact_key(text):
    return normalize_text(text)


def subject_key(text):
    n = " " + normalize_text(text) + " "
    split_pos = None
    for marker in RELATION_MARKERS:
        pos = n.find(marker)
        if pos > 0:
            split_pos = pos
            break
    if split_pos is None:
        toks = content_tokens(text)[:6]
    else:
        prefix = n[:split_pos]
        toks = content_tokens(prefix)[:8]
    if len(toks) < 2:
        toks = content_tokens(text)[:5]
    return " ".join(toks)


def classify_sensitivity(text):
    if SENSITIVE_HIGH_RE.search(text or ""):
        return "high"
    if SENSITIVE_MEDIUM_RE.search(text or ""):
        return "medium"
    return "low"


def guess_memory_type(text, label=None):
    sensitivity = classify_sensitivity(text)
    if sensitivity == "high":
        return "sensitive"
    low = normalize_text(text)
    if label in {"UPDATE", "MERGE", "FORGET"}:
        return "semantic"
    if any(x in low for x in ["prefers", "likes", "enjoys", "wants", "interested in", "favorite"]):
        return "preference"
    if any(x in low for x in ["plans to", "working on", "trying to", "preparing", "project"]):
        return "project_state"
    return "semantic"


def is_strong_forget_request(text):
    if not text or BAD_FORGET_RE.search(text):
        return False
    return bool(STRONG_FORGET_RE.search(text))


def is_ambiguous_forget(text):
    if not text or BAD_FORGET_RE.search(text):
        return False
    return bool(AMBIGUOUS_FORGET_RE.search(text))


def has_strong_update_marker(text):
    return bool(STRONG_UPDATE_RE.search(text or ""))


def has_weak_update_marker(text):
    return bool(WEAK_UPDATE_RE.search(text or ""))


def maybe_split_fact_string(text):
    text = normalize_ws(text)
    if not text:
        return []
    lines = [normalize_ws(x) for x in text.splitlines() if normalize_ws(x)]
    if len(lines) >= 2:
        fact_like = 0
        for line in lines:
            if re.match(r"^\s*(fact\s*)?\d+\s*[:.)-]\s*", line, flags=re.IGNORECASE):
                fact_like += 1
        if fact_like >= max(2, int(0.4 * len(lines))):
            return lines
    return [text]


def extract_context_items(obj, source_path="context", parent_id=None):
    items = []

    def rec(value, path, inherited_id):
        if value is None:
            return

        if isinstance(value, str):
            for idx, piece in enumerate(maybe_split_fact_string(value)):
                if len(piece) >= 8:
                    fact_id = inherited_id if inherited_id is not None else f"{path}:{idx}"
                    items.append({
                        "fact_id": str(fact_id),
                        "text": piece,
                        "source_path": path,
                    })
            return

        if isinstance(value, list):
            for i, item in enumerate(value):
                rec(item, f"{path}[{i}]", inherited_id=None)
            return

        if isinstance(value, dict):
            explicit_id = inherited_id
            for id_key in ["fact_id", "id", "memory_id", "idx", "index", "fact_idx"]:
                if id_key in value and value[id_key] is not None:
                    explicit_id = str(value[id_key])
                    break

            for text_key in ["text", "fact", "content", "statement", "memory", "sentence", "value"]:
                if text_key in value and isinstance(value[text_key], str):
                    rec(value[text_key], f"{path}.{text_key}", explicit_id)
                    return

            for k, v in value.items():
                if k in {"metadata", "answers", "questions"}:
                    continue
                rec(v, f"{path}.{k}", explicit_id)
            return

    rec(obj, source_path, parent_id)
    deduped = []
    seen = set()
    for item in items:
        key = compact_key(item["text"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def extract_qas(row):
    questions = row.get("questions", [])
    answers = row.get("answers", [])

    if isinstance(questions, str):
        questions = [questions]
    if isinstance(answers, str):
        answers = [answers]

    if not isinstance(questions, list):
        questions = []
    if not isinstance(answers, list):
        answers = []

    qas = []
    max_len = max(len(questions), len(answers))
    for i in range(max_len):
        q = questions[i] if i < len(questions) else None
        a = answers[i] if i < len(answers) else None

        if isinstance(q, dict):
            q_text = q.get("question") or q.get("query") or q.get("text") or json.dumps(q, ensure_ascii=False)
        else:
            q_text = q

        if isinstance(a, dict):
            a_text = a.get("answer") or a.get("text") or json.dumps(a, ensure_ascii=False)
        elif isinstance(a, list):
            a_text = " | ".join(str(x) for x in a)
        else:
            a_text = a

        if q_text or a_text:
            qas.append({
                "qa_idx": i,
                "question": normalize_ws(q_text),
                "answer": normalize_ws(a_text),
            })

    return qas


def memory_obj(text, memory_id=None, label=None, source_type="benchmark_fact"):
    return {
        "memory_id": memory_id,
        "text": normalize_ws(text),
        "memory_type": guess_memory_type(text, label=label),
        "source_type": source_type,
        "source_trust": "benchmark",
        "sensitivity": classify_sensitivity(text),
        "confidence": 1.0,
        "valid_from": None,
        "valid_to": None,
    }


def make_example(
    label,
    source_file,
    row_idx,
    line_no,
    candidate_text,
    existing_texts,
    reason,
    derivation,
    confidence,
    scenario_type,
    source_subset=None,
    fact_ids=None,
    qa=None,
    flags=None,
    status="review",
):
    flags = flags or []
    fact_ids = fact_ids or []
    qa = qa or {}

    example_id = "mab_op_expand_" + stable_hash(
        source_file,
        row_idx,
        line_no,
        label,
        candidate_text,
        "|".join(existing_texts),
        derivation,
    )

    existing = []
    for i, text in enumerate(existing_texts):
        mid = "existing_" + stable_hash(source_file, row_idx, text, i, length=10)
        existing.append(memory_obj(text, memory_id=mid, label=label))

    target_ids = [m["memory_id"] for m in existing]

    return {
        "example_id": example_id,
        "source_dataset": "MemoryAgentBench",
        "source_file": source_file,
        "split_group": {
            "source_file": source_file,
            "row_idx": row_idx,
            "source_subset": source_subset,
            "derivation": derivation,
        },
        "task_level": "operation_governance",
        "scenario_type": scenario_type,
        "failure_type": None,
        "history_context": [],
        "candidate_memory": memory_obj(candidate_text, memory_id=None, label=label),
        "existing_memories": existing,
        "current_query": "Decide how the memory system should handle the candidate memory.",
        "expected_answer": qa.get("answer"),
        "gold": {
            "primary_decision": label,
            "secondary_decision": None,
            "target_memory_ids": target_ids,
            "new_status_for_targets": None,
            "should_retrieve_candidate": None,
            "should_retrieve_existing": None,
            "reason": reason,
        },
        "diagnostic_tags": {
            "requires_temporal_reasoning": label == "UPDATE",
            "requires_contradiction_resolution": label == "UPDATE",
            "requires_sensitivity_reasoning": label in {"ASK_USER", "QUARANTINE"},
            "requires_appropriateness_reasoning": label in {"IGNORE", "ASK_USER", "QUARANTINE", "FORGET"},
            "requires_provenance_reasoning": True,
        },
        "annotation_metadata": {
            "version": VERSION,
            "label_source": "benchmark_mined_heuristic",
            "human_audited": False,
            "confidence": confidence,
            "status": status,
            "derivation": derivation,
            "review_flags": flags,
            "source_subset": source_subset,
            "row_idx": row_idx,
            "line_no": line_no,
            "fact_ids": fact_ids,
            "qa_idx": qa.get("qa_idx"),
            "question": qa.get("question"),
        },
    }


def route_status(label, confidence, flags, derivation):
    review_flags = set(flags)

    if "ambiguous_forget_marker" in review_flags:
        return "review"
    if label == "FORGET" and "no_target_memory_found" in review_flags:
        return "review"
    if "weak_update_marker" in review_flags:
        return "review"
    if "world_fact_not_user_centered" in review_flags:
        return "review"

    if label == "FORGET" and confidence >= 0.90 and derivation == "explicit_memory_deletion_request":
        return "clean"

    if label == "MERGE" and confidence >= 0.88 and derivation in {
        "exact_duplicate_memory_fact",
        "near_duplicate_memory_fact",
    }:
        return "clean"

    if label == "UPDATE" and confidence >= 0.82 and derivation == "strong_update_marker_same_subject":
        return "clean"

    return "review"


def is_probably_world_fact(text):
    low = normalize_text(text)
    if low.startswith("user ") or low.startswith("the user ") or " i " in f" {low} " or " my " in f" {low} ":
        return False
    if re.search(r"\b(capital|country|citizenship|born|died|spouse|author|director|sport|team|city|government)\b", low):
        return True
    return False


def add_candidate(
    rows,
    label,
    source_file,
    row_idx,
    line_no,
    candidate_text,
    existing_texts,
    reason,
    derivation,
    confidence,
    scenario_type,
    source_subset=None,
    fact_ids=None,
    qa=None,
    flags=None,
):
    flags = list(flags or [])
    if is_probably_world_fact(candidate_text):
        flags.append("world_fact_not_user_centered")

    status = route_status(label, confidence, flags, derivation)

    row = make_example(
        label=label,
        source_file=source_file,
        row_idx=row_idx,
        line_no=line_no,
        candidate_text=candidate_text,
        existing_texts=existing_texts,
        reason=reason,
        derivation=derivation,
        confidence=confidence,
        scenario_type=scenario_type,
        source_subset=source_subset,
        fact_ids=fact_ids,
        qa=qa,
        flags=sorted(set(flags)),
        status=status,
    )

    rows.append(row)


def mine_forget_from_qas(rows, source_file, row_idx, line_no, facts, qas, source_subset, scenario_type, max_per_row):
    made = 0

    for qa in qas:
        combined = normalize_ws(f"{qa.get('question', '')} {qa.get('answer', '')}")
        if not combined:
            continue

        if is_strong_forget_request(combined):
            answer_tokens = set(content_tokens(qa.get("answer", "")))
            existing = []
            if answer_tokens:
                for fact in facts:
                    ftoks = set(content_tokens(fact["text"]))
                    if answer_tokens and len(answer_tokens & ftoks) >= min(2, len(answer_tokens)):
                        existing.append(fact["text"])
                        if len(existing) >= 3:
                            break

            candidate = qa.get("question") or combined
            add_candidate(
                rows=rows,
                label="FORGET",
                source_file=source_file,
                row_idx=row_idx,
                line_no=line_no,
                candidate_text=candidate,
                existing_texts=existing,
                reason="Question or answer contains an explicit request to delete, forget, remove, erase, or stop retaining a memory.",
                derivation="explicit_memory_deletion_request",
                confidence=0.92,
                scenario_type=scenario_type,
                source_subset=source_subset,
                fact_ids=[],
                qa=qa,
                flags=[] if existing else ["no_target_memory_found"],
            )
            made += 1

        elif is_ambiguous_forget(combined):
            candidate = qa.get("question") or combined
            add_candidate(
                rows=rows,
                label="FORGET",
                source_file=source_file,
                row_idx=row_idx,
                line_no=line_no,
                candidate_text=candidate,
                existing_texts=[],
                reason="Text contains a forgetting-related marker, but it is not clearly a request to delete assistant memory.",
                derivation="ambiguous_forget_marker",
                confidence=0.55,
                scenario_type=scenario_type,
                source_subset=source_subset,
                fact_ids=[],
                qa=qa,
                flags=["ambiguous_forget_marker"],
            )
            made += 1

        if made >= max_per_row:
            break

    return made


def mine_forget_from_facts(rows, source_file, row_idx, line_no, facts, source_subset, scenario_type, max_per_row):
    made = 0

    for fact in facts:
        text = fact["text"]

        if is_strong_forget_request(text):
            add_candidate(
                rows=rows,
                label="FORGET",
                source_file=source_file,
                row_idx=row_idx,
                line_no=line_no,
                candidate_text=text,
                existing_texts=[],
                reason="Context fact contains an explicit memory deletion or forgetting request.",
                derivation="explicit_memory_deletion_request",
                confidence=0.90,
                scenario_type=scenario_type,
                source_subset=source_subset,
                fact_ids=[fact["fact_id"]],
                qa=None,
                flags=["no_target_memory_found"],
            )
            made += 1

        elif is_ambiguous_forget(text):
            add_candidate(
                rows=rows,
                label="FORGET",
                source_file=source_file,
                row_idx=row_idx,
                line_no=line_no,
                candidate_text=text,
                existing_texts=[],
                reason="Context fact contains a forgetting-related word, but not necessarily a deletion request.",
                derivation="ambiguous_forget_marker",
                confidence=0.50,
                scenario_type=scenario_type,
                source_subset=source_subset,
                fact_ids=[fact["fact_id"]],
                qa=None,
                flags=["ambiguous_forget_marker"],
            )
            made += 1

        if made >= max_per_row:
            break

    return made


def mine_updates(rows, source_file, row_idx, line_no, facts, source_subset, scenario_type, max_per_row):
    previous_by_subject = defaultdict(list)
    made = 0

    for fact in facts:
        text = fact["text"]
        skey = subject_key(text)

        if not skey:
            continue

        previous = previous_by_subject.get(skey, [])

        strong = has_strong_update_marker(text)
        weak = has_weak_update_marker(text)

        if previous and (strong or weak):
            prev_text = previous[-1]["text"]
            ratio = char_ratio(text, prev_text)
            jacc = token_jaccard(text, prev_text)

            if ratio < 0.98:
                if strong:
                    confidence = 0.84
                    derivation = "strong_update_marker_same_subject"
                    flags = ["update_marker_pair"]
                else:
                    confidence = 0.62
                    derivation = "weak_update_marker_same_subject"
                    flags = ["weak_update_marker"]

                reason = (
                    "Candidate contains a temporal or change marker and shares a subject with an earlier memory. "
                    f"char_similarity={ratio:.3f}, token_jaccard={jacc:.3f}."
                )

                add_candidate(
                    rows=rows,
                    label="UPDATE",
                    source_file=source_file,
                    row_idx=row_idx,
                    line_no=line_no,
                    candidate_text=text,
                    existing_texts=[prev_text],
                    reason=reason,
                    derivation=derivation,
                    confidence=confidence,
                    scenario_type=scenario_type,
                    source_subset=source_subset,
                    fact_ids=[previous[-1]["fact_id"], fact["fact_id"]],
                    qa=None,
                    flags=flags,
                )
                made += 1

        previous_by_subject[skey].append(fact)

        if len(previous_by_subject[skey]) > 5:
            previous_by_subject[skey] = previous_by_subject[skey][-5:]

        if made >= max_per_row:
            break

    return made


def mine_merges(rows, source_file, row_idx, line_no, facts, source_subset, scenario_type, max_per_row, near_threshold):
    made = 0
    seen_exact = {}

    for fact in facts:
        key = compact_key(fact["text"])
        if not key:
            continue

        if key in seen_exact:
            prev = seen_exact[key]
            add_candidate(
                rows=rows,
                label="MERGE",
                source_file=source_file,
                row_idx=row_idx,
                line_no=line_no,
                candidate_text=fact["text"],
                existing_texts=[prev["text"]],
                reason="Candidate is an exact duplicate of an existing memory and should be merged rather than stored separately.",
                derivation="exact_duplicate_memory_fact",
                confidence=0.95,
                scenario_type=scenario_type,
                source_subset=source_subset,
                fact_ids=[prev["fact_id"], fact["fact_id"]],
                qa=None,
                flags=[],
            )
            made += 1
            if made >= max_per_row:
                return made
        else:
            seen_exact[key] = fact

    buckets = defaultdict(list)
    compared_pairs = set()

    for fact in facts:
        toks = content_tokens(fact["text"])
        if len(toks) < 4:
            continue
        bkey = " ".join(toks[:4])
        bucket = buckets[bkey]

        for prev in bucket[:60]:
            pair_key = tuple(sorted([fact["fact_id"], prev["fact_id"]]))
            if pair_key in compared_pairs:
                continue
            compared_pairs.add(pair_key)

            ratio = char_ratio(fact["text"], prev["text"])
            jacc = token_jaccard(fact["text"], prev["text"])

            if ratio >= near_threshold or jacc >= near_threshold:
                confidence = max(ratio, jacc)
                add_candidate(
                    rows=rows,
                    label="MERGE",
                    source_file=source_file,
                    row_idx=row_idx,
                    line_no=line_no,
                    candidate_text=fact["text"],
                    existing_texts=[prev["text"]],
                    reason=(
                        "Candidate is a near-duplicate or complementary paraphrase of an existing memory. "
                        f"char_similarity={ratio:.3f}, token_jaccard={jacc:.3f}."
                    ),
                    derivation="near_duplicate_memory_fact",
                    confidence=round(float(confidence), 4),
                    scenario_type=scenario_type,
                    source_subset=source_subset,
                    fact_ids=[prev["fact_id"], fact["fact_id"]],
                    qa=None,
                    flags=[],
                )
                made += 1
                break

        bucket.append(fact)

        if made >= max_per_row:
            return made

    return made


def profile_raw_file(path):
    profile = {
        "rows": 0,
        "parse_errors": 0,
        "top_level_keys": Counter(),
        "metadata_keys": Counter(),
        "context_type_counter": Counter(),
        "questions_type_counter": Counter(),
        "answers_type_counter": Counter(),
        "context_item_count": {
            "min": None,
            "max": 0,
            "total": 0,
        },
    }

    for _, row in read_jsonl(path):
        if "__parse_error__" in row:
            profile["parse_errors"] += 1
            continue

        profile["rows"] += 1

        for k in row.keys():
            profile["top_level_keys"][k] += 1

        metadata = row.get("metadata", {})
        if isinstance(metadata, dict):
            for k in metadata.keys():
                profile["metadata_keys"][k] += 1

        profile["context_type_counter"][type(row.get("context")).__name__] += 1
        profile["questions_type_counter"][type(row.get("questions")).__name__] += 1
        profile["answers_type_counter"][type(row.get("answers")).__name__] += 1

        facts = extract_context_items(row.get("context"))
        n = len(facts)
        profile["context_item_count"]["total"] += n
        profile["context_item_count"]["max"] = max(profile["context_item_count"]["max"], n)
        if profile["context_item_count"]["min"] is None:
            profile["context_item_count"]["min"] = n
        else:
            profile["context_item_count"]["min"] = min(profile["context_item_count"]["min"], n)

    if profile["rows"]:
        profile["context_item_count"]["mean"] = profile["context_item_count"]["total"] / profile["rows"]
    else:
        profile["context_item_count"]["mean"] = 0.0

    for key in ["top_level_keys", "metadata_keys", "context_type_counter", "questions_type_counter", "answers_type_counter"]:
        profile[key] = dict(profile[key])

    return profile


def metadata_subset(row, source_file):
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["source", "subset", "task", "split", "name", "dataset"]:
            if key in metadata and metadata[key]:
                return str(metadata[key])
    return source_file.replace(".jsonl", "")


def scenario_from_file(source_file):
    base = source_file.replace(".jsonl", "").lower()
    if "accurate" in base:
        return "memoryagentbench_accurate_retrieval"
    if "long_range" in base:
        return "memoryagentbench_long_range_understanding"
    if "test_time" in base:
        return "memoryagentbench_test_time_learning"
    if "conflict" in base:
        return "memoryagentbench_conflict_resolution"
    return "memoryagentbench"


def mine_file(path, args):
    source_file = path.name
    rows = []
    rejected = []
    file_stats = {
        "source_file": source_file,
        "rows": 0,
        "parse_errors": 0,
        "context_items_scanned": 0,
        "qas_scanned": 0,
        "mined_counter": Counter(),
        "status_counter": Counter(),
        "derivation_counter": Counter(),
    }

    for row_idx, (line_no, row) in enumerate(read_jsonl(path)):
        if "__parse_error__" in row:
            file_stats["parse_errors"] += 1
            rejected.append({
                "source_file": source_file,
                "line_no": line_no,
                "reason": "parse_error",
                "error": row["__parse_error__"],
            })
            continue

        file_stats["rows"] += 1

        facts = extract_context_items(row.get("context"))
        qas = extract_qas(row)

        if len(facts) > args.max_facts_per_row:
            facts_for_pair_mining = facts[:args.max_facts_per_row]
        else:
            facts_for_pair_mining = facts

        source_subset = metadata_subset(row, source_file)
        scenario_type = scenario_from_file(source_file)

        file_stats["context_items_scanned"] += len(facts)
        file_stats["qas_scanned"] += len(qas)

        before = len(rows)

        mine_forget_from_qas(
            rows=rows,
            source_file=source_file,
            row_idx=row_idx,
            line_no=line_no,
            facts=facts_for_pair_mining,
            qas=qas,
            source_subset=source_subset,
            scenario_type=scenario_type,
            max_per_row=args.max_forget_per_row,
        )

        mine_forget_from_facts(
            rows=rows,
            source_file=source_file,
            row_idx=row_idx,
            line_no=line_no,
            facts=facts_for_pair_mining,
            source_subset=source_subset,
            scenario_type=scenario_type,
            max_per_row=args.max_forget_per_row,
        )

        mine_updates(
            rows=rows,
            source_file=source_file,
            row_idx=row_idx,
            line_no=line_no,
            facts=facts_for_pair_mining,
            source_subset=source_subset,
            scenario_type=scenario_type,
            max_per_row=args.max_update_per_row,
        )

        mine_merges(
            rows=rows,
            source_file=source_file,
            row_idx=row_idx,
            line_no=line_no,
            facts=facts_for_pair_mining,
            source_subset=source_subset,
            scenario_type=scenario_type,
            max_per_row=args.max_merge_per_row,
            near_threshold=args.merge_threshold,
        )

        after = len(rows)

        if after == before and len(rejected) < args.max_rejected_samples:
            rejected.append({
                "source_file": source_file,
                "line_no": line_no,
                "row_idx": row_idx,
                "reason": "no_operation_governance_signal_detected",
                "context_item_count": len(facts),
                "qa_count": len(qas),
                "source_subset": source_subset,
            })

    for row in rows:
        label = row["gold"]["primary_decision"]
        status = row["annotation_metadata"]["status"]
        derivation = row["annotation_metadata"]["derivation"]
        file_stats["mined_counter"][label] += 1
        file_stats["status_counter"][status] += 1
        file_stats["derivation_counter"][derivation] += 1

    for key in ["mined_counter", "status_counter", "derivation_counter"]:
        file_stats[key] = dict(file_stats[key])

    return rows, rejected, file_stats


def dedupe_rows(rows):
    deduped = []
    seen = set()

    for row in rows:
        label = row["gold"]["primary_decision"]
        cand = compact_key(row["candidate_memory"]["text"])
        existing = tuple(compact_key(m["text"]) for m in row.get("existing_memories", []))
        key = (label, cand, existing, row["source_file"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def cap_by_label(rows, max_per_label):
    counters = Counter()
    kept = []
    overflow = []

    for row in rows:
        label = row["gold"]["primary_decision"]
        if counters[label] < max_per_label:
            kept.append(row)
            counters[label] += 1
        else:
            overflow.append(row)

    return kept, overflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_conflict", action="store_true")
    parser.add_argument("--max_per_label", type=int, default=2000)
    parser.add_argument("--max_facts_per_row", type=int, default=5000)
    parser.add_argument("--max_merge_per_row", type=int, default=80)
    parser.add_argument("--max_update_per_row", type=int, default=80)
    parser.add_argument("--max_forget_per_row", type=int, default=40)
    parser.add_argument("--merge_threshold", type=float, default=0.88)
    parser.add_argument("--max_rejected_samples", type=int, default=500)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(DEFAULT_FILES)
    if args.include_conflict:
        files.append(OPTIONAL_CONFLICT_FILE)

    raw_profiles = {}
    all_rows = []
    all_rejected = []
    file_reports = {}

    for fname in files:
        path = RAW_DIR / fname
        if not path.exists():
            file_reports[fname] = {"missing": True}
            continue

        raw_profiles[fname] = profile_raw_file(path)
        rows, rejected, stats = mine_file(path, args)

        all_rows.extend(rows)
        all_rejected.extend(rejected)
        file_reports[fname] = stats

    pre_dedupe_count = len(all_rows)
    all_rows = dedupe_rows(all_rows)
    post_dedupe_count = len(all_rows)

    all_rows, overflow = cap_by_label(all_rows, args.max_per_label)
    all_rejected.extend([
        {
            "example_id": row["example_id"],
            "source_file": row["source_file"],
            "label": row["gold"]["primary_decision"],
            "reason": "max_per_label_overflow",
        }
        for row in overflow
    ])

    clean_rows = []
    review_rows = []

    for row in all_rows:
        status = row["annotation_metadata"]["status"]
        if status == "clean":
            clean_rows.append(row)
        else:
            review_rows.append(row)

    all_path = OUT_DIR / "all.jsonl"
    clean_path = OUT_DIR / "clean.jsonl"
    review_path = OUT_DIR / "review.jsonl"
    rejected_path = OUT_DIR / "rejected.jsonl"
    report_path = OUT_DIR / "report.json"
    schema_path = OUT_DIR / "memoryagentbench_remaining_schema_report.json"

    write_jsonl(all_path, all_rows)
    write_jsonl(clean_path, clean_rows)
    write_jsonl(review_path, review_rows)
    write_jsonl(rejected_path, all_rejected)
    write_json(schema_path, raw_profiles)

    def stats(rows):
        return {
            "total": len(rows),
            "label_counter": dict(Counter(row["gold"]["primary_decision"] for row in rows)),
            "source_file_counter": dict(Counter(row["source_file"] for row in rows)),
            "scenario_counter": dict(Counter(row["scenario_type"] for row in rows)),
            "derivation_counter": dict(Counter(row["annotation_metadata"]["derivation"] for row in rows)),
            "memory_type_counter": dict(Counter(row["candidate_memory"]["memory_type"] for row in rows)),
            "sensitivity_counter": dict(Counter(row["candidate_memory"]["sensitivity"] for row in rows)),
            "review_flag_counter": dict(Counter(flag for row in rows for flag in row["annotation_metadata"].get("review_flags", []))),
        }

    report = {
        "version": VERSION,
        "task": "operation_expansion_memoryagentbench_v2",
        "purpose": "Mine operation-governance candidates from the remaining MemoryAgentBench files before synthetic generation.",
        "input_files": [str(RAW_DIR / f) for f in files],
        "outputs": {
            "all": str(all_path),
            "clean": str(clean_path),
            "review": str(review_path),
            "rejected": str(rejected_path),
            "schema_report": str(schema_path),
        },
        "important_note": "Only clean examples should be considered for automatic merging. Review examples require manual inspection or stricter filtering.",
        "parameters": {
            "include_conflict": args.include_conflict,
            "max_per_label": args.max_per_label,
            "max_facts_per_row": args.max_facts_per_row,
            "max_merge_per_row": args.max_merge_per_row,
            "max_update_per_row": args.max_update_per_row,
            "max_forget_per_row": args.max_forget_per_row,
            "merge_threshold": args.merge_threshold,
        },
        "deduplication": {
            "pre_dedupe_count": pre_dedupe_count,
            "post_dedupe_count": post_dedupe_count,
            "duplicates_removed": pre_dedupe_count - post_dedupe_count,
            "max_per_label_overflow": len(overflow),
        },
        "all": stats(all_rows),
        "clean": stats(clean_rows),
        "review": stats(review_rows),
        "rejected_count": len(all_rejected),
        "file_reports": file_reports,
    }

    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
