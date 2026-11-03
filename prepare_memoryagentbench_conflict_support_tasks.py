import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = BASE_DIR / "raw" / "memoryagentbench" / "Conflict_Resolution.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "processed" / "memoryagentbench_conflict_support_selection_tasks.jsonl"
DEFAULT_REPORT = BASE_DIR / "processed" / "memoryagentbench_conflict_support_selection_report.json"
DEFAULT_SAMPLE = BASE_DIR / "processed" / "samples" / "memoryagentbench_conflict_support_selection_sample.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "him", "his", "how", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
    "you", "your", "into", "about", "associated", "related", "known", "name",
    "current", "location", "country", "city", "continent", "sport", "position",
    "team", "played", "plays", "person", "place"
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(prefix, payload):
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def tokenize(text):
    if text is None:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def normalize_answer_list(answer_item):
    if isinstance(answer_item, list):
        return [str(x) for x in answer_item if str(x).strip()]
    if answer_item is None:
        return []
    return [str(answer_item)]


def parse_facts(context):
    facts = []
    current_id = None
    current_text_parts = []

    for raw_line in context.splitlines():
        line = raw_line.rstrip()
        match = re.match(r"^\s*(\d+)\.\s+(.*)\s*$", line)

        if match:
            if current_id is not None:
                text = " ".join(current_text_parts).strip()
                if text:
                    facts.append({"fact_id": str(current_id), "text": text})
            current_id = match.group(1)
            current_text_parts = [match.group(2).strip()]
        else:
            stripped = line.strip()
            if current_id is not None and stripped:
                current_text_parts.append(stripped)

    if current_id is not None:
        text = " ".join(current_text_parts).strip()
        if text:
            facts.append({"fact_id": str(current_id), "text": text})

    return facts


def extract_capitalized_phrases(text):
    if not text:
        return []

    phrases = []
    pattern = r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*\b"
    for match in re.finditer(pattern, text):
        phrase = match.group(0).strip()
        if len(phrase) >= 3 and phrase.lower() not in STOPWORDS:
            phrases.append(phrase.lower())

    return list(dict.fromkeys(phrases))


def score_fact(fact, question, answers):
    fact_text = fact["text"]
    fact_low = fact_text.lower()

    q_tokens = set(tokenize(question))
    answer_text = " ".join(answers)
    a_tokens = set(tokenize(answer_text))

    fact_tokens = set(tokenize(fact_text))
    score = 0.0

    overlap_q = q_tokens & fact_tokens
    overlap_a = a_tokens & fact_tokens

    score += len(overlap_q) * 2.0
    score += len(overlap_a) * 4.0

    for answer in answers:
        ans = str(answer).strip().lower()
        if ans and len(ans) > 1 and ans in fact_low:
            score += 12.0

    for phrase in extract_capitalized_phrases(question):
        if phrase in fact_low:
            score += 10.0

    relation_keywords = [
        "spouse", "author", "citizenship", "birthplace", "capital", "sport",
        "position", "religion", "founded", "director", "manager", "occupation",
        "language", "country", "continent", "location", "educated", "capital"
    ]
    for kw in relation_keywords:
        if kw in question.lower() and kw in fact_low:
            score += 2.5

    if re.search(r"\bplays the position\b", fact_low):
        score += 2.0
    if re.search(r"\bassociated with the sport\b", fact_low):
        score += 2.0
    if re.search(r"\bis married to\b", fact_low):
        score += 2.0
    if re.search(r"\bspouse\b", question.lower()) and "married" in fact_low:
        score += 3.0

    return score


def select_candidate_facts(facts, question, answers, top_k):
    scored = []
    for fact in facts:
        score = score_fact(fact, question, answers)
        fact_id = fact["fact_id"]
        try:
            numeric_id = int(fact_id)
        except ValueError:
            numeric_id = 10**12
        scored.append((score, numeric_id, fact))

    scored.sort(key=lambda x: (-x[0], x[1]))

    selected = []
    seen = set()

    for score, _, fact in scored:
        if len(selected) >= top_k:
            break
        selected.append(fact)
        seen.add(fact["fact_id"])

    if len(selected) < top_k:
        for fact in facts:
            if fact["fact_id"] not in seen:
                selected.append(fact)
                seen.add(fact["fact_id"])
                if len(selected) >= top_k:
                    break

    return selected


def build_prompt(source_subset, question, answers, candidate_facts):
    allowed_ids = [str(f["fact_id"]) for f in candidate_facts]
    facts_text = "\n".join(
        f"- {fact['fact_id']}: {fact['text']}"
        for fact in candidate_facts
    )

    expected_answers = json.dumps(answers, ensure_ascii=False)
    allowed_ids_text = ", ".join(allowed_ids)

    return f"""You are building a memory-governance dataset for an LLM agent.

Your task is to identify which listed memory facts are needed to answer a benchmark question.

Important rules:
1. Use only the listed candidate facts.
2. Do not use outside knowledge.
3. The facts may be synthetic, fictional, outdated, or inconsistent with the real world. Treat the listed facts as the only source of truth.
4. Select only fact IDs that appear explicitly in the candidate facts list.
5. If the answer cannot be derived from the listed candidate facts, return an empty list.
6. Do not invent fact IDs.
7. Do not select a fact because it is true in the real world. Select it only if it appears in the candidate facts.
8. Prefer the smallest set of facts needed to support the expected answer.
9. Return valid JSON only. No markdown.

Return schema:
{{
  "support_fact_ids": ["fact_id_1", "fact_id_2"],
  "reason": "short explanation based only on listed candidate facts"
}}

Source subset:
{source_subset}

Question:
{question}

Expected answer or acceptable answers:
{expected_answers}

Allowed fact IDs are exactly:
{allowed_ids_text}

Candidate facts:
{facts_text}
"""


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def summarize_lengths(values):
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--sample_output", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--top_k", type=int, default=120)
    parser.add_argument("--max_questions_per_row", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    sample_output_path = Path(args.sample_output)

    tasks = []
    source_counter = Counter()
    fact_count_values = []
    candidate_count_values = []
    prompt_lengths = []

    total_rows = 0
    skipped_rows_no_facts = 0
    skipped_empty_question = 0

    for row_idx, row in enumerate(read_jsonl(input_path)):
        total_rows += 1

        context = row.get("context", "")
        facts = parse_facts(context)
        if not facts:
            skipped_rows_no_facts += 1
            continue

        questions = row.get("questions") or []
        answers = row.get("answers") or []
        metadata = row.get("metadata") or {}
        qa_pair_ids = metadata.get("qa_pair_ids") or []
        source_subset = metadata.get("source") or f"row_{row_idx}"

        fact_count_values.append(len(facts))

        max_q = min(len(questions), args.max_questions_per_row)
        for qa_idx in range(max_q):
            question = str(questions[qa_idx]).strip()
            if not question:
                skipped_empty_question += 1
                continue

            answer_item = answers[qa_idx] if qa_idx < len(answers) else []
            expected_answers = normalize_answer_list(answer_item)
            qa_pair_id = qa_pair_ids[qa_idx] if qa_idx < len(qa_pair_ids) else f"{source_subset}_no{qa_idx}"

            candidate_facts = select_candidate_facts(
                facts=facts,
                question=question,
                answers=expected_answers,
                top_k=args.top_k,
            )

            candidate_count_values.append(len(candidate_facts))

            prompt = build_prompt(
                source_subset=source_subset,
                question=question,
                answers=expected_answers,
                candidate_facts=candidate_facts,
            )

            prompt_lengths.append(len(prompt))

            payload_for_id = {
                "source_subset": source_subset,
                "row_idx": row_idx,
                "qa_idx": qa_idx,
                "qa_pair_id": qa_pair_id,
                "question": question,
            }

            task = {
                "task_id": stable_id("mab_conflict_support", payload_for_id),
                "task_type": "support_fact_selection",
                "source_dataset": "MemoryAgentBench",
                "source_file": "Conflict_Resolution.jsonl",
                "source_subset": source_subset,
                "row_idx": row_idx,
                "qa_idx": qa_idx,
                "qa_pair_id": qa_pair_id,
                "scenario_type": "memoryagentbench_conflict_resolution",
                "current_query": question,
                "expected_answers": expected_answers,
                "all_fact_count": len(facts),
                "candidate_fact_count": len(candidate_facts),
                "candidate_facts": candidate_facts,
                "prompt": prompt,
                "annotation_metadata": {
                    "task_builder": Path(__file__).name,
                    "label_source": "pending_llm_support_selection",
                    "uses_expected_answer_for_support_selection": True,
                    "top_k": args.top_k,
                    "strict_allowed_fact_ids": True,
                    "outside_knowledge_allowed": False,
                },
            }

            tasks.append(task)
            source_counter[source_subset] += 1

    write_jsonl(output_path, tasks)

    sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    if tasks:
        with open(sample_output_path, "w", encoding="utf-8") as f:
            json.dump(tasks[0], f, indent=2, ensure_ascii=False)

    report = {
        "version": "v3.33",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "total_rows": total_rows,
        "total_tasks": len(tasks),
        "skipped_rows_no_facts": skipped_rows_no_facts,
        "skipped_empty_question": skipped_empty_question,
        "source_counter": dict(source_counter),
        "top_k": args.top_k,
        "fact_count_per_row": summarize_lengths(fact_count_values),
        "candidate_fact_count_per_task": summarize_lengths(candidate_count_values),
        "prompt_length_chars": summarize_lengths(prompt_lengths),
        "sample_output": str(sample_output_path.resolve()),
        "fixes": [
            "Prompt now forbids outside knowledge.",
            "Prompt now states that benchmark facts may be synthetic or inconsistent with real-world facts.",
            "Prompt now lists allowed fact IDs explicitly.",
            "Prompt now states that unlisted fact IDs are invalid.",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
