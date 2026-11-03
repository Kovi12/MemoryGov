import json
import argparse
import hashlib
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_hash(*parts):
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def clip_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED_FOR_EXTRACTION_TASK]"


def build_prompt(example, max_session_chars):
    candidate = example.get("candidate_memory") or {}
    session_text = candidate.get("text", "")
    clipped_session = clip_text(session_text, max_session_chars)

    question = example.get("current_query", "")
    answer = example.get("expected_answer", "")
    scenario_type = example.get("scenario_type", "")
    source_file = example.get("source_file", "")

    prompt = f"""You are building a long-term memory dataset for an LLM agent.

Your task is to extract atomic memory candidates from one conversation session.

The final system must decide whether each candidate should be written to long-term memory, updated, merged, suppressed, forgotten, quarantined, or ignored. For this step, extract useful atomic memory candidates and assign metadata.

Important rules:
1. Extract atomic memories, not long summaries.
2. Each memory must contain exactly one clear fact, preference, event, user state, user profile detail, or procedural preference.
3. Prefer user-specific and future-useful information.
4. Do not store generic assistant advice unless it captures a user preference, instruction, or stable procedural rule.
5. Do not invent facts.
6. If a fact is only temporary or not useful long-term, either omit it or include it with should_write=false.
7. Sensitive memories are allowed, but mark sensitivity correctly.
8. The benchmark question and answer are provided only to identify which part of the session was useful. Do not copy the question as a memory.
9. Return valid JSON only. No markdown.

Memory types:
- episodic: specific event or interaction
- semantic: stable fact about the user or their world
- preference: user likes, dislikes, habits, recurring choices
- procedural: how the user wants something done
- sensitive: medical, financial, identity, legal, private relationship, access, security, or high-impact information
- authorization: permission, access, allowed recipient, policy-related fact
- project_state: user project direction, decisions, milestones, constraints

Sensitivity:
- low
- medium
- high

Stability:
- temporary
- session_relevant
- long_term
- unknown

Return schema:
{{
  "memories": [
    {{
      "memory_text": "atomic memory sentence",
      "memory_type": "episodic|semantic|preference|procedural|sensitive|authorization|project_state",
      "sensitivity": "low|medium|high",
      "stability": "temporary|session_relevant|long_term|unknown",
      "should_write": true,
      "reason": "short reason",
      "evidence_span": "short quote from the session supporting the memory"
    }}
  ]
}}

Source file:
{source_file}

Scenario type:
{scenario_type}

Benchmark future question:
{question}

Benchmark expected answer:
{answer}

Conversation session:
{clipped_session}
"""
    return prompt


def build_task(example, max_session_chars):
    split = example.get("split_group") or {}
    candidate = example.get("candidate_memory") or {}

    source_example_id = example.get("example_id")
    question_id = split.get("question_id", "unknown_question")
    session_id = split.get("session_id", "unknown_session")
    source_file = example.get("source_file", "unknown_source")

    task_id = "lme_extract_" + stable_hash(source_example_id, question_id, session_id, source_file)

    return {
        "task_id": task_id,
        "task_type": "atomic_memory_extraction",
        "source_dataset": "LongMemEval",
        "source_file": source_file,
        "source_example_id": source_example_id,
        "question_id": question_id,
        "session_id": session_id,
        "scenario_type": example.get("scenario_type"),
        "current_query": example.get("current_query"),
        "expected_answer": example.get("expected_answer"),
        "candidate_memory_type": candidate.get("memory_type"),
        "candidate_sensitivity": candidate.get("sensitivity"),
        "candidate_valid_from": candidate.get("valid_from"),
        "prompt": build_prompt(example, max_session_chars),
        "annotation_metadata": {
            "task_builder": "prepare_longmemeval_candidate_extraction_tasks.py",
            "label_source": "pending_llm_extraction",
            "uses_expected_answer_for_extraction": True,
            "max_session_chars": max_session_chars,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="processed/longmemeval_operation_write_canonical.jsonl",
    )
    parser.add_argument(
        "--output",
        default="processed/longmemeval_candidate_extraction_tasks.jsonl",
    )
    parser.add_argument(
        "--stats",
        default="processed/longmemeval_candidate_extraction_tasks_stats.json",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_session_chars",
        type=int,
        default=28000,
    )
    args = parser.parse_args()

    input_path = BASE_DIR / args.input
    output_path = BASE_DIR / args.output
    stats_path = BASE_DIR / args.stats

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    scenario_counter = Counter()
    source_counter = Counter()
    sensitivity_counter = Counter()
    memory_type_counter = Counter()
    prompt_lengths = []

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, example in enumerate(read_jsonl(input_path)):
            if args.max_examples is not None and idx >= args.max_examples:
                break

            task = build_task(example, args.max_session_chars)
            out.write(json.dumps(task, ensure_ascii=False) + "\n")

            total += 1
            scenario_counter[task.get("scenario_type", "UNKNOWN")] += 1
            source_counter[task.get("source_file", "UNKNOWN")] += 1
            sensitivity_counter[task.get("candidate_sensitivity", "UNKNOWN")] += 1
            memory_type_counter[task.get("candidate_memory_type", "UNKNOWN")] += 1
            prompt_lengths.append(len(task["prompt"]))

    prompt_lengths_sorted = sorted(prompt_lengths)
    if prompt_lengths_sorted:
        n = len(prompt_lengths_sorted)

        def pct(p):
            idx = int((p / 100) * (n - 1))
            return prompt_lengths_sorted[idx]

        length_stats = {
            "min": prompt_lengths_sorted[0],
            "p50": pct(50),
            "p90": pct(90),
            "p95": pct(95),
            "p99": pct(99),
            "max": prompt_lengths_sorted[-1],
        }
    else:
        length_stats = {}

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total_tasks": total,
        "scenario_counter": dict(scenario_counter),
        "source_counter": dict(source_counter),
        "sensitivity_counter": dict(sensitivity_counter),
        "memory_type_counter": dict(memory_type_counter),
        "prompt_length_chars": length_stats,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nSaved tasks to: {output_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
