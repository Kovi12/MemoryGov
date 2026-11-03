import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short(text, max_len=240):
    text = str(text).replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def main():
    input_path = BASE_DIR / "processed/longmemeval_atomic_operation_candidates_qwen32b.jsonl"
    out_dir = BASE_DIR / "processed/audit_longmemeval_atomic"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    decision_counter = Counter()
    memory_type_counter = Counter()
    sensitivity_counter = Counter()
    stability_counter = Counter()
    scenario_counter = Counter()
    decision_by_type = defaultdict(Counter)
    decision_by_sensitivity = defaultdict(Counter)
    scenario_by_decision = defaultdict(Counter)

    text_lengths = []
    evidence_lengths = []

    samples_by_decision = defaultdict(list)
    samples_by_type = defaultdict(list)
    samples_by_sensitivity = defaultdict(list)
    suspicious = []

    for ex in read_jsonl(input_path):
        total += 1

        candidate = ex.get("candidate_memory") or {}
        gold = ex.get("gold") or {}

        decision = gold.get("primary_decision", "UNKNOWN")
        memory_type = candidate.get("memory_type", "UNKNOWN")
        sensitivity = candidate.get("sensitivity", "UNKNOWN")
        stability = candidate.get("stability", "UNKNOWN")
        scenario = ex.get("scenario_type", "UNKNOWN")
        text = candidate.get("text", "")
        evidence = candidate.get("evidence_span", "")

        decision_counter[decision] += 1
        memory_type_counter[memory_type] += 1
        sensitivity_counter[sensitivity] += 1
        stability_counter[stability] += 1
        scenario_counter[scenario] += 1
        decision_by_type[decision][memory_type] += 1
        decision_by_sensitivity[decision][sensitivity] += 1
        scenario_by_decision[scenario][decision] += 1

        text_lengths.append(len(text))
        evidence_lengths.append(len(evidence))

        compact = {
            "example_id": ex.get("example_id"),
            "decision": decision,
            "memory_type": memory_type,
            "sensitivity": sensitivity,
            "stability": stability,
            "scenario_type": scenario,
            "memory_text": text,
            "evidence_span": evidence,
            "reason": gold.get("reason"),
            "query": ex.get("current_query"),
            "expected_answer": ex.get("expected_answer"),
        }

        if len(samples_by_decision[decision]) < 15:
            samples_by_decision[decision].append(compact)

        if len(samples_by_type[memory_type]) < 15:
            samples_by_type[memory_type].append(compact)

        if len(samples_by_sensitivity[sensitivity]) < 15:
            samples_by_sensitivity[sensitivity].append(compact)

        lowered = text.lower()

        suspicious_flags = []

        if "user" not in lowered and "the user" not in lowered and not lowered.startswith("user"):
            suspicious_flags.append("memory_text_may_not_be_user_centered")

        if len(text) < 15:
            suspicious_flags.append("very_short_memory")

        if len(text) > 300:
            suspicious_flags.append("possibly_not_atomic")

        if not evidence:
            suspicious_flags.append("missing_evidence")

        if sensitivity == "low" and any(term in lowered for term in [
            "credit card", "bank", "insurance", "medical", "doctor", "diagnosis",
            "therapy", "salary", "password", "passport", "religion", "transgender",
            "political", "adoption", "accident"
        ]):
            suspicious_flags.append("possibly_under_sensitive")

        if memory_type == "procedural" and any(term in lowered for term in [
            "uses", "likes", "prefers", "favorite", "usually", "tracks"
        ]):
            suspicious_flags.append("possibly_preference_or_semantic_not_procedural")

        if suspicious_flags:
            compact["suspicious_flags"] = suspicious_flags
            suspicious.append(compact)

    def length_stats(values):
        if not values:
            return {}
        values = sorted(values)
        n = len(values)

        def pct(p):
            idx = int((p / 100) * (n - 1))
            return values[idx]

        return {
            "min": values[0],
            "p50": pct(50),
            "p90": pct(90),
            "p95": pct(95),
            "p99": pct(99),
            "max": values[-1],
        }

    report = {
        "input": str(input_path),
        "total_examples": total,
        "decision_counter": dict(decision_counter),
        "memory_type_counter": dict(memory_type_counter),
        "sensitivity_counter": dict(sensitivity_counter),
        "stability_counter": dict(stability_counter),
        "scenario_counter": dict(scenario_counter),
        "decision_by_type": {k: dict(v) for k, v in decision_by_type.items()},
        "decision_by_sensitivity": {k: dict(v) for k, v in decision_by_sensitivity.items()},
        "scenario_by_decision": {k: dict(v) for k, v in scenario_by_decision.items()},
        "memory_text_length_chars": length_stats(text_lengths),
        "evidence_span_length_chars": length_stats(evidence_lengths),
        "suspicious_count": len(suspicious),
        "suspicious_preview_count": min(200, len(suspicious)),
    }

    write_json(out_dir / "audit_report.json", report)
    write_json(out_dir / "samples_by_decision.json", dict(samples_by_decision))
    write_json(out_dir / "samples_by_type.json", dict(samples_by_type))
    write_json(out_dir / "samples_by_sensitivity.json", dict(samples_by_sensitivity))
    write_jsonl(out_dir / "suspicious_examples_preview.jsonl", suspicious[:200])

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved audit files in: {out_dir}")


if __name__ == "__main__":
    main()
