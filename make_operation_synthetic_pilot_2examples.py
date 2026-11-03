import json
import re
from pathlib import Path


BASE = Path(__file__).resolve().parent

INPUT = BASE / "processed/operation_synthetic_v1/tasks_pilot.jsonl"
OUTPUT = BASE / "processed/operation_synthetic_v1/tasks_pilot_2examples.jsonl"
REPORT = BASE / "processed/operation_synthetic_v1/tasks_pilot_2examples_report.json"


def rewrite_prompt(prompt):
    if not isinstance(prompt, str):
        return prompt

    replacements = [
        (r"exactly\s+5\s+examples", "exactly 2 examples"),
        (r"exactly\s+five\s+examples", "exactly two examples"),
        (r"generate\s+5\s+examples", "generate 2 examples"),
        (r"generate\s+five\s+examples", "generate two examples"),
        (r"create\s+5\s+examples", "create 2 examples"),
        (r"create\s+five\s+examples", "create two examples"),
        (r"produce\s+5\s+examples", "produce 2 examples"),
        (r"produce\s+five\s+examples", "produce two examples"),
        (r"\"examples\"\s*array\s*must\s*contain\s*exactly\s*5", "\"examples\" array must contain exactly 2"),
        (r"examples_requested\s*[:=]\s*5", "examples_requested: 2"),
    ]

    out = prompt
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    return out


def main():
    rows = []
    unchanged_prompt_contains_5 = 0

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            row["examples_requested"] = 2
            row["pilot_variant"] = "all_tasks_2_examples"

            if "prompt" in row:
                row["prompt"] = rewrite_prompt(row["prompt"])
                if re.search(r"\b5\s+examples\b|\bfive\s+examples\b|exactly\s+5\b|exactly\s+five\b", row["prompt"], flags=re.IGNORECASE):
                    unchanged_prompt_contains_5 += 1

            rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": str(INPUT),
        "output": str(OUTPUT),
        "total_tasks": len(rows),
        "examples_requested": 2,
        "prompts_still_containing_5_examples_pattern": unchanged_prompt_contains_5,
    }

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
