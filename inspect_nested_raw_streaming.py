import json
from pathlib import Path
import ijson

BASE_DIR = Path(__file__).resolve().parent
MAX_TEXT = 700


def short(value, max_len=MAX_TEXT):
    text = str(value).replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_dict_structure(obj, indent=0, max_depth=3):
    prefix = "  " * indent

    if max_depth < 0:
        print(prefix + "...")
        return

    if isinstance(obj, dict):
        print(prefix + f"dict with {len(obj)} keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(prefix + f"- {k}: {type(v).__name__}")
            if isinstance(v, (dict, list)) and max_depth > 0:
                print_dict_structure(v, indent + 1, max_depth - 1)
            else:
                print(prefix + f"  value sample: {short(v, 300)}")

    elif isinstance(obj, list):
        print(prefix + f"list length={len(obj)}")
        if obj:
            print(prefix + f"first item type: {type(obj[0]).__name__}")
            print_dict_structure(obj[0], indent + 1, max_depth - 1)
    else:
        print(prefix + f"{type(obj).__name__}: {short(obj, 300)}")


def inspect_locomo():
    path = BASE_DIR / "raw/locomo/data/locomo10.json"
    print_header(f"LoCoMo nested inspection: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[0]
    print(f"Number of samples: {len(data)}")
    print(f"Sample keys: {list(sample.keys())}")
    print(f"sample_id: {sample.get('sample_id')}")

    for key in ["conversation", "qa", "session_summary", "event_summary", "observation"]:
        print_header(f"LoCoMo key: {key}")
        print_dict_structure(sample.get(key), max_depth=4)

    print_header("LoCoMo QA examples")
    qa = sample.get("qa", [])
    for i, item in enumerate(qa[:5]):
        print(f"\nQA {i}")
        print_dict_structure(item, max_depth=3)


def inspect_longmemeval_file(file_name, max_examples=3, count_types=True):
    path = BASE_DIR / "raw/longmemeval" / file_name
    if not path.exists():
        print_header(f"Missing LongMemEval file: {path}")
        return

    print_header(f"LongMemEval streaming inspection: {path}")

    examples = []
    question_type_counts = {}

    with open(path, "rb") as f:
        for idx, item in enumerate(ijson.items(f, "item")):
            if idx < max_examples:
                examples.append(item)

            if count_types:
                qt = item.get("question_type", "UNKNOWN")
                question_type_counts[qt] = question_type_counts.get(qt, 0) + 1

            if idx >= max_examples - 1 and not count_types:
                break

    print(f"Preview examples loaded: {len(examples)}")

    if examples:
        sample = examples[0]
        print(f"Sample keys: {list(sample.keys())}")

        print_header(f"{file_name} first example structure")
        print_dict_structure(sample, max_depth=4)

    if count_types:
        print_header(f"{file_name} question types distribution")
        for k, v in sorted(question_type_counts.items(), key=lambda x: str(x[0])):
            print(f"{k}: {v}")

    print_header(f"{file_name} first {len(examples)} examples, compact")
    for i, item in enumerate(examples):
        print(f"\nExample {i}")
        print(f"question_id: {item.get('question_id')}")
        print(f"question_type: {item.get('question_type')}")
        print(f"question_date: {item.get('question_date')}")
        print(f"question: {short(item.get('question'))}")
        print(f"answer: {short(item.get('answer'))}")
        print(f"answer_session_ids: {item.get('answer_session_ids')}")
        print(f"haystack_dates len: {len(item.get('haystack_dates', []))}")
        print(f"haystack_session_ids len: {len(item.get('haystack_session_ids', []))}")

        hs = item.get("haystack_sessions", [])
        print(f"haystack_sessions type: {type(hs).__name__}, len: {len(hs) if isinstance(hs, list) else 'NA'}")
        if isinstance(hs, list) and hs:
            print(f"first haystack session type: {type(hs[0]).__name__}")
            print(f"first haystack session sample: {short(hs[0], 700)}")


def inspect_memoryagentbench_file(file_name):
    path = BASE_DIR / "raw/memoryagentbench" / file_name
    if not path.exists():
        print_header(f"Missing MemoryAgentBench file: {path}")
        return

    print_header(f"MemoryAgentBench nested inspection: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            rows.append(json.loads(line))

    print(f"Loaded preview rows: {len(rows)}")

    for i, row in enumerate(rows):
        print_header(f"{file_name} row {i}")
        print_dict_structure(row, max_depth=4)
        print("\nCompact fields:")
        print(f"context sample: {short(row.get('context'), 700)}")
        print(f"questions: {short(row.get('questions'), 700)}")
        print(f"answers: {short(row.get('answers'), 700)}")
        print(f"metadata: {short(row.get('metadata'), 700)}")


def main():
    inspect_locomo()

    inspect_longmemeval_file("longmemeval_oracle.json", count_types=True)
    inspect_longmemeval_file("longmemeval_s_cleaned.json", count_types=True)

    # This file is large. Streaming avoids loading it into memory.
    inspect_longmemeval_file("longmemeval_m_cleaned.json", count_types=True)

    for file_name in [
        "Accurate_Retrieval.jsonl",
        "Test_Time_Learning.jsonl",
        "Long_Range_Understanding.jsonl",
        "Conflict_Resolution.jsonl",
    ]:
        inspect_memoryagentbench_file(file_name)


if __name__ == "__main__":
    main()
