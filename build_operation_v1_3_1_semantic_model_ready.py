import json
import re
from collections import Counter
from pathlib import Path


SRC_DIR = Path("processed/memorygovbench_operation_v1_3_model_ready_noleak")
OUT_DIR = Path("processed/memorygovbench_operation_v1_3_1_semantic_model_ready_noleak")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_KEYS = ["operation_label_v1_3", "label", "operation_label", "target", "class_label", "y"]
TEXT_KEYS = ["text", "input", "prompt"]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_label(row):
    for key in LABEL_KEYS:
        if key in row and isinstance(row[key], str):
            return row[key]
    raise KeyError(f"No label field found: {sorted(row.keys())}")


def get_raw_text(row):
    for key in TEXT_KEYS:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]

    parts = []
    for key in ["user_message", "message", "candidate", "candidate_memory", "target_memory", "context"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(f"{key}: {v}")
    return "\n".join(parts)


def remove_allowed_labels(text):
    # Handles one-line and multiline allowed-label blocks.
    text = re.sub(
        r"Allowed labels:\s*(?:WRITE|IGNORE|UPDATE|MERGE|FORGET|ASK_USER|QUARANTINE|,\s*|\s*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


def remove_boilerplate(text):
    lines = []
    skip_next_label_line = False

    for line in text.splitlines():
        s = line.strip()

        if skip_next_label_line:
            skip_next_label_line = False
            continue

        if s.startswith("Task:"):
            continue
        if s.startswith("Question:"):
            continue
        if s.startswith("Decision target:"):
            continue
        if s.startswith("Choose the correct operation label"):
            continue
        if s.startswith("Allowed labels:"):
            skip_next_label_line = True
            continue

        lines.append(line)

    return "\n".join(lines)


def remove_metadata(text):
    # Remove explicit candidate metadata section.
    text = re.sub(
        r"Candidate metadata:\n(?:.*\n?){0,8}?(?=\n(?:Existing memories:|Decision target:|History context:|Candidate memory:|$))",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove inline bracket metadata.
    text = re.sub(r"\[type=.*?; sensitivity=.*?; status=.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[type=.*?; sensitivity=.*?\]", "", text, flags=re.IGNORECASE)

    # Remove standalone key=value metadata.
    text = re.sub(r"\btype\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsensitivity\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstatus\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)

    return text


def normalize_headers(text):
    text = text.replace("History context:", "History:")
    text = text.replace("Candidate memory:", "Candidate:")
    text = text.replace("Existing memories:", "Existing:")
    return text


def semantic_only(text):
    text = remove_boilerplate(text)
    text = remove_allowed_labels(text)
    text = remove_metadata(text)
    text = normalize_headers(text)

    # Drop excessive blank lines and whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_row(row):
    out = dict(row)
    raw_text = get_raw_text(row)
    sem_text = semantic_only(raw_text)

    out["raw_text_before_semantic_v1_3_1"] = raw_text
    out["text"] = sem_text
    out["input_variant"] = "semantic_only_v1_3_1"

    # Remove old prompt/input fields so trainers cannot accidentally pick scaffolded text.
    for key in ["input", "prompt"]:
        if key in out:
            out[f"raw_{key}_before_semantic_v1_3_1"] = out.pop(key)

    label = get_label(out)
    out["operation_label_v1_3_1"] = label

    return out


def main():
    src_files = sorted(SRC_DIR.glob("*.jsonl"))
    if not src_files:
        raise SystemExit(f"No JSONL files found in {SRC_DIR}")

    report = {
        "source_dir": str(SRC_DIR),
        "output_dir": str(OUT_DIR),
        "files": {},
        "label_counts_total": Counter(),
    }

    for src in src_files:
        rows = read_jsonl(src)
        out_rows = [convert_row(r) for r in rows]

        counts = Counter(get_label(r) for r in out_rows)

        dst = OUT_DIR / src.name
        write_jsonl(dst, out_rows)

        report["files"][src.name] = {
            "rows": len(out_rows),
            "label_counts": dict(counts),
            "output_file": str(dst),
        }
        report["label_counts_total"].update(counts)

    report["label_counts_total"] = dict(report["label_counts_total"])
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
