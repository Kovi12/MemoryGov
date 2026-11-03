import json
import re
from pathlib import Path
from collections import Counter, defaultdict


SRC_DIR = Path("processed/memorygovbench_operation_v1_3_1_semantic_model_ready_noleak")
OUT_DIR = Path("processed/memorygovbench_operation_v1_3_2_no_confirmation_semantic_model_ready_noleak")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_KEYS = [
    "operation_label_v1_3_1",
    "operation_label_v1_3",
    "label",
    "operation_label",
    "target",
    "class_label",
    "y",
]


UNCERTAINTY_PATTERNS = [
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\buncertain\b",
    r"\bmaybe\b",
    r"\bmight\b",
    r"\bshould you\b",
    r"\bdo not know if\b",
    r"\bdon't know if\b",
    r"\bnot ready\b",
    r"\bhesitant\b",
    r"\bunsure whether\b",
    r"\bnot certain\b",
]

MEMORY_CONSENT_PATTERNS = [
    r"\bremember\b",
    r"\bstore\b",
    r"\bsave\b",
    r"\bkeep\b",
    r"\brecord\b",
    r"\bretain\b",
    r"\bmemorize\b",
    r"\bmemory\b",
]

DIRECT_CONFIRMATION_PATTERNS = [
    r"not sure i want you to remember",
    r"not sure whether i want you to remember",
    r"i am not sure i want you to remember",
    r"i'm not sure i want you to remember",
    r"ask me before remembering",
    r"ask before remembering",
    r"confirm before storing",
    r"confirm before saving",
    r"do not store this unless i confirm",
    r"don't store this unless i confirm",
]


def read_jsonl(path):
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


def get_text(row):
    v = row.get("text")
    if isinstance(v, str) and v.strip():
        return v
    for key in ["input", "prompt", "raw_text_before_semantic_v1_3_1"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return json.dumps(row, ensure_ascii=False)


def is_confirmation_boundary_example(row):
    text = get_text(row).lower()

    # Direct old ASK_USER-style consent-boundary examples.
    for pat in DIRECT_CONFIRMATION_PATTERNS:
        if re.search(pat, text):
            return True

    has_uncertainty = any(re.search(pat, text) for pat in UNCERTAINTY_PATTERNS)
    has_memory_consent = any(re.search(pat, text) for pat in MEMORY_CONSENT_PATTERNS)

    if has_uncertainty and has_memory_consent:
        return True

    return False


def main():
    src_files = sorted(SRC_DIR.glob("*.jsonl"))
    if not src_files:
        raise SystemExit(f"No JSONL files found in {SRC_DIR}")

    report = {
        "source_dir": str(SRC_DIR),
        "output_dir": str(OUT_DIR),
        "files": {},
        "removed_examples_preview": [],
    }

    total_kept = Counter()
    total_removed = Counter()

    for src in src_files:
        rows = read_jsonl(src)

        kept = []
        removed = []

        for r in rows:
            if is_confirmation_boundary_example(r):
                removed.append(r)
            else:
                kept.append(r)

        dst = OUT_DIR / src.name
        write_jsonl(dst, kept)

        kept_counts = Counter(get_label(r) for r in kept)
        removed_counts = Counter(get_label(r) for r in removed)

        total_kept.update(kept_counts)
        total_removed.update(removed_counts)

        report["files"][src.name] = {
            "input_rows": len(rows),
            "kept_rows": len(kept),
            "removed_rows": len(removed),
            "kept_label_counts": dict(kept_counts),
            "removed_label_counts": dict(removed_counts),
            "output_file": str(dst),
        }

        for r in removed[:10]:
            report["removed_examples_preview"].append({
                "file": src.name,
                "label": get_label(r),
                "text": get_text(r)[:500],
            })

    report["total_kept_label_counts"] = dict(total_kept)
    report["total_removed_label_counts"] = dict(total_removed)
    report["total_kept_rows"] = sum(total_kept.values())
    report["total_removed_rows"] = sum(total_removed.values())

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
