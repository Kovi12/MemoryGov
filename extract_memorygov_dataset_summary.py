#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FINAL_LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

SOURCE_NAMES = [
    "LongMemEval",
    "LoCoMo",
    "MemoryAgentBench",
    "SyntheticAnchored",
    "synthetic",
    "locomo",
    "longmemeval",
    "memoryagentbench",
]

REDACTION_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "sk-***"),
    (re.compile(r"(?i)(api[_\- ]?key|token|password)\s*[:=]\s*\S+"), r"\1: ***"),
    (re.compile(r"[\w.\-]+@[\w.\-]+\.\w+"), "***@***"),
]

def redact(text):
    if not isinstance(text, str):
        return ""
    for pattern, repl in REDACTION_PATTERNS:
        text = pattern.sub(repl, text)
    return text

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {e}") from e


def get_label(row):
    if not isinstance(row, dict):
        return None

    gold = row.get("gold")

    if isinstance(gold, str):
        return gold

    if isinstance(gold, dict):
        return (
            gold.get("primary_decision")
            or gold.get("decision")
            or gold.get("label")
            or gold.get("target")
        )

    return (
        row.get("primary_decision")
        or row.get("decision")
        or row.get("label")
        or row.get("true_label")
        or row.get("target")
    )

def normalize_source(src):
    if not src:
        return "Unknown"
    s = str(src)
    low = s.lower()
    if "longmem" in low:
        return "LongMemEval"
    if "locomo" in low:
        return "LoCoMo"
    if "memoryagent" in low:
        return "MemoryAgentBench"
    if "synthetic" in low:
        return "SyntheticAnchored"
    return s

def get_source(row, path):
    src = (
        row.get("source_dataset")
        or row.get("source")
        or row.get("dataset")
        or row.get("origin")
        or str(path)
    )
    return normalize_source(src)

def get_candidate(row):
    cm = row.get("candidate_memory")
    if isinstance(cm, dict):
        return cm.get("text", "")
    if isinstance(cm, str):
        return cm
    return row.get("candidate") or row.get("candidate_text") or row.get("memory_text") or ""

def get_existing(row, limit=3):
    vals = []
    existing = row.get("existing_memories") or row.get("existing") or []
    for mem in existing:
        if isinstance(mem, dict):
            vals.append(mem.get("text", ""))
        elif isinstance(mem, str):
            vals.append(mem)
    return vals[:limit]

def get_history(row, limit=3):
    vals = []
    hist = row.get("history_context") or row.get("history") or []
    for item in hist:
        if isinstance(item, dict):
            speaker = item.get("speaker", "speaker")
            text = item.get("text", "")
            vals.append(f"{speaker}: {text}")
        elif isinstance(item, str):
            vals.append(item)
    return vals[:limit]

def get_query(row):
    return row.get("current_query") or row.get("query") or row.get("question") or ""


def get_gold_effect(row):
    if not isinstance(row, dict):
        return {}

    gold = row.get("gold")

    if isinstance(gold, str):
        return {"primary_decision": gold}

    if not isinstance(gold, dict):
        gold = {}

    out = {}
    for key in ["primary_decision", "decision", "label", "new_status_for_targets", "target_memory_ids", "reason"]:
        if key in gold:
            out[key] = gold[key]

    if not out:
        label = get_label(row)
        if label:
            out["primary_decision"] = label

    return out

def markdown_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)

def split_display_name(path):
    p = str(path).lower()
    name = path.name.lower()

    if "train" in p and "balanced" in p:
        return "Antrenare echilibrată"
    if "train" in p:
        return "Antrenare completă"
    if "val" in p or "valid" in p:
        return "Validare"
    if "benchmark" in p and "test" in p:
        return "Test benchmark-derived"
    if "synthetic" in p and "test" in p:
        return "Test sintetic lifecycle"
    if "combined" in p and "test" in p:
        return "Test combinat"
    if "test" in p:
        return "Test"
    return path.name

def discover_files(root):
    root = Path(root)
    candidates = []
    for p in root.rglob("*.jsonl"):
        low = str(p).lower()
        if "operation" not in low:
            continue
        if any(x in low for x in ["train", "val", "valid", "test"]):
            candidates.append(p)
    return sorted(candidates)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="processed", help="Folder scanned if --files is omitted.")
    parser.add_argument("--files", nargs="*", help="Explicit JSONL split files.")
    parser.add_argument("--samples-per-label", type=int, default=2)
    parser.add_argument("--out", default="memorygov_dataset_summary.md")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    if args.files:
        files = [Path(x) for x in args.files]
    else:
        files = discover_files(args.root)

    files = [p for p in files if p.exists()]

    if not files:
        raise SystemExit(
            "No JSONL files found. Use --files with explicit paths, or check --root."
        )

    label_rows = []
    source_rows_by_split = []
    examples = defaultdict(list)

    for path in files:
        label_counts = Counter()
        source_counts = Counter()

        for row in load_jsonl(path):
            label = get_label(row)
            if label not in FINAL_LABELS:
                continue

            label_counts[label] += 1
            source = get_source(row, path)
            source_counts[source] += 1

            if len(examples[label]) < args.samples_per_label:
                examples[label].append({
                    "source_file": str(path),
                    "source": source,
                    "query": redact(get_query(row)),
                    "candidate_memory": redact(get_candidate(row)),
                    "existing_memories": [redact(x) for x in get_existing(row)],
                    "history_context": [redact(x) for x in get_history(row)],
                    "gold": get_gold_effect(row),
                })

        total = sum(label_counts.values())
        if total == 0:
            continue

        label_rows.append(
            [split_display_name(path), total] + [label_counts[label] for label in FINAL_LABELS]
        )

        source_rows_by_split.append((split_display_name(path), total, source_counts))

    md = []
    md.append("# MemoryGov dataset summary\n")

    md.append("## Tabelul 3.1. Distribuția exemplelor operation governance pentru etichetele lifecycle finale\n")
    md.append(markdown_table(["Split", "Total"] + FINAL_LABELS, label_rows))

    md.append("\n## Tabelul 3.2. Proveniența exemplelor operation governance pentru etichetele lifecycle finale\n")

    all_sources = ["LongMemEval", "LoCoMo", "MemoryAgentBench", "SyntheticAnchored", "Unknown"]
    source_table_rows = []
    for split, total, counts in source_rows_by_split:
        source_table_rows.append(
            [split, total] + [counts.get(src, 0) for src in all_sources]
        )

    md.append(markdown_table(["Split", "Total"] + all_sources, source_table_rows))

    if not args.tables_only:
        md.append("\n## Exemple reale extrase din dataset\n")
        for label in FINAL_LABELS:
            md.append(f"\n### {label}\n")
            if not examples[label]:
                md.append("_Nu a fost găsit niciun exemplu pentru această etichetă._\n")
                continue
            for i, ex in enumerate(examples[label], 1):
                md.append(f"\nExemplul {i}\n")
                md.append("```json")
                md.append(json.dumps(ex, ensure_ascii=False, indent=2))
                md.append("```")

    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out}")
    print("Files used:")
    for p in files:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
