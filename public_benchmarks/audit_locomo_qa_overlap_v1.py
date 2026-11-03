import hashlib
import json
import re
from pathlib import Path
from collections import Counter

RAW_PATH = Path("raw/locomo")
OUT_DIR = Path("public_benchmarks/processed/locomo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEMGOV_PATHS = [
    Path("processed/locomo_operation_write.jsonl"),
    Path("processed/locomo_retrieval_governance.jsonl"),
    Path("processed/splits_locomo/operation_train.jsonl"),
    Path("processed/splits_locomo/operation_val.jsonl"),
    Path("processed/splits_locomo/operation_test.jsonl"),
    Path("processed/splits_locomo/retrieval_balanced_train.jsonl"),
    Path("processed/splits_locomo/retrieval_balanced_val.jsonl"),
    Path("processed/splits_locomo/retrieval_balanced_test.jsonl"),
    Path("processed/splits_locomo/retrieval_natural_train.jsonl"),
    Path("processed/splits_locomo/retrieval_natural_val.jsonl"),
    Path("processed/splits_locomo/retrieval_natural_test.jsonl"),
]

THRESHOLD = 0.20
MIN_SHARED_SHINGLES = 3

OUT_NON = OUT_DIR / "locomo_qa_audited_nonoverlap.jsonl"
OUT_SEEN = OUT_DIR / "locomo_qa_source_overlap_or_seen.jsonl"
SUMMARY = OUT_DIR / "locomo_qa_overlap_summary.json"


def norm(s):
    return " ".join(str(s or "").lower().split())


def tokens(s):
    return re.findall(r"[a-z0-9_./:@-]+", norm(s))


def h(x):
    return hashlib.blake2b(repr(x).encode("utf-8", errors="ignore"), digest_size=8).hexdigest()


def shingles(text, n=5):
    toks = tokens(text)
    if len(toks) < n:
        return {h(t) for t in toks}
    return {h(tuple(toks[i:i+n])) for i in range(len(toks) - n + 1)}


def flatten(obj):
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if str(k).lower() in {"embedding", "vector"}:
                continue
            parts.append(str(k))
            parts.append(flatten(v))
        return " ".join(parts)
    if isinstance(obj, list):
        return " ".join(flatten(x) for x in obj)
    return str(obj)


def iter_json_files(paths):
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            for f in sorted(list(p.rglob("*.json")) + list(p.rglob("*.jsonl"))):
                yield f
        elif p.suffix in {".json", ".jsonl"}:
            yield p


def iter_records(path):
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return

    if path.suffix == ".jsonl":
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield f"{path}:{i}", json.loads(line)
            except Exception:
                continue
        return

    if path.suffix == ".json":
        try:
            obj = json.loads(text)
        except Exception:
            return
        yield str(path), obj


def recursive_find_qa(obj, path):
    found = []

    if isinstance(obj, dict):
        keys = {str(k).lower() for k in obj.keys()}

        if "question" in keys and "answer" in keys:
            found.append((path, obj))

        for k, v in obj.items():
            found.extend(recursive_find_qa(v, f"{path}.{k}"))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(recursive_find_qa(v, f"{path}[{i}]"))

    return found


def build_reference_shingles():
    ref = set()
    records = 0

    for f in iter_json_files(MEMGOV_PATHS):
        for _, obj in iter_records(f) or []:
            records += 1
            ref.update(shingles(flatten(obj)))

    return ref, records


def main():
    ref_shingles, ref_records = build_reference_shingles()

    qa_rows = []

    for f in iter_json_files([RAW_PATH]):
        for rid, obj in iter_records(f) or []:
            for qpath, qa in recursive_find_qa(obj, rid):
                text = flatten(qa)
                sh = shingles(text)
                shared = len(sh & ref_shingles)
                ratio = shared / max(1, len(sh))

                row = {
                    "benchmark": "LoCoMo",
                    "record_id": qpath,
                    "source_file": str(f),
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "category": qa.get("category"),
                    "evidence": qa.get("evidence"),
                    "object": qa,
                    "overlap_ratio": ratio,
                    "shared_shingles": shared,
                    "total_shingles": len(sh),
                }

                is_seen = ratio >= THRESHOLD and shared >= MIN_SHARED_SHINGLES
                row["split"] = "source_overlap_or_seen" if is_seen else "audited_nonoverlap_subset"
                qa_rows.append(row)

    seen = [r for r in qa_rows if r["split"] == "source_overlap_or_seen"]
    non = [r for r in qa_rows if r["split"] == "audited_nonoverlap_subset"]

    with OUT_SEEN.open("w", encoding="utf-8") as f:
        for r in seen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with OUT_NON.open("w", encoding="utf-8") as f:
        for r in non:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": "LoCoMo",
        "qa_records_total": len(qa_rows),
        "qa_source_overlap_or_seen": len(seen),
        "qa_audited_nonoverlap": len(non),
        "memorygov_reference_records": ref_records,
        "reference_unique_shingles": len(ref_shingles),
        "threshold": THRESHOLD,
        "min_shared_shingles": MIN_SHARED_SHINGLES,
        "nonoverlap_qa_path": str(OUT_NON),
        "seen_qa_path": str(OUT_SEEN),
        "note": "QA-specific recursive extraction from raw LoCoMo. Still source-family non-independent because LoCoMo contributed to MemoryGovBench construction.",
    }

    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nSample audited non-overlap QA:")
    for r in non[:3]:
        print("=" * 100)
        print("record_id:", r["record_id"])
        print("question:", r["question"])
        print("answer:", r["answer"])
        print("category:", r["category"])
        print("evidence:", r["evidence"])


if __name__ == "__main__":
    main()
