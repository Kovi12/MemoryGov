import argparse
import hashlib
import json
import re
from pathlib import Path

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

def iter_files(paths):
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.is_dir():
            for f in sorted(list(p.rglob("*.json")) + list(p.rglob("*.jsonl"))):
                yield f
        elif p.suffix in {".json", ".jsonl"}:
            yield p

def iter_records(path):
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return

    if path.suffix == ".jsonl":
        for i, line in enumerate(txt.splitlines(), start=1):
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
            obj = json.loads(txt)
        except Exception:
            return

        if isinstance(obj, list):
            for i, x in enumerate(obj):
                yield f"{path}:{i}", x
        elif isinstance(obj, dict):
            emitted = False
            for k, v in obj.items():
                if isinstance(v, list):
                    for i, x in enumerate(v):
                        yield f"{path}:{k}:{i}", x
                        emitted = True
            if not emitted:
                yield str(path), obj

def build_reference_shingles(paths):
    ref = set()
    ref_records = 0

    for f in iter_files(paths):
        for _, obj in iter_records(f) or []:
            ref_records += 1
            ref.update(shingles(flatten(obj)))

    return ref, ref_records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_name", required=True)
    ap.add_argument("--benchmark_paths", nargs="+", required=True)
    ap.add_argument("--memorygov_paths", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--min_shared_shingles", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_shingles, ref_records = build_reference_shingles(args.memorygov_paths)

    seen_path = out_dir / f"{args.benchmark_name}_source_overlap_or_seen.jsonl"
    non_path = out_dir / f"{args.benchmark_name}_audited_nonoverlap_subset.jsonl"
    summary_path = out_dir / f"{args.benchmark_name}_overlap_summary.json"

    bench_records = 0
    seen_records = 0
    non_records = 0

    with seen_path.open("w", encoding="utf-8") as seen_f, non_path.open("w", encoding="utf-8") as non_f:
        for f in iter_files(args.benchmark_paths):
            for rid, obj in iter_records(f) or []:
                bench_records += 1
                sh = shingles(flatten(obj))
                shared = len(sh & ref_shingles)
                ratio = shared / max(1, len(sh))

                row = {
                    "benchmark": args.benchmark_name,
                    "record_id": rid,
                    "source_file": str(f),
                    "overlap_ratio": ratio,
                    "shared_shingles": shared,
                    "total_shingles": len(sh),
                    "object": obj,
                }

                is_seen = ratio >= args.threshold and shared >= args.min_shared_shingles
                if is_seen:
                    row["split"] = "source_overlap_or_seen"
                    seen_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    seen_records += 1
                else:
                    row["split"] = "audited_nonoverlap_subset"
                    non_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    non_records += 1

    summary = {
        "benchmark": args.benchmark_name,
        "benchmark_records": bench_records,
        "memorygov_reference_records": ref_records,
        "reference_unique_shingles": len(ref_shingles),
        "source_overlap_or_seen": seen_records,
        "audited_nonoverlap_subset": non_records,
        "threshold": args.threshold,
        "min_shared_shingles": args.min_shared_shingles,
        "benchmark_paths": args.benchmark_paths,
        "memorygov_paths": args.memorygov_paths,
        "note": "Low-memory hash-shingle audit. This separates obvious source overlap from an audited non-overlap subset, but it is not proof of complete independence."
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
