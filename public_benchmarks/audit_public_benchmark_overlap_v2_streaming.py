import argparse
import hashlib
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

def norm(s):
    return " ".join(str(s or "").lower().split())

def tokens(s):
    return re.findall(r"[a-z0-9_./:@-]+", norm(s))

def h64(x):
    return hashlib.blake2b(repr(x).encode("utf-8", errors="ignore"), digest_size=8).hexdigest()

def shingles(s, n=5):
    t = tokens(s)
    if len(t) < n:
        return {h64(x) for x in t}
    return {h64(tuple(t[i:i+n])) for i in range(len(t)-n+1)}

def flatten(obj):
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in {"embedding", "vector"}:
                continue
            parts.append(str(k))
            parts.append(flatten(v))
    elif isinstance(obj, list):
        for x in obj:
            parts.append(flatten(x))
    else:
        parts.append(str(obj))
    return " ".join(parts)

def iter_json_records(path):
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

def iter_paths(paths):
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.is_dir():
            files = list(p.rglob("*.json")) + list(p.rglob("*.jsonl"))
        else:
            files = [p]
        for f in files:
            if f.exists() and f.suffix in {".json", ".jsonl"}:
                yield f

def build_reference_index(paths):
    shingle_counts = Counter()
    shingle_example = {}

    records = 0
    for f in iter_paths(paths):
        for rid, obj in iter_json_records(f) or []:
            records += 1
            text = flatten(obj)
            for sh in shingles(text):
                shingle_counts[sh] += 1
                if sh not in shingle_example:
                    shingle_example[sh] = rid

    return shingle_counts, shingle_example, records

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

    shingle_counts, shingle_example, ref_records = build_reference_index(args.memorygov_paths)

    seen_path = out_dir / f"{args.benchmark_name}_source_overlap_or_seen.jsonl"
    rest_path = out_dir / f"{args.benchmark_name}_audited_nonoverlap_subset.jsonl"

    seen_n = 0
    rest_n = 0
    bench_n = 0

    with seen_path.open("w", encoding="utf-8") as seen_f, rest_path.open("w", encoding="utf-8") as rest_f:
        for f in iter_paths(args.benchmark_paths):
            for rid, obj in iter_json_records(f) or []:
                bench_n += 1
                text = flatten(obj)
                sh = shingles(text)

                shared = [x for x in sh if x in shingle_counts]
                shared_count = len(shared)
                ratio = shared_count / max(1, len(sh))

                best_ref = None
                if shared:
                    # Not exact best record, but a concrete reference record sharing at least one shingle.
                    best_ref = shingle_example.get(shared[0])

                row = {
                    "benchmark": args.benchmark_name,
                    "record_id": rid,
                    "source_file": str(f),
                    "object": obj,
                    "split": None,
                    "overlap_ratio": ratio,
                    "shared_shingles": shared_count,
                    "total_shingles": len(sh),
                    "example_memorygov_record_id": best_ref,
                }

                is_seen = ratio >= args.threshold and shared_count >= args.min_shared_shingles
                if is_seen:
                    row["split"] = "source_overlap_or_seen"
                    seen_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    seen_n += 1
                else:
                    row["split"] = "audited_nonoverlap_subset"
                    rest_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rest_n += 1

    summary = {
        "benchmark": args.benchmark_name,
        "benchmark_records": bench_n,
        "memorygov_reference_records": ref_records,
        "source_overlap_or_seen": seen_n,
        "audited_nonoverlap_subset": rest_n,
        "reference_unique_shingles": len(shingle_counts),
        "threshold": args.threshold,
        "min_shared_shingles": args.min_shared_shingles,
        "benchmark_paths": args.benchmark_paths,
        "memorygov_paths": args.memorygov_paths,
        "note": "Streaming hash-shingle overlap audit. This is a conservative text-overlap audit, not proof of complete independence."
    }

    (out_dir / f"{args.benchmark_name}_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
