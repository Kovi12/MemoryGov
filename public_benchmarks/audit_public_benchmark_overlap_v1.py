import argparse
import json
import re
from pathlib import Path
from collections import Counter

def norm(s):
    return " ".join(str(s or "").lower().split())

def tokens(s):
    return re.findall(r"[a-z0-9_./:@-]+", norm(s))

def shingles(s, n=5):
    t = tokens(s)
    if len(t) < n:
        return set(t)
    return set(tuple(t[i:i+n]) for i in range(len(t)-n+1))

def flatten(obj):
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in {"embedding", "vector"}:
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
                obj = json.loads(line)
            except Exception:
                continue
            yield f"{path}:{i}", obj
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
            # If the JSON has top-level lists, expose each item separately.
            emitted = False
            for k, v in obj.items():
                if isinstance(v, list):
                    for i, x in enumerate(v):
                        yield f"{path}:{k}:{i}", x
                        emitted = True
            if not emitted:
                yield str(path), obj

def collect_records(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = []
            files.extend(p.rglob("*.json"))
            files.extend(p.rglob("*.jsonl"))
        else:
            files = [p]

        for f in files:
            for rid, obj in iter_json_records(f) or []:
                text = flatten(obj)
                rows.append({
                    "record_id": rid,
                    "source_file": str(f),
                    "text": text,
                    "object": obj,
                })
    return rows

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

    bench = collect_records(args.benchmark_paths)
    memgov = collect_records(args.memorygov_paths)

    memgov_shingles = []
    for r in memgov:
        sh = shingles(r["text"])
        if sh:
            memgov_shingles.append((r["record_id"], sh))

    seen = []
    rest = []

    for r in bench:
        sh = shingles(r["text"])
        best = {
            "best_overlap_ratio": 0.0,
            "best_shared_shingles": 0,
            "best_memorygov_record_id": None,
        }

        if sh:
            for mid, msh in memgov_shingles:
                shared = len(sh & msh)
                ratio = shared / max(1, len(sh))
                if ratio > best["best_overlap_ratio"]:
                    best = {
                        "best_overlap_ratio": ratio,
                        "best_shared_shingles": shared,
                        "best_memorygov_record_id": mid,
                    }

        row = {
            "benchmark": args.benchmark_name,
            "record_id": r["record_id"],
            "source_file": r["source_file"],
            "text": r["text"],
            "object": r["object"],
            **best,
        }

        is_seen = (
            best["best_overlap_ratio"] >= args.threshold
            and best["best_shared_shingles"] >= args.min_shared_shingles
        )

        if is_seen:
            row["split"] = "source_overlap_or_seen"
            seen.append(row)
        else:
            row["split"] = "audited_nonoverlap_subset"
            rest.append(row)

    def write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / f"{args.benchmark_name}_source_overlap_or_seen.jsonl", seen)
    write_jsonl(out_dir / f"{args.benchmark_name}_audited_nonoverlap_subset.jsonl", rest)

    summary = {
        "benchmark": args.benchmark_name,
        "benchmark_records": len(bench),
        "memorygov_reference_records": len(memgov),
        "source_overlap_or_seen": len(seen),
        "audited_nonoverlap_subset": len(rest),
        "threshold": args.threshold,
        "min_shared_shingles": args.min_shared_shingles,
        "benchmark_paths": args.benchmark_paths,
        "memorygov_paths": args.memorygov_paths,
        "note": "Text-shingle overlap audit. Non-overlap subset is suitable only for partial/sanity evaluation, not a proof of complete independence.",
    }

    (out_dir / f"{args.benchmark_name}_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
