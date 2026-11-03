#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import os
from collections import Counter
from pathlib import Path

LABEL_KEYS = {
    "label", "gold_label", "gold", "decision", "primary_decision",
    "operation", "target", "class", "retrieval_label", "gold_decision"
}
META_KEYS = {"task_level", "task", "level", "split", "source", "dataset", "origin"}
LIFECYCLE = {"WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"}
RETRIEVAL = {"RETRIEVE", "SUPPRESS"}
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "wandb", "runs", "checkpoints"}

def ext_full(p):
    n = p.name.lower()
    if n.endswith(".jsonl.gz"):
        return ".jsonl.gz"
    if n.endswith(".json.gz"):
        return ".json.gz"
    return p.suffix.lower()

def rel(p, root):
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)

def size_fmt(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}" if u != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"

def walk_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            yield Path(dp) / fn

def read_lines(p, max_lines):
    opener = gzip.open if p.name.lower().endswith(".gz") else open
    try:
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                yield line.rstrip("\n")
    except Exception:
        return

def iter_json_records(p, max_records=300):
    e = ext_full(p)
    if e in {".jsonl", ".jsonl.gz"}:
        count = 0
        for line in read_lines(p, max_records * 5):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
                count += 1
                if count >= max_records:
                    return
        return

    if e in {".json", ".json.gz"}:
        text = "\n".join(read_lines(p, 10000))
        if not text.strip():
            return
        try:
            data = json.loads(text)
        except Exception:
            return

        if isinstance(data, list):
            for item in data[:max_records]:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            for k in ["data", "examples", "items", "records", "train", "validation", "test"]:
                if isinstance(data.get(k), list):
                    for item in data[k][:max_records]:
                        if isinstance(item, dict):
                            yield item
                    return
            yield data

def flatten_keys(x, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return []
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.append(kk)
            out.extend(flatten_keys(v, kk, depth + 1, max_depth))
    elif isinstance(x, list) and x:
        out.extend(flatten_keys(x[0], prefix + "[]", depth + 1, max_depth))
    return out

def find_values(x, wanted):
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k) in wanted:
                out.append(v)
            out.extend(find_values(v, wanted))
    elif isinstance(x, list):
        for item in x[:30]:
            out.extend(find_values(item, wanted))
    return out

def norm_label(v):
    if v is None:
        return None
    if isinstance(v, dict):
        for k in ["primary_decision", "decision", "label", "gold_label", "operation", "target", "class"]:
            if k in v:
                return norm_label(v[k])
        return None
    if isinstance(v, list):
        return None
    s = str(v).strip().upper()
    return s or None

def role_guess(path, labels):
    low = str(path).lower()
    labs = set(labels)
    if labs & RETRIEVAL or "retriev" in low or "suppress" in low:
        return "retrieval_governance"
    if labs & LIFECYCLE or "operation" in low or "lifecycle" in low or "governance" in low:
        return "operation_governance"
    if any(x in low for x in ["locomo", "longmemeval", "memoryagentbench", "agentbench", "benchmark"]):
        return "source_benchmark"
    if any(x in low for x in ["eval", "result", "metric", "score", "baseline", "ood"]):
        return "evaluation_results"
    return "other_data"

def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def make_tree(root, max_depth=6, max_entries=80):
    lines = [str(root)]
    def rec(d, pref="", depth=1):
        if depth > max_depth:
            return
        try:
            entries = sorted([x for x in d.iterdir() if x.name not in SKIP_DIRS], key=lambda x: (x.is_file(), x.name.lower()))
        except Exception as e:
            lines.append(pref + f"[cannot read: {e}]")
            return
        shown = entries[:max_entries]
        for i, e in enumerate(shown):
            last = i == len(shown) - 1 and len(entries) <= max_entries
            conn = "└── " if last else "├── "
            if e.is_dir():
                lines.append(pref + conn + e.name + "/")
                rec(e, pref + ("    " if last else "│   "), depth + 1)
            else:
                try:
                    sz = size_fmt(e.stat().st_size)
                except Exception:
                    sz = "?"
                lines.append(pref + conn + f"{e.name} [{sz}]")
        if len(entries) > max_entries:
            lines.append(pref + f"└── ... {len(entries) - max_entries} more entries")
    rec(root)
    return "\n".join(lines)

def inspect_json(p, root, max_records):
    keys = Counter()
    labels = Counter()
    meta = Counter()
    sample = None
    count = 0

    for rec in iter_json_records(p, max_records):
        sample = sample or rec
        count += 1
        keys.update(flatten_keys(rec))
        for v in find_values(rec, LABEL_KEYS):
            lab = norm_label(v)
            if lab:
                labels[lab] += 1
        for v in find_values(rec, META_KEYS):
            if isinstance(v, (str, int, float)):
                meta[str(v)[:120]] += 1

    return {
        "path": rel(p, root),
        "size": size_fmt(p.stat().st_size),
        "records": count,
        "keys": keys,
        "labels": labels,
        "meta": meta,
        "role": role_guess(p, labels),
        "sample": sample
    }

def inspect_csv(p, root):
    header = []
    rows = 0
    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f, delimiter=sep)
            for i, row in enumerate(r):
                if i == 0:
                    header = row
                elif i < 50:
                    rows += 1
                else:
                    break
    except Exception:
        pass
    low_cols = " ".join(c.lower() for c in header)
    labels = Counter()
    if "retrieve" in low_cols or "suppress" in low_cols:
        labels["RETRIEVE_OR_SUPPRESS_COLUMN"] += 1
    return {
        "path": rel(p, root),
        "size": size_fmt(p.stat().st_size),
        "records": rows,
        "keys": Counter({c: 1 for c in header}),
        "labels": labels,
        "meta": Counter(),
        "role": role_guess(p, labels),
        "sample": None
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="processed")
    ap.add_argument("--out", default="_inspection_processed_outputs")
    ap.add_argument("--max-json-records", type=int, default=500)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    (out / "00_project_tree.txt").write_text(make_tree(root), encoding="utf-8")

    inventory = []
    reports = []
    exts = Counter()

    for p in walk_files(root):
        e = ext_full(p)
        exts[e or "<no_ext>"] += 1
        try:
            st = p.stat()
        except Exception:
            continue

        rep = None
        if e in {".json", ".jsonl", ".json.gz", ".jsonl.gz"}:
            rep = inspect_json(p, root, args.max_json_records)
        elif e in {".csv", ".tsv"}:
            rep = inspect_csv(p, root)

        role = rep["role"] if rep else role_guess(p, Counter())
        inventory.append({
            "path": rel(p, root),
            "extension": e,
            "size": size_fmt(st.st_size),
            "size_bytes": st.st_size,
            "role_guess": role,
        })
        if rep:
            reports.append(rep)

    write_csv(out / "01_inventory.csv", inventory, ["path", "extension", "size", "size_bytes", "role_guess"])
    write_csv(out / "02_inventory_by_extension.csv", [{"extension": k, "count": v} for k, v in exts.most_common()], ["extension", "count"])

    candidates = []
    labels_rows = []
    for r in reports:
        candidates.append({
            "path": r["path"],
            "role_guess": r["role"],
            "size": r["size"],
            "sampled_records": r["records"],
            "labels": "; ".join(f"{k}:{v}" for k, v in r["labels"].most_common(20)),
            "meta": "; ".join(f"{k}:{v}" for k, v in r["meta"].most_common(15)),
            "top_keys": "; ".join(k for k, _ in r["keys"].most_common(30)),
        })
        for lab, cnt in r["labels"].most_common():
            labels_rows.append({
                "path": r["path"],
                "role_guess": r["role"],
                "label": lab,
                "sample_count": cnt,
                "sampled_records": r["records"],
            })

    write_csv(out / "03_dataset_candidates.csv", candidates, ["path", "role_guess", "size", "sampled_records", "labels", "meta", "top_keys"])
    write_csv(out / "04_label_summary.csv", labels_rows, ["path", "role_guess", "label", "sample_count", "sampled_records"])

    md = []
    md.append(f"# Processed schema report\n\nRoot: `{root}`\n\n")
    for r in sorted(reports, key=lambda x: (x["role"], x["path"])):
        md.append(f"\n## `{r['path']}`\n\n")
        md.append(f"- Role guess: **{r['role']}**\n")
        md.append(f"- Size: `{r['size']}`\n")
        md.append(f"- Sampled records/rows: `{r['records']}`\n")
        if r["labels"]:
            md.append("- Labels: " + ", ".join(f"`{k}`={v}" for k, v in r["labels"].most_common(20)) + "\n")
        if r["meta"]:
            md.append("- Meta/source/split fields: " + ", ".join(f"`{k}`={v}" for k, v in r["meta"].most_common(15)) + "\n")
        md.append("- Top keys/columns:\n")
        for k, v in r["keys"].most_common(35):
            md.append(f"  - `{k}` ({v})\n")
        if r["sample"] is not None:
            s = json.dumps(r["sample"], ensure_ascii=False, indent=2)
            if len(s) > 5000:
                s = s[:5000] + "\n..."
            md.append("\nSample:\n\n```json\n" + s + "\n```\n")

    (out / "05_schema_report.md").write_text("".join(md), encoding="utf-8")

    print("[OK] Inspection complete")
    print(f"Root: {root}")
    print(f"Output: {out}")
    print("\nMain files:")
    for name in [
        "00_project_tree.txt",
        "01_inventory.csv",
        "03_dataset_candidates.csv",
        "04_label_summary.csv",
        "05_schema_report.md",
    ]:
        print(f"  {out / name}")

    print("\nPreview:")
    print(f"  sed -n '1,180p' {out}/00_project_tree.txt")
    print(f"  column -s, -t < {out}/03_dataset_candidates.csv | sed -n '1,120p'")
    print(f"  column -s, -t < {out}/04_label_summary.csv | sed -n '1,160p'")
    print(f"  sed -n '1,260p' {out}/05_schema_report.md")

if __name__ == "__main__":
    main()
