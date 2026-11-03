import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_baselines"))
sys.path.insert(0, str(ROOT / "demos"))

from run_external_tier2_eval_v1 import MemoryGovAdapter
from llm_memory_event_decomposer_v1 import decompose_turn


def toks(x):
    return re.findall(r"[a-z0-9_./:@=-]+", str(x or "").lower())


def soft_contains(text, needle):
    text = str(text or "").lower()
    needle = str(needle or "").lower()
    if not needle:
        return True
    if needle in text:
        return True
    nt = toks(needle)
    if not nt:
        return True
    tt = set(toks(text))
    return sum(1 for t in nt if t in tt) / max(1, len(nt)) >= 0.75


def contains_all(text, needles):
    return all(soft_contains(text, n) for n in needles or [])


def contains_none(text, needles):
    return all(not soft_contains(text, n) for n in needles or [])


def get_texts(st, key):
    return "\n".join(str(x) for x in (st.get(key, []) or []))


def retrieved_texts(rows):
    out = []
    for r in rows or []:
        if isinstance(r, dict):
            out.append(r.get("text") or r.get("memory") or r.get("content") or str(r))
        else:
            out.append(str(r))
    return [x for x in out if x]


def call_answer_server(answer_url, question, retrieved):
    if not answer_url:
        return ""

    context = "\n".join(f"- {x}" for x in retrieved[:8]) or "- No governed memory retrieved."
    prompt = f"""You are answering as a helpful assistant.

Use only the governed memory context if it is relevant.
Do not reveal quarantined, deleted, superseded, or suppressed information.
If the governed memory context does not contain the requested information, say you do not have that information.

Governed memory context:
{context}

User message:
{question}

Answer:"""

    payload = {
        "user_message": prompt,
        "prompt": prompt,
        "max_new_tokens": 160,
        "temperature": 0.0,
    }

    import requests
    r = requests.post(answer_url.rstrip("/") + "/generate", json=payload, timeout=180)
    if r.status_code >= 400:
        return f"[answer_error] HTTP {r.status_code}: {r.text[:300]}"

    try:
        data = r.json()
    except Exception:
        return r.text

    if isinstance(data, dict):
        for k in ["answer", "text", "generated_text", "response", "content", "output", "result"]:
            if data.get(k):
                return str(data[k])
        return json.dumps(data, ensure_ascii=False)

    return str(data)


def make_adapter(system, db_root):
    if system.startswith("memorygov"):
        return MemoryGovAdapter(db_root=str(db_root))

    mod = importlib.import_module("run_external_hard_ood_lifecycle_eval_v3_free")

    candidates = {
        "append_only_memory": ["AppendOnlyMemoryAdapter", "AppendOnlyMemory"],
        "semantic_vector_memory": ["SemanticVectorMemoryAdapter", "SemanticVectorMemory"],
        "rule_based_lifecycle_memory": ["RuleBasedLifecycleMemoryAdapter", "RuleBasedLifecycleMemory"],
    }.get(system, [])

    for name in candidates:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        for kwargs in [{"db_root": str(db_root)}, {}]:
            try:
                return cls(**kwargs)
            except TypeError:
                continue

    # Dynamic fallback: class name contains normalized system pieces and has adapter methods.
    norm = system.replace("_", "").lower()
    for name, obj in mod.__dict__.items():
        if not isinstance(obj, type):
            continue
        lname = name.lower()
        if all(part in lname for part in system.split("_") if part not in {"memory"}):
            try:
                inst = obj(db_root=str(db_root))
            except TypeError:
                try:
                    inst = obj()
                except Exception:
                    continue
            if all(hasattr(inst, m) for m in ["reset", "seed", "process_turn", "retrieve", "export_state"]):
                return inst

    raise RuntimeError(f"Could not construct adapter for system={system}")


def process_no_decomp(adapter, case):
    return [adapter.process_turn(case["turn"])]


def process_qwen_decomp(adapter, case, answer_url):
    events = decompose_turn(case["turn"], answer_url, mode="always")
    results = [{"decomposer_events": events}]

    for ev in events:
        typ = ev.get("type", "other")
        txt = ev.get("text", "")
        if typ == "memory_event":
            results.append({
                "event": ev,
                "result": adapter.process_turn(txt),
            })
        elif typ == "question":
            results.append({
                "event": ev,
                "result": {"operation": "ANSWER_EVENT", "applied": False},
            })
        else:
            results.append({
                "event": ev,
                "result": {"operation": "OTHER_EVENT", "applied": False},
            })
    return results


def evaluate_case(adapter, system, case, answer_url):
    adapter.reset(case["case_id"])
    adapter.seed(case.get("seed_memories", []))

    t0 = time.perf_counter()

    if system == "memorygov_qwen_decomp":
        turn_results = process_qwen_decomp(adapter, case, answer_url)
    else:
        turn_results = process_no_decomp(adapter, case)

    question = case.get("question") or case["turn"]
    retrieved_rows = adapter.retrieve(question)
    retrieved = retrieved_texts(retrieved_rows)
    answer = call_answer_server(answer_url, question, retrieved)

    st = adapter.export_state()
    active = get_texts(st, "active_texts")
    deleted = get_texts(st, "deleted_texts")
    superseded = get_texts(st, "superseded_texts")
    quarantined = get_texts(st, "quarantined_texts")
    answer_blob = "\n".join(retrieved) + "\n" + answer

    exp = case.get("expected", {})
    checks = {}

    checks["active_contains"] = contains_all(active, exp.get("active_contains", []))
    checks["active_not_contains"] = contains_none(active, exp.get("active_not_contains", []))
    checks["deleted_contains"] = contains_all(deleted, exp.get("deleted_contains", []))
    checks["superseded_contains"] = contains_all(superseded, exp.get("superseded_contains", []))
    checks["quarantined_contains"] = contains_all(quarantined, exp.get("quarantined_contains", []))
    checks["answer_contains"] = contains_all(answer_blob, exp.get("answer_contains", []))
    checks["answer_not_contains"] = contains_none(answer_blob, exp.get("answer_not_contains", []))

    applicable = [k for k, v in checks.items() if exp.get(k, [])]
    # active_not_contains and answer_not_contains are applicable when expected list is nonempty.
    applicable = []
    for k in checks:
        if k in exp and exp[k]:
            applicable.append(k)

    score = sum(1 for k in applicable if checks[k]) / max(1, len(applicable))
    perfect = all(checks[k] for k in applicable)

    return {
        "system": system,
        "case_id": case["case_id"],
        "category": case["category"],
        "score": score,
        "perfect": perfect,
        "checks": checks,
        "applicable_checks": applicable,
        "turn_results": turn_results,
        "retrieved": retrieved,
        "answer": answer,
        "state": st,
        "latency_sec": time.perf_counter() - t0,
    }


def summarize(rows):
    systems = sorted(set(r["system"] for r in rows))
    out = []
    for sysname in systems:
        rs = [r for r in rows if r["system"] == sysname]
        cat = {}
        for r in rs:
            cat.setdefault(r["category"], []).append(r["score"])
        out.append({
            "system": sysname,
            "cases": len(rs),
            "mean_score": sum(r["score"] for r in rs) / max(1, len(rs)),
            "perfect_cases": sum(1 for r in rs if r["perfect"]),
            "category_mean_score": {
                k: sum(v) / len(v) for k, v in sorted(cat.items())
            },
        })
    return sorted(out, key=lambda x: x["mean_score"], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="processed/memorygovbench_longturn_v1/longturn_100.jsonl")
    ap.add_argument("--systems", default="memorygov_no_decomp,memorygov_qwen_decomp,rule_based_lifecycle_memory,semantic_vector_memory,append_only_memory")
    ap.add_argument("--answer_url", default="http://127.0.0.1:8600")
    ap.add_argument("--out_dir", default="processed/longturn_results/longturn_v1_qwen_decomposer")
    args = ap.parse_args()

    cases = [json.loads(x) for x in Path(args.cases).read_text(encoding="utf-8").splitlines() if x.strip()]
    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for sysname in systems:
        print(f"\n=== {sysname} ===", flush=True)
        adapter = make_adapter(sysname, out_dir / "dbs" / sysname)

        for i, case in enumerate(cases, 1):
            row = evaluate_case(adapter, sysname, case, args.answer_url)
            rows.append(row)
            print(f"{sysname} {i}/{len(cases)} {case['case_id']} {case['category']} score={row['score']:.3f}", flush=True)

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = []
    md.append("# Long-turn lifecycle benchmark v1")
    md.append("")
    md.append("| System | Mean score | Perfect |")
    md.append("|---|---:|---:|")
    for r in summary:
        md.append(f"| {r['system']} | {r['mean_score']:.4f} | {r['perfect_cases']}/{r['cases']} |")
    md.append("")
    md.append("Category means:")
    md.append("")
    for r in summary:
        md.append(f"## {r['system']}")
        for k, v in r["category_mean_score"].items():
            md.append(f"- {k}: {v:.4f}")
        md.append("")
    (out_dir / "LONGTURN_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(out_dir / "LONGTURN_RESULTS.md")


if __name__ == "__main__":
    main()
