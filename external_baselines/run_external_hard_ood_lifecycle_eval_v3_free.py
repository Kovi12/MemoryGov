import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_external_hard_ood_lifecycle_eval_v2 import (
    DEFAULT_CASES,
    evaluate_case,
    summarize,
)
from run_external_tier2_eval_v1 import (
    MemoryGovAdapter,
    CapabilityOnlyAdapter,
    load_jsonl,
    write_jsonl,
)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

DEFAULT_OUT_DIR = Path("processed/external_baseline_results/hard_ood_v2b_lifecycle_v3_free")

STOP = {
    "the", "and", "for", "with", "that", "this", "user", "memory", "candidate",
    "active", "current", "previous", "previously", "stored", "message", "from",
    "about", "into", "their", "there", "should", "would", "could", "when",
    "what", "have", "has", "had", "are", "was", "were", "will", "not", "now",
    "please", "remember", "forget", "delete", "remove", "update", "change",
}

SENSITIVE_PATTERNS = [
    r"\bapi[_ -]?key\b",
    r"\bsecret\b",
    r"\bpassword\b",
    r"\bpasscode\b",
    r"\btoken\b",
    r"\bprivate[_ -]?key\b",
    r"\bpassport\b",
    r"\bsocial security\b",
    r"\bssn\b",
    r"\bcredit card\b",
    r"\bcard number\b",
    r"\bbank account\b",
    r"\biban\b",
    r"\b\d{12,19}\b",
]


def norm(x):
    return " ".join(str(x or "").lower().split())


def toks(x):
    raw = re.findall(r"[a-z0-9_./:@-]+", norm(x))
    return [t for t in raw if len(t) >= 3 and t not in STOP]


def overlap_score(a, b):
    ta = set(toks(a))
    tb = set(toks(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def lexical_score(query, text):
    q = toks(query)
    if not q:
        return 0.0
    tf = Counter(toks(text))
    score = 0.0
    for t in q:
        if t in tf:
            score += 1.0 + math.log(1 + tf[t])
    return score / max(1, len(set(q)))


def is_sensitive(text):
    s = str(text or "").lower()
    return any(re.search(p, s, flags=re.IGNORECASE) for p in SENSITIVE_PATTERNS)


def looks_forget(text):
    s = norm(text)
    return any(x in s for x in [
        "forget",
        "delete",
        "remove",
        "erase",
        "do not remember",
        "don't remember",
        "stop remembering",
        "no longer remember",
    ])


def looks_ignore(text):
    s = norm(text)
    return any(x in s for x in [
        "temporary",
        "just for this session",
        "do not store",
        "don't store",
        "do not save",
        "don't save",
        "ignore this",
        "not important",
    ])


def looks_update(text):
    s = norm(text)
    return any(x in s for x in [
        "update",
        "changed",
        "change",
        "instead",
        "actually",
        "correction",
        "correct",
        "no longer",
        "not anymore",
        "active path is",
        "current path is",
        "now it is",
        "now my",
    ])


def extract_candidate(turn):
    text = str(turn or "").strip()
    if not text:
        return ""

    patterns = [
        r"candidate_memory\s*[:=]\s*(.+)",
        r"candidate memory\s*[:=]\s*(.+)",
        r"candidate\s*[:=]\s*(.+)",
        r"new memory\s*[:=]\s*(.+)",
        r"memory candidate\s*[:=]\s*(.+)",
        r"store\s*[:=]\s*(.+)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"\n\s*(existing_memory|existing memory|expected|label|operation)\s*[:=]", cand, flags=re.IGNORECASE)[0].strip()
            return cand or text

    # For baselines, storing the raw user turn is fair: naive memories often store raw events.
    return text


class LocalMemoryAdapter:
    name = "local_base"

    def __init__(self, retrieve_mode="lexical"):
        self.retrieve_mode = retrieve_mode
        self.reset("init")

    def reset(self, case_id):
        self.case_id = str(case_id)
        self.active = []
        self.deleted = []
        self.quarantined = []
        self.superseded = []
        self.turn_results = []

    def seed(self, memories):
        for m in memories or []:
            if str(m).strip():
                self.active.append(str(m).strip())

    def process_turn(self, turn):
        cand = extract_candidate(turn)
        if cand:
            self.active.append(cand)
        result = {
            "operation": "WRITE_APPEND",
            "candidate": cand,
            "applied": bool(cand),
            "note": "Naive local memory baseline.",
        }
        self.turn_results.append(result)
        return result

    def retrieve(self, query):
        rows = []
        for i, text in enumerate(self.active):
            if self.retrieve_mode == "recency":
                score = i / max(1, len(self.active))
            elif self.retrieve_mode == "hybrid_recency":
                score = lexical_score(query, text) + 0.10 * (i / max(1, len(self.active)))
            else:
                score = lexical_score(query, text)
            rows.append((score, i, text))

        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [{"text": t, "score": s} for s, _, t in rows[:5]]

    def export_state(self):
        return {
            "active_texts": list(self.active),
            "deleted_texts": list(self.deleted),
            "quarantined_texts": list(self.quarantined),
            "superseded_texts": list(self.superseded),
            "notes": {
                "baseline": self.name,
                "retrieve_mode": self.retrieve_mode,
            },
        }

    def close(self):
        pass


class AppendOnlyMemoryAdapter(LocalMemoryAdapter):
    name = "append_only_memory"

    def __init__(self):
        super().__init__(retrieve_mode="recency")


class LexicalTfidfMemoryAdapter(LocalMemoryAdapter):
    name = "lexical_tfidf_memory"

    def __init__(self):
        super().__init__(retrieve_mode="lexical")


class RecencyMemoryAdapter(LocalMemoryAdapter):
    name = "recency_memory"

    def __init__(self):
        super().__init__(retrieve_mode="hybrid_recency")


class LastWriteWinsMemoryAdapter(LocalMemoryAdapter):
    name = "last_write_wins_memory"

    def __init__(self):
        super().__init__(retrieve_mode="lexical")

    def process_turn(self, turn):
        cand = extract_candidate(turn)
        result = {
            "operation": "WRITE",
            "candidate": cand,
            "applied": False,
            "note": "Last-write-wins heuristic baseline.",
        }

        if looks_forget(turn):
            kept = []
            moved = []
            for m in self.active:
                if overlap_score(m, turn) >= 0.25:
                    moved.append(m)
                else:
                    kept.append(m)
            self.active = kept
            self.deleted.extend(moved)
            result.update({
                "operation": "FORGET_HEURISTIC",
                "applied": bool(moved),
                "deleted_count": len(moved),
            })
            self.turn_results.append(result)
            return result

        if cand and looks_update(turn):
            kept = []
            moved = []
            for m in self.active:
                if overlap_score(m, turn) >= 0.25 or overlap_score(m, cand) >= 0.25:
                    moved.append(m)
                else:
                    kept.append(m)
            self.active = kept
            self.superseded.extend(moved)
            self.active.append(cand)
            result.update({
                "operation": "UPDATE_HEURISTIC",
                "applied": True,
                "superseded_count": len(moved),
            })
            self.turn_results.append(result)
            return result

        if cand:
            self.active.append(cand)
            result.update({
                "operation": "WRITE_APPEND",
                "applied": True,
            })

        self.turn_results.append(result)
        return result


class SemanticVectorMemoryAdapter(LocalMemoryAdapter):
    name = "semantic_vector_memory"
    _model = None
    _model_error = None

    def __init__(self):
        super().__init__(retrieve_mode="semantic")
        self.embeddings = []
        self._ensure_model()

    @classmethod
    def _ensure_model(cls):
        if cls._model is not None or cls._model_error is not None:
            return
        if SentenceTransformer is None:
            cls._model_error = "sentence_transformers_not_available"
            return
        try:
            cls._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as exc:
            cls._model_error = str(exc)
            cls._model = None

    def reset(self, case_id):
        super().reset(case_id)
        self.embeddings = []

    def _embed(self, texts):
        self._ensure_model()
        if self._model is None:
            return None
        return self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _rebuild_embeddings(self):
        if not self.active:
            self.embeddings = []
            return
        emb = self._embed(self.active)
        self.embeddings = [] if emb is None else [x for x in emb]

    def seed(self, memories):
        self.active = []
        for m in memories or []:
            if str(m).strip():
                self.active.append(str(m).strip())
        self._rebuild_embeddings()

    def process_turn(self, turn):
        cand = extract_candidate(turn)
        if cand:
            self.active.append(cand)
            emb = self._embed([cand])
            if emb is not None:
                self.embeddings.append(emb[0])
            else:
                self._rebuild_embeddings()

        result = {
            "operation": "WRITE_APPEND",
            "candidate": cand,
            "applied": bool(cand),
            "note": "SBERT semantic vector memory baseline. Stores everything; no lifecycle governance.",
        }
        self.turn_results.append(result)
        return result

    def retrieve(self, query):
        self._ensure_model()

        # Fallback to lexical if the embedding model cannot load.
        if self._model is None:
            rows = super().retrieve(query)
            for r in rows:
                r["fallback"] = "lexical"
                r["model_error"] = self._model_error
            return rows

        if len(self.embeddings) != len(self.active):
            self._rebuild_embeddings()

        q = self._embed([query])
        if q is None or not self.embeddings:
            return []

        qv = q[0]
        rows = []
        for i, (text, emb) in enumerate(zip(self.active, self.embeddings)):
            score = float((qv * emb).sum())
            rows.append((score, i, text))

        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [{"text": t, "score": s, "retriever": "sbert_cosine"} for s, _, t in rows[:5]]

    def export_state(self):
        st = super().export_state()
        st["notes"]["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
        st["notes"]["model_error"] = self._model_error
        return st


class LangChainStyleVectorMemoryAdapter(SemanticVectorMemoryAdapter):
    name = "langchain_style_vector_memory"

    def export_state(self):
        st = super().export_state()
        st["notes"]["baseline"] = self.name
        st["notes"]["interpretation"] = (
            "Local LangChain-style vector memory: append all memories, retrieve by SBERT cosine. "
            "No Chroma/FAISS dependency and no lifecycle governance."
        )
        return st


class SummaryMemoryNaiveAdapter(LocalMemoryAdapter):
    name = "summary_memory_naive"

    def __init__(self):
        super().__init__(retrieve_mode="lexical")
        self.summary = ""

    def reset(self, case_id):
        super().reset(case_id)
        self.summary = ""

    def seed(self, memories):
        self.summary = " ".join(str(m).strip() for m in memories or [] if str(m).strip())
        self.active = [self.summary] if self.summary else []

    def process_turn(self, turn):
        cand = extract_candidate(turn)
        if cand:
            self.summary = (self.summary + " " + cand).strip()
            self.active = [self.summary]
        result = {
            "operation": "SUMMARY_APPEND",
            "candidate": cand,
            "applied": bool(cand),
            "note": "Naive summary-memory baseline; no true lifecycle control.",
        }
        self.turn_results.append(result)
        return result


class RuleBasedLifecycleMemoryAdapter(LocalMemoryAdapter):
    name = "rule_based_lifecycle_memory"

    def __init__(self):
        super().__init__(retrieve_mode="lexical")

    def _delete_matching(self, query, threshold=0.25):
        kept = []
        moved = []
        for m in self.active:
            if overlap_score(m, query) >= threshold:
                moved.append(m)
            else:
                kept.append(m)
        self.active = kept
        return moved

    def _supersede_matching(self, query, candidate, threshold=0.25):
        kept = []
        moved = []
        for m in self.active:
            if overlap_score(m, query) >= threshold or overlap_score(m, candidate) >= threshold:
                moved.append(m)
            else:
                kept.append(m)
        self.active = kept
        return moved

    def process_turn(self, turn):
        cand = extract_candidate(turn)
        result = {
            "operation": "WRITE",
            "candidate": cand,
            "applied": False,
            "note": "Rule-based lifecycle baseline using hand-written ignore/forget/update/quarantine heuristics.",
        }

        if looks_ignore(turn):
            result.update({
                "operation": "IGNORE_RULE",
                "applied": True,
            })
            self.turn_results.append(result)
            return result

        if looks_forget(turn):
            moved = self._delete_matching(turn)
            self.deleted.extend(moved)
            result.update({
                "operation": "FORGET_RULE",
                "applied": bool(moved),
                "deleted_count": len(moved),
            })
            self.turn_results.append(result)
            return result

        if cand and is_sensitive(cand):
            self.quarantined.append(cand)
            result.update({
                "operation": "QUARANTINE_RULE",
                "applied": True,
            })
            self.turn_results.append(result)
            return result

        if cand and looks_update(turn):
            moved = self._supersede_matching(turn, cand)
            self.superseded.extend(moved)
            self.active.append(cand)
            result.update({
                "operation": "UPDATE_RULE",
                "applied": True,
                "superseded_count": len(moved),
            })
            self.turn_results.append(result)
            return result

        if cand:
            self.active.append(cand)
            result.update({
                "operation": "WRITE_OR_MERGE_RULE",
                "applied": True,
            })

        self.turn_results.append(result)
        return result


class RegexSafetyMemoryAdapter(LocalMemoryAdapter):
    name = "regex_safety_memory"

    def __init__(self):
        super().__init__(retrieve_mode="lexical")

    def process_turn(self, turn):
        cand = extract_candidate(turn)
        result = {
            "operation": "WRITE",
            "candidate": cand,
            "applied": False,
            "note": "Keyword/regex policy baseline.",
        }

        if looks_ignore(turn):
            result.update({
                "operation": "IGNORE_REGEX",
                "applied": True,
            })
            self.turn_results.append(result)
            return result

        if looks_forget(turn):
            kept = []
            moved = []
            for m in self.active:
                if overlap_score(m, turn) >= 0.25:
                    moved.append(m)
                else:
                    kept.append(m)
            self.active = kept
            self.deleted.extend(moved)
            result.update({
                "operation": "FORGET_REGEX",
                "applied": bool(moved),
                "deleted_count": len(moved),
            })
            self.turn_results.append(result)
            return result

        if cand and is_sensitive(cand):
            self.quarantined.append(cand)
            result.update({
                "operation": "QUARANTINE_REGEX",
                "applied": True,
            })
            self.turn_results.append(result)
            return result

        if cand:
            self.active.append(cand)
            result.update({
                "operation": "WRITE_REGEX",
                "applied": True,
            })

        self.turn_results.append(result)
        return result


def build_free_adapters(system_names, out_dir, margin_threshold):
    adapters = []

    for name in system_names:
        if name == "memorygov":
            adapters.append(MemoryGovAdapter(
                db_root=Path(out_dir) / "memorygov_dbs",
                margin_threshold=margin_threshold,
            ))

        elif name == "append_only_memory":
            adapters.append(AppendOnlyMemoryAdapter())

        elif name == "lexical_tfidf_memory":
            adapters.append(LexicalTfidfMemoryAdapter())

        elif name == "last_write_wins_memory":
            adapters.append(LastWriteWinsMemoryAdapter())

        elif name == "recency_memory":
            adapters.append(RecencyMemoryAdapter())

        elif name == "semantic_vector_memory":
            adapters.append(SemanticVectorMemoryAdapter())

        elif name == "langchain_style_vector_memory":
            adapters.append(LangChainStyleVectorMemoryAdapter())

        elif name == "summary_memory_naive":
            adapters.append(SummaryMemoryNaiveAdapter())

        elif name == "regex_safety_memory":
            adapters.append(RegexSafetyMemoryAdapter())

        elif name == "rule_based_lifecycle_memory":
            adapters.append(RuleBasedLifecycleMemoryAdapter())

        elif name in {"langgraph_vector_memory", "letta_local", "graphiti_local", "cognee_local"}:
            adapters.append(CapabilityOnlyAdapter(
                name=name,
                capabilities=["planned_open_source_baseline"],
                note="Placeholder only. Not behaviorally scored until local integration is implemented.",
            ))

        else:
            raise ValueError(f"Unknown system: {name}")

    return adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--systems",
        default="append_only_memory,lexical_tfidf_memory,last_write_wins_memory,recency_memory,summary_memory_naive,regex_safety_memory",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--margin_threshold", type=float, default=0.10)
    parser.add_argument("--merge_existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    cases = load_jsonl(Path(args.cases))
    if args.limit:
        cases = cases[:args.limit]

    adapters = build_free_adapters(systems, out_dir, args.margin_threshold)

    rows = []
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    if args.merge_existing and results_path.exists():
        old_rows = load_jsonl(results_path)
        rerun_systems = {a.name for a in adapters}
        rows.extend([r for r in old_rows if r.get("system") not in rerun_systems])

    for adapter in adapters:
        print(f"\n=== Running system: {adapter.name} ===")
        for i, case in enumerate(cases, start=1):
            row = evaluate_case(adapter, case)
            rows.append(row)
            print(
                f"{adapter.name} {i}/{len(cases)} "
                f"{case.get('case_id')} op={case.get('expected_operation')} score={row.get('score')}"
            )
        adapter.close()

    write_jsonl(results_path, rows)

    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
