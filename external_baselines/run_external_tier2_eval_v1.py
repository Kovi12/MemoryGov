import argparse
import importlib.util
import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path


PROJECT_ROOT = Path(".")
CASES_PATH = Path("processed/external_baselines_tier2_v1/external_tier2_cases.jsonl")
OUT_DIR = Path("processed/external_baseline_results/tier2_v1")
MEMGOV_MODULE_PATH = Path("memory_governor_cli_v4_10.py")
MEMGOV_MODEL_PATH = Path("processed/operation_classifier_results/non_llm_v1_3_2_no_confirmation_semantic_selected/semantic_tfidf_sgd_logloss_balanced/model.joblib")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def contains_any(texts, needles):
    joined = "\n".join(x or "" for x in texts).lower()
    return any(n.lower() in joined for n in needles or [])


def contains_all(texts, needles):
    joined = "\n".join(x or "" for x in texts).lower()
    return all(n.lower() in joined for n in needles or [])


def redact(text):
    if text is None:
        return None
    return re.sub(r"\b\d{5,}\b", "[NUMERIC_ID_REDACTED]", text)


class BaseAdapter:
    name = "base"

    def reset(self, case_id):
        raise NotImplementedError

    def seed(self, memories):
        raise NotImplementedError

    def fallback_target_by_overlap(self, candidate):
        import re

        if not candidate:
            return None, "fallback_no_candidate"

        cand_tokens = set(
            t for t in re.findall(r"[A-Za-z0-9_]+", str(candidate).lower())
            if len(t) >= 4
        )

        rows = [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT memory_id, text, memory_type, sensitivity, status, supersedes_id
                FROM memories
                WHERE user_id = ? AND status = 'active'
                """,
                (self.user_id,),
            )
        ]

        best = None
        best_score = 0

        for row in rows:
            mem_tokens = set(
                t for t in re.findall(r"[A-Za-z0-9_]+", str(row.get("text", "")).lower())
                if len(t) >= 4
            )
            score = len(cand_tokens & mem_tokens)

            # Prefer exact project/entity overlap when available.
            if score > best_score:
                best = row
                best_score = score

        if best is not None and best_score > 0:
            return best, f"fallback_overlap_score_{best_score}"

        return None, "fallback_no_overlap"


    def process_turn(self, turn):
        raise NotImplementedError

    def retrieve(self, query):
        raise NotImplementedError

    def export_state(self):
        raise NotImplementedError

    def close(self):
        pass


class CapabilityOnlyAdapter(BaseAdapter):
    def __init__(self, name, capabilities, note):
        self.name = name
        self.capabilities = capabilities
        self.note = note

    def reset(self, case_id):
        self.case_id = case_id

    def seed(self, memories):
        pass

    def process_turn(self, turn):
        return {
            "operation": "UNSUPPORTED_ADAPTER",
            "applied": False,
            "note": self.note,
        }

    def retrieve(self, query):
        return []

    def export_state(self):
        return {
            "capabilities": self.capabilities,
            "note": self.note,
        }


class MemoryGovAdapter(BaseAdapter):
    name = "memorygov"

    def __init__(self, db_root, margin_threshold=0.10):
        self.db_root = Path(db_root)
        self.margin_threshold = margin_threshold
        self.mgov = self._load_mgov()
        self.model, self.encoder = self.mgov.load_model(str(MEMGOV_MODEL_PATH))
        self.conn = None
        self.user_id = "tier2_user"

        self.mgov.GLOBAL_RETRIEVAL_BACKEND = "hybrid"
        self.mgov.GLOBAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            self.mgov.GLOBAL_EMBEDDER = self.mgov.load_embedder(self.mgov.GLOBAL_EMBEDDING_MODEL)
            if self.mgov.GLOBAL_EMBEDDER is None:
                print("[memorygov] embedder unavailable; falling back to lexical retrieval")
                self.mgov.GLOBAL_RETRIEVAL_BACKEND = "lexical"
        except Exception as exc:
            print(f"[memorygov] embedder load failed: {exc}; falling back to lexical retrieval")
            self.mgov.GLOBAL_EMBEDDER = None
            self.mgov.GLOBAL_RETRIEVAL_BACKEND = "lexical"

    def _load_mgov(self):
        spec = importlib.util.spec_from_file_location("mgov", MEMGOV_MODULE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def reset(self, case_id):
        if self.conn:
            self.conn.close()

        db_path = self.db_root / f"{case_id}.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.mgov.init_db(self.conn)

    def seed(self, memories):
        for m in memories:
            self.mgov.insert_memory(
                self.conn,
                user_id=self.user_id,
                text=m,
                memory_type="general",
                sensitivity="low",
                status="active",
                supersedes_id=None,
                source="external_tier2_seed",
            )

    def process_turn(self, turn):
        t0 = time.perf_counter()

        answer_retrieved = self.mgov.retrieve_for_answer_context(self.conn, self.user_id, turn)
        candidate = self.mgov.extract_candidate_memory(turn)

        if not candidate:
            return {
                "operation": "SUPPRESS",
                "candidate": None,
                "applied": False,
                "retrieved": answer_retrieved,
                "latency_sec": time.perf_counter() - t0,
            }

        memory_type = self.mgov.infer_memory_type(turn)
        sensitivity = self.mgov.infer_sensitivity(turn)
        target_for_model = self.mgov.retrieve_for_target(self.conn, self.user_id, candidate)

        op_input = self.mgov.build_operation_input(
            turn,
            candidate,
            target_for_model,
            memory_type,
            sensitivity,
        )

        raw_label, scores = self.mgov.predict_operation(self.model, self.encoder, op_input)
        final_label, override_reason = self.mgov.policy_override(
            raw_label,
            turn,
            candidate,
            memory_type,
            sensitivity,
        )

        target_retrieved = self.mgov.refine_targets_for_final_action(
            self.conn,
            self.user_id,
            final_label,
            turn,
            candidate,
            memory_type,
            target_for_model,
        )

        margin = self.mgov.score_margin(scores)

        target = None
        target_reason = "not_targeted"
        if final_label in {"UPDATE", "MERGE", "FORGET"}:
            target, target_reason = self.mgov.choose_target_automatically(target_retrieved)

            if target is None:
                fallback_target, fallback_reason = self.fallback_target_by_overlap(candidate)
                if fallback_target is not None:
                    target = fallback_target
                    target_reason = fallback_reason

        # For offline benchmark, apply the model's final action automatically when a target exists.
        # This measures system policy behavior, not user-confirmation UX.
        if final_label in {"UPDATE", "MERGE", "FORGET"} and target is None:
            applied = False
            action_note = "missing_target"
        elif final_label == "ASK_USER":
            applied = False
            action_note = "ask_user_not_auto_applied"
        else:
            self.mgov.apply_final_action(
                conn=self.conn,
                user_id=self.user_id,
                final_label=final_label,
                user_message=turn,
                candidate=candidate,
                memory_type=memory_type,
                sensitivity=sensitivity,
                raw_label=raw_label,
                scores=scores,
                margin=margin,
                target=target,
                notes=f"external_tier2_eval; {override_reason}; {target_reason}",
            )
            applied = True
            action_note = "applied"

        return {
            "operation": final_label,
            "raw_operation": raw_label,
            "candidate": candidate,
            "applied": applied,
            "action_note": action_note,
            "margin": margin,
            "target_id": target["memory_id"] if target else None,
            "target_reason": target_reason,
            "latency_sec": time.perf_counter() - t0,
        }

    def retrieve(self, query):
        return self.mgov.retrieve_for_answer_context(self.conn, self.user_id, query)

    def export_state(self):
        rows = [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT memory_id, status, memory_type, sensitivity, text, supersedes_id
                FROM memories
                ORDER BY created_at
                """
            )
        ]
        return {
            "memories": rows,
            "active_texts": [r["text"] for r in rows if r["status"] == "active"],
            "deleted_texts": [r["text"] for r in rows if r["status"] == "deleted"],
            "quarantined_texts": [r["text"] for r in rows if r["status"] == "quarantined"],
            "superseded_texts": [r["text"] for r in rows if r["status"] == "superseded"],
        }

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


class Mem0Adapter(BaseAdapter):
    name = "mem0"

    def __init__(self):
        self.api_key = os.environ.get("MEM0_API_KEY")
        if not self.api_key:
            raise RuntimeError("MEM0_API_KEY is not set")

        try:
            from mem0 import MemoryClient
        except Exception as exc:
            raise RuntimeError("mem0ai is not installed. Try: pip install --user mem0ai") from exc

        self.client = MemoryClient(api_key=self.api_key)
        self.user_id = None
        self.created_ids = []

    def reset(self, case_id):
        self.user_id = f"memorygov_tier2_{case_id}_{uuid.uuid4().hex[:8]}"
        self.created_ids = []

    def seed(self, memories):
        for m in memories:
            out = self.client.add(
                [{"role": "user", "content": m}],
                user_id=self.user_id,
            )
            self._track_created_ids(out)

    def _track_created_ids(self, out):
        # Mem0 API response shape can vary by version/platform.
        if isinstance(out, dict):
            for key in ("id", "memory_id"):
                if key in out:
                    self.created_ids.append(out[key])
            for key in ("results", "memories"):
                if isinstance(out.get(key), list):
                    for item in out[key]:
                        if isinstance(item, dict):
                            mid = item.get("id") or item.get("memory_id")
                            if mid:
                                self.created_ids.append(mid)
        elif isinstance(out, list):
            for item in out:
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("memory_id")
                    if mid:
                        self.created_ids.append(mid)

    def process_turn(self, turn):
        t0 = time.perf_counter()
        out = self.client.add(
            [{"role": "user", "content": turn}],
            user_id=self.user_id,
        )
        self._track_created_ids(out)

        return {
            "operation": "MEM0_AUTO_ADD",
            "raw_response": redact(json.dumps(out, ensure_ascii=False, default=str))[:4000],
            "applied": True,
            "latency_sec": time.perf_counter() - t0,
        }

    def retrieve(self, query):
        t0 = time.perf_counter()
        out = self.client.search(query, filters={"user_id": self.user_id})
        elapsed = time.perf_counter() - t0

        rows = []
        if isinstance(out, dict) and isinstance(out.get("results"), list):
            source = out["results"]
        elif isinstance(out, list):
            source = out
        else:
            source = []

        for item in source:
            if not isinstance(item, dict):
                continue
            rows.append({
                "memory_id": item.get("id") or item.get("memory_id"),
                "text": item.get("memory") or item.get("text") or item.get("content") or str(item),
                "score": item.get("score"),
                "latency_sec": elapsed,
            })

        return rows

    def export_state(self):
        # Mem0 does not necessarily expose all memories through the same SDK method
        # across all versions, so we approximate visible state by searching broad probes.
        probes = [
            "memory governance project dark mode old address national ID therapy laptop battery",
            "What do you remember about the user?",
            "User preferences and projects",
        ]

        retrieved = []
        seen = set()

        for q in probes:
            try:
                for r in self.retrieve(q):
                    key = r.get("memory_id") or r.get("text")
                    if key not in seen:
                        seen.add(key)
                        retrieved.append(r)
            except Exception:
                pass

        return {
            "visible_retrieved_memories": retrieved,
            "active_texts": [x.get("text", "") for x in retrieved],
            "deleted_texts": [],
            "quarantined_texts": [],
            "superseded_texts": [],
            "note": "Mem0 state export is approximated through search probes unless a full list API is configured.",
        }



class Mem0GovernedAdapter(MemoryGovAdapter):
    name = "mem0_governed"

    def __init__(self, db_root, margin_threshold=0.10):
        super().__init__(db_root=db_root, margin_threshold=margin_threshold)

        self.api_key = os.environ.get("MEM0_API_KEY")
        if not self.api_key:
            raise RuntimeError("MEM0_API_KEY is not set")

        try:
            from mem0 import MemoryClient
        except Exception as exc:
            raise RuntimeError("mem0ai is not installed. Try: pip install --user mem0ai") from exc

        self.client = MemoryClient(api_key=self.api_key)
        self.mem0_user_id = None
        self.mem0_created_ids = []

    def reset(self, case_id):
        super().reset(case_id)
        self.mem0_user_id = f"memorygov_governed_{case_id}_{uuid.uuid4().hex[:8]}"
        self.mem0_created_ids = []

    def _track_created_ids(self, out):
        if isinstance(out, dict):
            for key in ("id", "memory_id"):
                if key in out:
                    self.mem0_created_ids.append(out[key])
            for key in ("results", "memories"):
                if isinstance(out.get(key), list):
                    for item in out[key]:
                        if isinstance(item, dict):
                            mid = item.get("id") or item.get("memory_id")
                            if mid:
                                self.mem0_created_ids.append(mid)
        elif isinstance(out, list):
            for item in out:
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("memory_id")
                    if mid:
                        self.mem0_created_ids.append(mid)

    def seed(self, memories):
        # Seed both MemoryGov local governance state and Mem0 visible storage.
        super().seed(memories)

        for m in memories:
            try:
                out = self.client.add(
                    [{"role": "user", "content": m}],
                    user_id=self.mem0_user_id,
                )
                self._track_created_ids(out)
            except Exception as exc:
                print(f"[mem0_governed] seed add failed: {exc}")

    def _mem0_add_text(self, text):
        if not text:
            return

        try:
            out = self.client.add(
                [{"role": "user", "content": text}],
                user_id=self.mem0_user_id,
            )
            self._track_created_ids(out)
        except Exception as exc:
            print(f"[mem0_governed] add failed: {exc}")

    def process_turn(self, turn):
        # Use MemoryGov to decide and mutate the governed local state.
        result = super().process_turn(turn)

        op = result.get("operation")
        candidate = result.get("candidate")
        applied = result.get("applied")

        # Mem0 is used as the storage/retrieval backend only after governance.
        # We do not write ignored, quarantined, or ask-user-pending facts.
        if applied and op in {"WRITE", "MERGE", "UPDATE"}:
            self._mem0_add_text(candidate)

        # FORGET is respected in the governed local state. Mem0 deletion is SDK/API-version
        # dependent, so visible Mem0 retrieval may still surface stale seed facts. The
        # exported governed state remains the MemoryGov state of record.
        return result

    def retrieve(self, query):
        # Prefer Mem0 search as the retrieval backend, but filter through governed active state
        # so superseded/deleted/quarantined local memories do not count as active context.
        active_texts = set((self.export_state().get("active_texts") or []))

        try:
            out = self.client.search(query, filters={"user_id": self.mem0_user_id})
        except Exception as exc:
            print(f"[mem0_governed] search failed: {exc}")
            return super().retrieve(query)

        if isinstance(out, dict) and isinstance(out.get("results"), list):
            source = out["results"]
        elif isinstance(out, list):
            source = out
        else:
            source = []

        rows = []
        for item in source:
            if not isinstance(item, dict):
                continue

            text = item.get("memory") or item.get("text") or item.get("content") or str(item)

            # Keep only memories that are active in MemoryGov's governed state.
            if text in active_texts:
                rows.append({
                    "memory_id": item.get("id") or item.get("memory_id"),
                    "text": text,
                    "score": item.get("score"),
                    "backend": "mem0_governed_filtered",
                })

        # If Mem0 search misses active governed memory because of semantic/search mismatch,
        # fall back to MemoryGov hybrid retrieval. This keeps the ablation focused on
        # governance correctness while still recording Mem0 as attempted backend.
        if not rows:
            rows = super().retrieve(query)
            for r in rows:
                r["backend"] = "memorygov_hybrid_fallback"

        return rows


def build_adapters(args):
    adapters = []

    if "memorygov" in args.systems:
        adapters.append(MemoryGovAdapter(
            db_root=OUT_DIR / "memorygov_dbs",
            margin_threshold=args.margin_threshold,
        ))

    if "mem0" in args.systems:
        try:
            adapters.append(Mem0Adapter())
        except Exception as exc:
            print(f"[skip mem0] {exc}")

    if "mem0_governed" in args.systems:
        try:
            adapters.append(Mem0GovernedAdapter(
                db_root=OUT_DIR / "mem0_governed_dbs",
                margin_threshold=args.margin_threshold,
            ))
        except Exception as exc:
            print(f"[skip mem0_governed] {exc}")

    if "graphiti" in args.systems:
        adapters.append(CapabilityOnlyAdapter(
            name="graphiti",
            capabilities={
                "WRITE": "partial/native episode ingestion",
                "UPDATE": "temporal graph update",
                "MERGE": "entity/fact consolidation, not MemoryGov policy label",
                "FORGET": "requires explicit deletion/CRUD configuration",
                "ASK_USER": "not native governance label",
                "QUARANTINE": "not native governance label",
                "RETRIEVE": "native hybrid retrieval",
                "SUPPRESS": "not native governance label",
            },
            note="Graphiti full adapter pending backend setup.",
        ))

    if "letta" in args.systems:
        adapters.append(CapabilityOnlyAdapter(
            name="letta",
            capabilities={
                "WRITE": "agent memory supported",
                "UPDATE": "tool/agent-mediated",
                "MERGE": "tool/agent-mediated",
                "FORGET": "tool/agent-mediated or custom",
                "ASK_USER": "possible via human-in-the-loop/tooling",
                "QUARANTINE": "custom policy required",
                "RETRIEVE": "agent memory/RAG supported",
                "SUPPRESS": "custom policy required",
            },
            note="Letta full adapter pending server/API setup.",
        ))

    if "langgraph" in args.systems:
        adapters.append(CapabilityOnlyAdapter(
            name="langgraph",
            capabilities={
                "WRITE": "framework-level store",
                "UPDATE": "custom implementation",
                "MERGE": "custom implementation",
                "FORGET": "custom implementation",
                "ASK_USER": "custom graph/control flow",
                "QUARANTINE": "custom implementation",
                "RETRIEVE": "store search supported",
                "SUPPRESS": "custom policy required",
            },
            note="LangGraph full adapter pending store implementation.",
        ))

    return adapters


def evaluate_case(adapter, case):
    adapter.reset(case["case_id"])
    adapter.seed(case.get("prior_memories", []))

    if isinstance(adapter, CapabilityOnlyAdapter):
        return {
            "system": adapter.name,
            "adapter_type": "capability_only",
            "case_id": case["case_id"],
            "category": case.get("category"),
            "risk": case.get("risk"),
            "turn_results": [],
            "retrieved": [],
            "state": adapter.export_state(),
            "checks": {},
            "score": None,
            "note": "Capability-only placeholder. Not behaviorally scored.",
        }

    turns = case.get("turn_sequence") or [case["turn"]]
    turn_results = []

    for turn in turns:
        turn_results.append(adapter.process_turn(turn))

    final_query = turns[-1]
    retrieved = adapter.retrieve(final_query)
    state = adapter.export_state()

    active_texts = state.get("active_texts", [])
    deleted_texts = state.get("deleted_texts", [])
    quarantined_texts = state.get("quarantined_texts", [])
    superseded_texts = state.get("superseded_texts", [])
    retrieved_texts = [x.get("text", "") for x in retrieved]

    checks = {}

    if "expected_operation" in case:
        checks["operation_match"] = turn_results[-1].get("operation") == case["expected_operation"]

    if "expected_operation_sequence" in case:
        observed = [x.get("operation") for x in turn_results]
        checks["operation_sequence_match"] = observed == case["expected_operation_sequence"]

    if "expected_active_contains" in case:
        checks["active_contains_all"] = contains_all(active_texts, case["expected_active_contains"])

    if "expected_inactive_contains" in case:
        inactive = deleted_texts + superseded_texts + quarantined_texts
        checks["inactive_contains_any"] = contains_any(inactive, case["expected_inactive_contains"])

    if "expected_deleted_contains" in case:
        checks["deleted_contains_all"] = contains_all(deleted_texts, case["expected_deleted_contains"])

    if "expected_not_active_contains" in case:
        checks["not_active_contains"] = not contains_any(active_texts, case["expected_not_active_contains"])

    if "expected_retrieval_contains" in case:
        checks["retrieval_contains_all"] = contains_all(retrieved_texts, case["expected_retrieval_contains"])

    if "expected_retrieval_not_contains" in case:
        checks["retrieval_not_contains"] = not contains_any(retrieved_texts, case["expected_retrieval_not_contains"])

    scored_checks = [v for v in checks.values() if isinstance(v, bool)]
    score = sum(1 for v in scored_checks if v) / len(scored_checks) if scored_checks else None

    return {
        "system": adapter.name,
        "case_id": case["case_id"],
        "category": case.get("category"),
        "risk": case.get("risk"),
        "turn_results": turn_results,
        "retrieved": retrieved,
        "state": state,
        "checks": checks,
        "score": score,
    }


def summarize(rows):
    by_system = {}

    for row in rows:
        sys = row["system"]
        by_system.setdefault(sys, {
            "system": sys,
            "cases": 0,
            "scored_cases": 0,
            "score_sum": 0.0,
            "perfect_cases": 0,
            "category_scores": {},
        })

        s = by_system[sys]
        s["cases"] += 1

        if row["score"] is not None:
            s["scored_cases"] += 1
            s["score_sum"] += row["score"]
            if row["score"] == 1.0:
                s["perfect_cases"] += 1

            cat = row.get("category") or "unknown"
            s["category_scores"].setdefault(cat, [])
            s["category_scores"][cat].append(row["score"])

    summaries = []
    for sys, s in by_system.items():
        category_mean = {
            k: sum(v) / len(v)
            for k, v in sorted(s["category_scores"].items())
        }

        summaries.append({
            "system": sys,
            "cases": s["cases"],
            "scored_cases": s["scored_cases"],
            "mean_score": s["score_sum"] / s["scored_cases"] if s["scored_cases"] else None,
            "perfect_cases": s["perfect_cases"],
            "category_mean_score": category_mean,
        })

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        default="memorygov,mem0,graphiti,letta,langgraph",
        help="Comma-separated: memorygov,mem0,mem0_governed,graphiti,letta,langgraph",
    )
    parser.add_argument("--margin_threshold", type=float, default=0.10)
    parser.add_argument(
        "--cases_path",
        default=str(CASES_PATH),
        help="Path to external Tier 2 cases JSONL file.",
    )
    args = parser.parse_args()

    args.systems = [x.strip() for x in args.systems.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(Path(args.cases_path))
    adapters = build_adapters(args)

    print("Systems:", [a.name for a in adapters])
    print("Cases:", len(cases))

    rows = []

    for adapter in adapters:
        print(f"\n=== Evaluating {adapter.name} ===")
        for case in cases:
            try:
                row = evaluate_case(adapter, case)
                print(f"{adapter.name} {case['case_id']} score={row['score']}")
            except Exception as exc:
                row = {
                    "system": adapter.name,
                    "case_id": case["case_id"],
                    "category": case.get("category"),
                    "risk": case.get("risk"),
                    "error": repr(exc),
                    "score": None,
                }
                print(f"{adapter.name} {case['case_id']} ERROR {exc}")
            rows.append(row)

        adapter.close()

    summary = summarize(rows)

    write_jsonl(OUT_DIR / "tier2_external_results.jsonl", rows)
    (OUT_DIR / "tier2_external_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
