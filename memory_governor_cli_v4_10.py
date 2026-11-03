import argparse
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib


LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

DEFAULT_MODEL = (
    "processed/operation_classifier_results/"
    "non_llm_v1_2_forget_aug_selected/"
    "tfidf_sgd_hinge_balanced/model.joblib"
)

DEFAULT_DB = "processed/demo_memory_store_v3/memories.sqlite"

STOPWORDS = {
    "i", "me", "my", "you", "your", "user", "the", "a", "an", "and", "or", "but",
    "to", "of", "in", "on", "for", "with", "about", "that", "this", "it", "is",
    "are", "was", "were", "be", "been", "being", "now", "currently", "please",
    "memory", "remember", "stored", "store", "delete", "forget", "remove", "from",
    "into", "should", "could", "would", "what", "when", "where", "why", "how",
}

QUESTION_STARTERS = {
    "what", "when", "where", "why", "how", "who", "which", "can", "could",
    "should", "would", "do", "does", "did", "is", "are", "will"
}


GLOBAL_RETRIEVAL_BACKEND = "hybrid"
GLOBAL_EMBEDDER = None
GLOBAL_EMBEDDING_MODEL = None
GLOBAL_EMBEDDING_WARNING_PRINTED = False


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def short_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def connect(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            memory_type TEXT,
            sensitivity TEXT,
            status TEXT NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            supersedes_id TEXT,
            merged_from_ids TEXT,
            metadata_json TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operation_log (
            log_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            user_message TEXT,
            candidate_memory TEXT,
            raw_predicted_label TEXT,
            final_policy_label TEXT,
            final_action TEXT,
            scores_json TEXT,
            margin REAL,
            selected_target_id TEXT,
            notes TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS retrieval_log (
            log_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            user_message TEXT,
            retrieved_memory_ids TEXT,
            retrieved_context_json TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (memory_id, model_name)
        )
        """
    )

    conn.commit()



def load_model(path):
    payload = joblib.load(path)

    if isinstance(payload, dict):
        if "model" in payload:
            return payload["model"], payload.get("label_encoder")
        if "pipeline" in payload:
            return payload["pipeline"], payload.get("label_encoder")
        if "clf" in payload:
            return payload["clf"], payload.get("label_encoder")

    return payload, None

def simple_tokens(text):
    toks = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def lexical_relevance(query, memory_text):
    q = set(simple_tokens(query))
    m = set(simple_tokens(memory_text))
    if not q or not m:
        return 0.0
    overlap = len(q & m)
    return overlap / max(1, min(len(q), len(m)))


def get_memories(conn, user_id, status=None):
    if status:
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND status = ?
            ORDER BY created_at ASC
            """,
            (user_id, status),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_memory(conn, user_id, memory_id):
    row = conn.execute(
        """
        SELECT * FROM memories
        WHERE user_id = ? AND memory_id = ?
        """,
        (user_id, memory_id),
    ).fetchone()
    return dict(row) if row else None



def load_embedder(model_name):
    global GLOBAL_EMBEDDING_WARNING_PRINTED

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        if not GLOBAL_EMBEDDING_WARNING_PRINTED:
            print(f"Embedding backend unavailable: sentence_transformers import failed: {exc}")
            GLOBAL_EMBEDDING_WARNING_PRINTED = True
        return None

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        # Older sentence-transformers may not support local_files_only.
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            if not GLOBAL_EMBEDDING_WARNING_PRINTED:
                print(f"Embedding backend unavailable: could not load {model_name}: {exc}")
                GLOBAL_EMBEDDING_WARNING_PRINTED = True
            return None
    except Exception as exc:
        if not GLOBAL_EMBEDDING_WARNING_PRINTED:
            print(f"Embedding backend unavailable: could not load {model_name}: {exc}")
            GLOBAL_EMBEDDING_WARNING_PRINTED = True
        return None


def normalize_vec(vec):
    import math
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(float(x) * float(y) for x, y in zip(a, b))


def encode_text(embedder, text):
    vec = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return [float(x) for x in vec]


def get_or_create_memory_embedding(conn, mem, embedder, model_name):
    row = conn.execute(
        """
        SELECT embedding_json FROM memory_embeddings
        WHERE memory_id = ? AND model_name = ?
        """,
        (mem["memory_id"], model_name),
    ).fetchone()

    if row:
        return json.loads(row["embedding_json"])

    vec = encode_text(embedder, mem["text"])

    conn.execute(
        """
        INSERT OR REPLACE INTO memory_embeddings (
            memory_id, model_name, embedding_json, updated_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            mem["memory_id"],
            model_name,
            json.dumps(vec),
            now_iso(),
        ),
    )
    conn.commit()

    return vec


def retrieve_by_embedding(conn, user_id, query, limit=5, min_score=0.20):
    if GLOBAL_EMBEDDER is None:
        return []

    active = get_memories(conn, user_id, status="active")
    if not active:
        return []

    qvec = encode_text(GLOBAL_EMBEDDER, query)

    scored = []
    for mem in active:
        mvec = get_or_create_memory_embedding(conn, mem, GLOBAL_EMBEDDER, GLOBAL_EMBEDDING_MODEL)
        score = cosine(qvec, mvec)
        if score >= min_score:
            mem2 = dict(mem)
            mem2["retrieval_score"] = score
            mem2["retrieval_backend"] = "embedding"
            scored.append(mem2)

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:limit]


def retrieve_by_lexical(conn, user_id, query, limit=5, min_score=0.15):
    active = get_memories(conn, user_id, status="active")
    scored = []

    for mem in active:
        score = lexical_relevance(query, mem["text"])
        if score >= min_score:
            mem2 = dict(mem)
            mem2["retrieval_score"] = score
            mem2["retrieval_backend"] = "lexical"
            scored.append(mem2)

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:limit]


def dedupe_ranked_memories(rows, limit=5):
    best = {}
    for row in rows:
        mid = row["memory_id"]
        if mid not in best or row["retrieval_score"] > best[mid]["retrieval_score"]:
            best[mid] = row

    out = list(best.values())
    out.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return out[:limit]


def retrieve_for_answer_context(conn, user_id, user_message, limit=5, min_score=0.15):
    if GLOBAL_RETRIEVAL_BACKEND == "embedding":
        emb = retrieve_by_embedding(conn, user_id, user_message, limit=limit, min_score=0.20)
        if emb:
            return emb
        return retrieve_by_lexical(conn, user_id, user_message, limit=limit, min_score=min_score)

    if GLOBAL_RETRIEVAL_BACKEND == "hybrid":
        emb = retrieve_by_embedding(conn, user_id, user_message, limit=limit, min_score=0.20)
        lex = retrieve_by_lexical(conn, user_id, user_message, limit=limit, min_score=min_score)
        return dedupe_ranked_memories(emb + lex, limit=limit)

    return retrieve_by_lexical(conn, user_id, user_message, limit=limit, min_score=min_score)


def retrieve_for_target(conn, user_id, candidate_memory, limit=5):
    if GLOBAL_RETRIEVAL_BACKEND == "embedding":
        emb = retrieve_by_embedding(conn, user_id, candidate_memory, limit=limit, min_score=0.20)
        if emb:
            return emb
        return retrieve_by_lexical(conn, user_id, candidate_memory, limit=limit, min_score=0.01)

    if GLOBAL_RETRIEVAL_BACKEND == "hybrid":
        emb = retrieve_by_embedding(conn, user_id, candidate_memory, limit=limit, min_score=0.20)
        lex = retrieve_by_lexical(conn, user_id, candidate_memory, limit=limit, min_score=0.01)
        return dedupe_ranked_memories(emb + lex, limit=limit)

    return retrieve_by_lexical(conn, user_id, candidate_memory, limit=limit, min_score=0.01)


def build_answer_context(user_message, retrieved):
    if not retrieved:
        memory_context = "No relevant active memories retrieved."
    else:
        parts = []
        for i, mem in enumerate(retrieved, 1):
            parts.append(
                f"{i}. [{mem['memory_id']}] score={mem['retrieval_score']:.3f} "
                f"type={mem['memory_type']} sensitivity={mem['sensitivity']} | {mem['text']}"
            )
        memory_context = "\n".join(parts)

    return f"""User message:
{user_message}

Relevant memory context:
{memory_context}

Instruction for answer model:
Use the relevant memory context only if it is useful for answering the user. Do not expose memory IDs to the user.""".strip()


def print_memories(memories):
    if not memories:
        print("No memories found.")
        return

    for mem in memories:
        score = ""
        if "retrieval_score" in mem:
            score = f" score={mem['retrieval_score']:.3f}"
        print()
        print(f"[{mem['memory_id']}] status={mem['status']} type={mem['memory_type']} sensitivity={mem['sensitivity']}{score}")
        print(f"  text: {mem['text']}")
        if mem.get("supersedes_id"):
            print(f"  supersedes: {mem['supersedes_id']}")
        if mem.get("merged_from_ids"):
            print(f"  merged_from_ids: {mem['merged_from_ids']}")


def is_deletion_request(user_message):
    t = user_message.lower()
    markers = [
        "forget", "delete", "remove", "stop remembering", "do not remember",
        "don't remember", "do not keep", "erase", "clear the memory"
    ]
    return any(m in t for m in markers)


def is_likely_pure_query(user_message):
    t = user_message.strip().lower()
    first = re.findall(r"[a-zA-Z]+", t)
    first = first[0] if first else ""

    if "?" in user_message and first in QUESTION_STARTERS:
        preference_markers = ["i prefer", "i like", "i no longer", "remember that", "my "]
        if not any(m in t for m in preference_markers):
            return True

    if first in QUESTION_STARTERS and not any(m in t for m in ["i prefer", "i like", "i no longer", "my "]):
        return True

    return False


def infer_memory_type(user_message):
    t = user_message.lower()

    preference_markers = [
        "prefer", "like", "dislike", "favorite", "rather", "instead",
        "dark mode", "light mode", "theme", "answer style", "format",
        "meeting", "meetings", "reminder", "reminders"
    ]

    project_markers = [
        "project", "thesis", "experiment", "baseline", "deadline",
        "classifier", "paper", "benchmark", "memorygovbench",
        "memory governance", "operation governor", "retrieval governor"
    ]

    sensitive_markers = [
        "doctor", "therapy", "diagnosis", "salary", "address", "id number",
        "credit card", "political", "religion", "medical", "national id"
    ]

    habit_markers = [
        "always", "usually", "normally", "every morning", "workflow"
    ]

    if any(x in t for x in sensitive_markers):
        return "sensitive"

    if any(x in t for x in project_markers):
        return "project_state"

    if any(x in t for x in preference_markers):
        return "preference"

    if any(x in t for x in habit_markers):
        return "habit"

    return "episodic"


def infer_sensitivity(user_message):
    t = user_message.lower()

    high = [
        "national id", "id number", "credit card", "address", "diagnosis", "hiv",
        "salary", "undocumented", "political", "religion", "sexual orientation",
        "illegal drugs", "medical", "doctor", "home address"
    ]
    medium = [
        "therapy", "lonely", "breakup", "relationship", "family conflict",
        "job hunting", "financial stress", "sick parent", "sleeping"
    ]

    if any(x in t for x in high):
        return "high"
    if any(x in t for x in medium):
        return "medium"
    return "low"


def extract_candidate_memory(user_message):
    # v4.8: natural correction/update extraction.
    # Handles turns such as:
    # "Do not use the old evaluation path anymore; the active evaluation path is processed/eval_v2."
    _msg = str(user_message or "").strip()
    _lower = _msg.lower()

    # v4.10: natural switch/update extraction.
    # Handles: "Switch the active evaluation path from OLD to NEW"
    if "switch" in _lower and " from " in f" {_lower} " and " to " in f" {_lower} ":
        import re as _re
        _m = _re.search(
            r"(?:switch|change|move)\s+(?:the\s+)?([A-Za-z0-9_ ./:-]{2,80}?)\s+from\s+([^;\n.]+?)\s+to\s+([^;\n.]+)",
            _msg,
            flags=_re.IGNORECASE,
        )
        if _m:
            _subject = " ".join(_m.group(1).strip().split())
            _new_value = _m.group(3).strip().rstrip(".; ")
            if _subject and _new_value:
                return f"The {_subject} is {_new_value}."


    if (
        ("do not use" in _lower or "don't use" in _lower or "not anymore" in _lower or "no longer" in _lower)
        and (" active " in f" {_lower} " or " current " in f" {_lower} ")
    ):
        import re as _re
        _m = _re.search(
            r"(?:the\s+)?(?:active|current)\s+([A-Za-z0-9_ ./:-]{2,80}?)\s+(?:is|=|:)\s+([^\n.;]+)",
            _msg,
            flags=_re.IGNORECASE,
        )
        if _m:
            _subject = " ".join(_m.group(1).strip().split())
            _value = _m.group(2).strip()
            _value = _value.rstrip(".; ")
            if _subject and _value:
                return f"The active {_subject} is {_value}."


    if is_likely_pure_query(user_message):
        return None

    t = user_message.strip()
    low = t.lower()

    if is_deletion_request(t):
        return "User asks to delete or forget: " + t

    if low.startswith("i prefer "):
        return "User prefers " + t[len("I prefer "):].rstrip(".") + "."

    if low.startswith("i like "):
        return "User likes " + t[len("I like "):].rstrip(".") + "."

    if low.startswith("i stopped using "):
        return "User stopped using " + t[len("I stopped using "):].rstrip(".") + "."

    if low.startswith("i no longer "):
        return "User no longer " + t[len("I no longer "):].rstrip(".") + "."

    if " no longer " in low:
        return "User stated a changed preference or state: " + t

    if " switched from " in low or " switched to " in low:
        return "User switched preferences or tools: " + t

    project_markers = [
        "project", "thesis", "benchmark", "memory governance",
        "memorygovbench", "operation governor", "retrieval governor",
        "classifier", "baseline"
    ]

    if any(p in low for p in project_markers):
        if "now includes" in low:
            after = t.split("now includes", 1)[1].strip().rstrip(".")
            return "User's memory governance project includes " + after + "."

        if "also includes" in low:
            after = t.split("also includes", 1)[1].strip().rstrip(".")
            return "User's memory governance project also includes " + after + "."

        if "now has" in low:
            after = t.split("now has", 1)[1].strip().rstrip(".")
            return "User's memory governance project has " + after + "."

        if "added" in low:
            return "User added a detail to the memory governance project: " + t

        if any(m in low for m in ["is", "now", "changed", "uses", "has", "includes"]):
            return "User stated a project update: " + t

    if low.startswith("i am considering ") or low.startswith("i'm considering "):
        after = t.split("considering", 1)[1].strip().rstrip(".")
        return "User is considering " + after + "."

    if low.startswith("my "):
        return "User stated: " + t

    durable_markers = [
        "i usually", "i always", "i normally", "remember that",
        "i work on", "i am working on", "i changed", "i moved",
        "i switched", "i use", "i now use"
    ]

    if any(m in low for m in durable_markers):
        return "User stated: " + t

    return None


def build_operation_input(user_message, candidate_memory, existing_memories, memory_type, sensitivity):
    history = f"user: {user_message}" if user_message else "None"

    if existing_memories:
        existing = []
        for mem in existing_memories:
            existing.append(
                f"{mem['memory_id']}: {mem['text']} "
                f"[type={mem['memory_type']}; sensitivity={mem['sensitivity']}; status={mem['status']}]"
            )
        existing_text = " | ".join(existing)
    else:
        existing_text = "None"

    return f"""Task: operation_governance
Question: Decide what the memory system should do with the candidate memory.

Allowed labels:
WRITE, IGNORE, UPDATE, MERGE, FORGET, QUARANTINE

History context:
{history}

Candidate memory:
{candidate_memory}

Candidate metadata:
type={memory_type}
sensitivity={sensitivity}

Existing memories:
{existing_text}

Decision target:
Choose the correct operation label for the candidate memory.""".strip()


def sorted_scores(scores):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def score_margin(scores):
    items = sorted_scores(scores)
    if len(items) < 2:
        return 999.0
    return float(items[0][1]) - float(items[1][1])



def predict_operation(model, label_encoder, op_input):
    import numpy as np

    X = [semanticize_operation_input(op_input)]
    raw_pred = model.predict(X)[0]

    if label_encoder is not None:
        try:
            if isinstance(raw_pred, str):
                label = raw_pred
            else:
                label = label_encoder.inverse_transform([raw_pred])[0]
        except Exception:
            label = str(raw_pred)
    else:
        label = str(raw_pred)

    def get_classes(m):
        if hasattr(m, "classes_"):
            return list(m.classes_)
        if hasattr(m, "named_steps"):
            try:
                return list(m.named_steps["clf"].classes_)
            except Exception:
                pass
        return []

    classes = get_classes(model)
    scores = {}

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)[0]
            if classes and len(classes) == len(probs):
                scores = {str(c): float(v) for c, v in zip(classes, probs)}
            else:
                scores = {label: float(max(probs))}
        except Exception:
            scores = {label: 1.0}

    elif hasattr(model, "decision_function"):
        try:
            margins = np.asarray(model.decision_function(X))

            if margins.ndim == 2:
                vals = margins[0]
                if classes and len(classes) == len(vals):
                    scores = {str(c): float(v) for c, v in zip(classes, vals)}
                else:
                    scores = {label: float(np.max(vals))}
            elif margins.ndim == 1:
                val = float(margins[0])
                if len(classes) == 2:
                    scores = {str(classes[0]): -val, str(classes[1]): val}
                else:
                    scores = {label: val}
            else:
                scores = {label: 1.0}
        except Exception:
            scores = {label: 1.0}
    else:
        scores = {label: 1.0}

    return label, scores

def is_temporary_or_session_fact(user_message, candidate):
    text = f"{user_message} {candidate}".lower()

    durable_exceptions = [
        "i prefer",
        "my preference",
        "i stopped",
        "i no longer",
        "from now on",
        "going forward",
        "remember that",
        "do not forget",
        "save this",
        "store this",
        "keep this in memory",
    ]

    if any(x in text for x in durable_exceptions):
        return False

    temporary_markers = [
        "currently",
        "right now",
        "at the moment",
        "today",
        "tonight",
        "this session",
        "for this session",
        "battery",
        "laptop battery",
        "phone battery",
        "playlist",
        "listening to",
        "weather",
        "notification",
        "just got",
        "just received",
        "current tab",
        "current window",
        "temporary",
        "for now",
    ]

    return any(x in text for x in temporary_markers)


def has_memory_consent_uncertainty(user_message, candidate):
    text = f"{user_message} {candidate}".lower()

    markers = [
        "not sure i want you to remember",
        "not sure if i want you to remember",
        "i don't know if i want you to remember",
        "i do not know if i want you to remember",
        "should you remember",
        "ask me before remembering",
        "ask before remembering",
        "don't remember unless i confirm",
        "do not remember unless i confirm",
    ]

    return any(x in text for x in markers)




def has_high_risk_sensitive_marker(user_message, candidate):
    text = f"{user_message} {candidate}".lower()

    markers = [
        "national id",
        "passport",
        "passport number",
        "bank account",
        "iban",
        "credit card",
        "api token",
        "access token",
        "secret token",
        "database password",
        "password",
        "private ssh key",
        "private key",
        "seed phrase",
        "recovery code",
        "tax identifier",
        "insurance number",
        "crypto seed",
        "home alarm code",
        "alarm code",
        "bank pin",
        " pin ",
        "cloud access key",
        "vpn secret",
        "exact home address",
        "home address",
        "private phone number",
        "medical diagnosis",
        "diagnosis",
        "medical prescription",
        "prescription",
        "hr complaint",
        "legal case number",
        "case file",
        "identity scan",
        "salary is",
        "salary amount",
        "child attends",
        "therapy",
        "medical leave",
        "doctor",
        "medication",
        "loan payment",
        "divorce",
    ]

    return any(m in text for m in markers)


def looks_like_natural_update_instruction(user_message, _policy_candidate):
    """
    v4.9: Distinguish correction/update wording from deletion.
    Example:
    "Do not use the old evaluation path anymore; the active evaluation path is processed/eval_v2."
    This contains deletion-like words, but semantically it gives a replacement value.
    """
    t = str(user_message or "").lower()
    c = str(_policy_candidate or "").lower()

    has_deprecation = any(x in t for x in [
        "do not use",
        "don't use",
        "not anymore",
        "no longer",
        "old ",
        "previous ",
        "stale ",
        "switch",
        "change",
        " from ",
    ])

    has_replacement = any(x in t for x in [
        " active ",
        " current ",
        " now ",
        " instead",
        " should be ",
        " is ",
        "=",
        ":",
    ])

    candidate_has_replacement = bool(c.strip()) and any(x in c for x in [
        "active",
        "current",
        "now",
        "processed/",
        "path",
        "version",
        "email",
        "preference",
    ])

    return has_deprecation and has_replacement and candidate_has_replacement

def normalize_legacy_ask_user_operation(label, user_message, candidate):
    # ASK_USER is no longer a final operation in v1.3/v4.4.
    # If any legacy path still emits it, convert it to the operation
    # that would be applied after confirmation.
    if label != "ASK_USER":
        return label, "no_legacy_ask_user"

    text = f"{user_message} {candidate}".lower()

    if "delete" in text or "forget" in text or "remove from memory" in text:
        return "FORGET", "legacy_ask_user_to_forget_confirm"

    if "no longer" in text or "stopped" in text or "instead" in text or "switched" in text:
        return "UPDATE", "legacy_ask_user_to_update_confirm"

    if has_high_risk_sensitive_marker(user_message, candidate):
        return "QUARANTINE", "legacy_ask_user_to_quarantine_confirm"

    return "WRITE", "legacy_ask_user_to_write_confirm"



def semanticize_operation_input(text):
    import re

    text = str(text)

    text = re.sub(
        r"Allowed labels:\s*(?:WRITE|IGNORE|UPDATE|MERGE|FORGET|QUARANTINE|,\s*|\s*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    lines = []
    skip_next_label_line = False

    for line in text.splitlines():
        s = line.strip()

        if skip_next_label_line:
            skip_next_label_line = False
            continue

        if s.startswith("Task:"):
            continue
        if s.startswith("Question:"):
            continue
        if s.startswith("Decision target:"):
            continue
        if s.startswith("Choose the correct operation label"):
            continue
        if s.startswith("Allowed labels:"):
            skip_next_label_line = True
            continue

        lines.append(line)

    text = "\n".join(lines)

    text = re.sub(
        r"Candidate metadata:\n(?:.*\n?){0,8}?(?=\n(?:Existing memories:|Decision target:|History context:|Candidate memory:|$))",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\[type=.*?; sensitivity=.*?; status=.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[type=.*?; sensitivity=.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\btype\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsensitivity\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstatus\s*=\s*\w+\b", "", text, flags=re.IGNORECASE)

    text = text.replace("History context:", "History:")
    text = text.replace("Candidate memory:", "Candidate:")
    text = text.replace("Existing memories:", "Existing:")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def policy_override(raw_label, user_message, candidate_memory, memory_type, sensitivity):

    # v4.9-fix: derive candidate inside policy_override for natural update detection.
    _policy_candidate = extract_candidate_memory(user_message)

    # Apply natural update precedence before deletion-like rules.
    # Example: "Do not use old X anymore; active/current X is Y."
    if looks_like_natural_update_instruction(user_message, _policy_candidate):
        return "UPDATE", "policy_override_natural_update_not_forget"


    _candidate_for_policy = locals().get(
        "candidate",
        locals().get("candidate_memory", locals().get("candidate_text", ""))
    )

    if has_high_risk_sensitive_marker(user_message, _candidate_for_policy):
        return "QUARANTINE", "policy_override_high_risk_sensitive_marker"
    _candidate_for_policy = locals().get(
        "candidate",
        locals().get("candidate_memory", locals().get("candidate_text", ""))
    )

    if is_temporary_or_session_fact(user_message, _candidate_for_policy):
        return "IGNORE", "policy_override_temporary_or_session_fact"

    if has_memory_consent_uncertainty(user_message, _candidate_for_policy):
        return "WRITE", "policy_confirm_uncertain_consent_write"

    t = user_message.lower()

    if is_deletion_request(user_message):
        if looks_like_natural_update_instruction(user_message, _policy_candidate):
            return "UPDATE", "policy_override_natural_update_not_forget"
        return "FORGET", "policy_override_explicit_deletion_request"

    high_pii_markers = [
        "national id", "id number", "credit card", "home address", "diagnosis",
        "hiv", "salary", "undocumented"
    ]

    if sensitivity == "high" and any(m in t for m in high_pii_markers):
        return "QUARANTINE", "policy_override_high_sensitivity"

    # Additive project details should normally merge into an existing project memory,
    # not supersede the whole project memory.
    additive_markers = [
        "now includes", "also includes", "includes an", "includes a",
        "added", "also has", "now has", "incorporates"
    ]

    project_markers = [
        "project", "thesis", "benchmark", "memory governance",
        "memorygovbench", "operation governor", "retrieval governor",
        "classifier", "baseline"
    ]

    if raw_label == "UPDATE" and any(a in t for a in additive_markers) and any(p in t for p in project_markers):
        return "MERGE", "policy_override_additive_project_detail"

    return raw_label, "model_label"


def insert_memory(conn, user_id, text, memory_type, sensitivity, status, source, metadata=None, supersedes_id=None, merged_from_ids=None):
    memory_id = short_id("mem")
    ts = now_iso()

    conn.execute(
        """
        INSERT INTO memories (
            memory_id, user_id, text, memory_type, sensitivity, status,
            source, created_at, updated_at, supersedes_id, merged_from_ids, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            user_id,
            text,
            memory_type,
            sensitivity,
            status,
            source,
            ts,
            ts,
            supersedes_id,
            json.dumps(merged_from_ids or [], ensure_ascii=False),
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )

    conn.commit()
    return memory_id


def update_memory_status(conn, user_id, memory_id, status):
    conn.execute(
        """
        UPDATE memories
        SET status = ?, updated_at = ?
        WHERE user_id = ? AND memory_id = ?
        """,
        (status, now_iso(), user_id, memory_id),
    )
    conn.commit()


def update_memory_text(conn, user_id, memory_id, new_text):
    conn.execute(
        """
        UPDATE memories
        SET text = ?, updated_at = ?
        WHERE user_id = ? AND memory_id = ?
        """,
        (new_text, now_iso(), user_id, memory_id),
    )
    conn.execute(
        """
        DELETE FROM memory_embeddings
        WHERE memory_id = ?
        """,
        (memory_id,),
    )
    conn.commit()


def log_retrieval(conn, user_id, user_message, retrieved):
    conn.execute(
        """
        INSERT INTO retrieval_log (
            log_id, user_id, created_at, user_message, retrieved_memory_ids, retrieved_context_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            short_id("rlog"),
            user_id,
            now_iso(),
            user_message,
            json.dumps([m["memory_id"] for m in retrieved], ensure_ascii=False),
            json.dumps(retrieved, ensure_ascii=False),
        ),
    )
    conn.commit()


def log_operation(conn, user_id, user_message, candidate_memory, raw_label, final_label, final_action, scores, margin, target_id=None, notes=""):
    conn.execute(
        """
        INSERT INTO operation_log (
            log_id, user_id, created_at, user_message, candidate_memory,
            raw_predicted_label, final_policy_label, final_action, scores_json,
            margin, selected_target_id, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            short_id("log"),
            user_id,
            now_iso(),
            user_message,
            candidate_memory,
            raw_label,
            final_label,
            final_action,
            json.dumps(scores, ensure_ascii=False),
            float(margin) if margin is not None else None,
            target_id,
            notes,
        ),
    )
    conn.commit()



def strip_deletion_words(text):
    low = text.lower()
    patterns = [
        "delete", "forget", "remove", "stop remembering", "do not remember",
        "don't remember", "do not keep", "erase", "clear the memory",
        "from memory", "memory"
    ]
    for pat in patterns:
        low = low.replace(pat, " ")
    low = re.sub(r"\s+", " ", low).strip()
    return low or text


def refine_targets_for_final_action(conn, user_id, final_label, user_message, candidate, memory_type, retrieved_targets):
    if not retrieved_targets:
        return []

    refined = []

    if final_label == "FORGET":
        deletion_query = strip_deletion_words(user_message)
        for mem in retrieved_targets:
            lex = lexical_relevance(deletion_query, mem["text"])
            same_sensitivity = mem.get("sensitivity") == sensitivity_from_message(user_message)
            keep = lex > 0 or mem.get("retrieval_score", 0) >= 0.80 or same_sensitivity
            if keep:
                mem2 = dict(mem)
                mem2["target_refine_score"] = max(lex, mem.get("retrieval_score", 0))
                refined.append(mem2)

        refined.sort(key=lambda x: x.get("target_refine_score", x.get("retrieval_score", 0)), reverse=True)
        return refined[:5] if refined else retrieved_targets[:1]

    if final_label in {"UPDATE", "MERGE"}:
        compatible = []
        fallback = []

        for mem in retrieved_targets:
            if mem.get("memory_type") == memory_type:
                compatible.append(mem)
            elif memory_type == "project_state" and mem.get("memory_type") == "project_state":
                compatible.append(mem)
            elif memory_type == "preference" and mem.get("memory_type") == "preference":
                compatible.append(mem)
            else:
                fallback.append(mem)

        return compatible[:5] if compatible else retrieved_targets[:3]

    return retrieved_targets


def sensitivity_from_message(user_message):
    return infer_sensitivity(user_message)


def choose_target_automatically(retrieved, min_score=0.20, min_gap=0.10):
    if not retrieved:
        return None, "no_retrieved_target"

    best = retrieved[0]
    second_score = retrieved[1]["retrieval_score"] if len(retrieved) > 1 else 0.0
    gap = best["retrieval_score"] - second_score

    if best["retrieval_score"] >= min_score and gap >= min_gap:
        return best, f"auto_target_score={best['retrieval_score']:.3f}_gap={gap:.3f}"

    return None, f"ambiguous_target_score={best['retrieval_score']:.3f}_gap={gap:.3f}"


def requires_confirmation(label, margin, margin_threshold, target, target_reason):
    if margin < margin_threshold:
        return True, f"low_margin_{margin:.4f}"

    if label == "FORGET":
        return True, "FORGET_high_impact_confirm_target"

    # UPDATE/MERGE with no selected target is handled structurally
    # in apply_final_action by falling back to WRITE.
    return False, f"confident_margin_{margin:.4f}"


def apply_final_action(conn, user_id, final_label, user_message, candidate, memory_type, sensitivity, raw_label, scores, margin, target=None, notes=""):
    if final_label == "WRITE":
        new_id = insert_memory(
            conn, user_id, candidate, memory_type, sensitivity, "active",
            "interactive_v3", {"operation": "WRITE"}
        )
        log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "WRITE", scores, margin, notes=f"{notes}; inserted {new_id}")
        print(f"Stored active memory: {new_id}")
        return

    if final_label == "IGNORE":
        log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "IGNORE", scores, margin, notes=notes)
        print("Ignored. Nothing stored.")
        return

    if final_label == "QUARANTINE":
        new_id = insert_memory(
            conn, user_id, candidate, memory_type, sensitivity, "quarantined",
            "interactive_v3", {"operation": "QUARANTINE"}
        )
        log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "QUARANTINE", scores, margin, notes=f"{notes}; quarantined {new_id}")
        print(f"Stored quarantined memory: {new_id}")
        return

    if final_label == "UPDATE":
        if not target:
            print("No target selected. Falling back to WRITE.")
            return apply_final_action(conn, user_id, "WRITE", user_message, candidate, memory_type, sensitivity, raw_label, scores, margin, notes="UPDATE_fallback_no_target")

        update_memory_status(conn, user_id, target["memory_id"], "superseded")
        new_id = insert_memory(
            conn, user_id, candidate, memory_type, sensitivity, "active",
            "interactive_v3", {"operation": "UPDATE"}, supersedes_id=target["memory_id"]
        )
        log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "UPDATE", scores, margin, target_id=target["memory_id"], notes=f"{notes}; superseded {target['memory_id']} with {new_id}")
        print(f"Updated. Superseded {target['memory_id']} with {new_id}")
        return

    if final_label == "MERGE":
        if not target:
            print("No target selected. Falling back to WRITE.")
            return apply_final_action(conn, user_id, "WRITE", user_message, candidate, memory_type, sensitivity, raw_label, scores, margin, notes="MERGE_fallback_no_target")

        merged_text = f"{target['text']} | {candidate}"
        update_memory_text(conn, user_id, target["memory_id"], merged_text)
        log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "MERGE", scores, margin, target_id=target["memory_id"], notes=f"{notes}; merged into {target['memory_id']}")
        print(f"Merged into memory: {target['memory_id']}")
        return

    if final_label == "FORGET":
        if not target:
            log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "FORGET_NO_TARGET", scores, margin, notes=f"{notes}; no target")
            print("No target selected. Nothing deleted.")
            return

        update_memory_status(conn, user_id, target["memory_id"], "deleted")
        log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "FORGET", scores, margin, target_id=target["memory_id"], notes=f"{notes}; deleted {target['memory_id']}")
        print(f"Deleted memory: {target['memory_id']}")
        return


def confirm_or_apply(conn, user_id, user_message, candidate, memory_type, sensitivity, raw_label, final_label, scores, retrieved_targets, margin_threshold, override_reason):
    margin = score_margin(scores)
    target = None
    target_reason = "not_targeted_action"

    if final_label in {"UPDATE", "MERGE", "FORGET"}:
        target, target_reason = choose_target_automatically(retrieved_targets)

    confirm, confirm_reason = requires_confirmation(final_label, margin, margin_threshold, target, target_reason)

    print()
    print(f"Raw model label: {raw_label}")
    print(f"Final policy label: {final_label}")
    print(f"Policy reason: {override_reason}")
    print(f"Margin: {margin:.4f}")
    print(f"Confirmation reason: {confirm_reason}")

    if retrieved_targets:
        print()
        print("Relevant target memories:")
        print_memories(retrieved_targets)

    if not confirm:
        apply_final_action(
            conn, user_id, final_label, user_message, candidate, memory_type, sensitivity,
            raw_label, scores, margin, target=target, notes=f"auto_applied; {override_reason}; {confirm_reason}; {target_reason}"
        )
        return

    print()
    print("Confirmation required.")
    print("Scores:")
    for label, score in sorted_scores(scores):
        print(f"  {label:12s} {score:.4f}")

    print()
    print("Choices:")
    print("  accept   = apply final policy label")
    print("  ignore   = do nothing")
    print("  override = choose another label")
    choice = input("Choice [accept/ignore/override]: ").strip().lower()

    if choice == "ignore":
        log_operation(conn, user_id, user_message, candidate, raw_label, final_label, "USER_DECLINED", scores, margin, notes=confirm_reason)
        print("Not applied.")
        return

    chosen_label = final_label
    if choice in {"override", "o"}:
        print("Allowed labels:", ", ".join(LABELS))
        chosen_label = input("Override label: ").strip().upper()
        if chosen_label not in LABELS:
            print("Invalid label. Not applied.")
            return

    chosen_target = target

    if chosen_label in {"UPDATE", "MERGE", "FORGET"}:
        if retrieved_targets:
            target_input = input("Target memory_id [blank = auto target if available]: ").strip()
            if target_input:
                chosen_target = get_memory(conn, user_id, target_input)
                if not chosen_target:
                    print("Target not found. Not applied.")
                    return
        elif not chosen_target:
            print("No target available. Not applied.")
            return

    apply_final_action(
        conn, user_id, chosen_label, user_message, candidate, memory_type, sensitivity,
        raw_label, scores, margin, target=chosen_target, notes=f"user_confirmed; {override_reason}; {confirm_reason}; {target_reason}"
    )


def seed_demo(conn, user_id):
    examples = [
        ("User prefers dark mode.", "preference", "low"),
        ("User works on a memory governance project.", "project_state", "low"),
        ("User's old address is stored in memory.", "sensitive", "high"),
        ("User prefers concise technical explanations followed by full scripts.", "preference", "low"),
        ("User is working on MemoryGovBench.", "project_state", "low"),
    ]

    for text, memory_type, sensitivity in examples:
        insert_memory(
            conn, user_id, text, memory_type, sensitivity, "active",
            "seed_demo", {"seed": True}
        )

    print(f"Seeded {len(examples)} demo memories.")


def print_logs(conn, user_id, limit=20):
    rows = conn.execute(
        """
        SELECT * FROM operation_log
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()

    if not rows:
        print("No operation logs.")
        return

    for row in rows:
        row = dict(row)
        print()
        print(f"[{row['created_at']}] raw={row['raw_predicted_label']} final_label={row['final_policy_label']} action={row['final_action']} margin={row['margin']}")
        print(f"  candidate: {row['candidate_memory']}")
        print(f"  target: {row['selected_target_id']}")
        print(f"  notes: {row['notes']}")


def print_help():
    print()
    print("Commands:")
    print("  chat       Process one natural user message")
    print("  context    Retrieve memories for a query without writing")
    print("  list       List active memories")
    print("  all        List all memories")
    print("  quarantine List quarantined memories")
    print("  deleted    List deleted memories")
    print("  logs       Show recent operation logs")
    print("  seed       Add demo seed memories")
    print("  help       Show commands")
    print("  quit       Exit")


def interactive_loop(conn, model, encoder, user_id, margin_threshold):
    print()
    print("Memory Governor CLI v4.1")
    print(f"user_id={user_id}")
    print(f"margin_threshold={margin_threshold}")
    print_help()

    while True:
        try:
            cmd = input("\nmgov> ").strip().lower()
        except KeyboardInterrupt:
            print("\nUse quit to exit cleanly.")
            continue

        if cmd in {"quit", "exit", "q"}:
            break

        if cmd == "help":
            print_help()
            continue

        if cmd == "seed":
            seed_demo(conn, user_id)
            continue

        if cmd == "list":
            print_memories(get_memories(conn, user_id, status="active"))
            continue

        if cmd == "all":
            print_memories(get_memories(conn, user_id, status=None))
            continue

        if cmd == "quarantine":
            print_memories(get_memories(conn, user_id, status="quarantined"))
            continue

        if cmd == "deleted":
            print_memories(get_memories(conn, user_id, status="deleted"))
            continue

        if cmd == "logs":
            print_logs(conn, user_id)
            continue

        if cmd == "context":
            user_message = input("Query: ").strip()
            retrieved = retrieve_for_answer_context(conn, user_id, user_message)
            log_retrieval(conn, user_id, user_message, retrieved)
            print()
            print(build_answer_context(user_message, retrieved))
            continue

        if cmd == "chat":
            user_message = input("User message: ").strip()
            if not user_message:
                print("Empty message.")
                continue

            answer_retrieved = retrieve_for_answer_context(conn, user_id, user_message)
            log_retrieval(conn, user_id, user_message, answer_retrieved)

            print()
            print("=== Answer context that would be sent to the answer model ===")
            print(build_answer_context(user_message, answer_retrieved))
            print("=== End answer context ===")

            candidate = extract_candidate_memory(user_message)

            if not candidate:
                print()
                print("No durable candidate memory extracted. Retrieval-only turn.")
                continue

            memory_type = infer_memory_type(user_message)
            sensitivity = infer_sensitivity(user_message)

            target_retrieved_for_model = retrieve_for_target(conn, user_id, candidate)
            op_input = build_operation_input(user_message, candidate, target_retrieved_for_model, memory_type, sensitivity)

            raw_label, scores = predict_operation(model, encoder, op_input)
            final_label, override_reason = policy_override(raw_label, user_message, candidate, memory_type, sensitivity)

            target_retrieved = refine_targets_for_final_action(
                conn=conn,
                user_id=user_id,
                final_label=final_label,
                user_message=user_message,
                candidate=candidate,
                memory_type=memory_type,
                retrieved_targets=target_retrieved_for_model,
            )

            print()
            print(f"Candidate memory: {candidate}")
            print(f"Inferred metadata: type={memory_type}, sensitivity={sensitivity}")

            confirm_or_apply(
                conn=conn,
                user_id=user_id,
                user_message=user_message,
                candidate=candidate,
                memory_type=memory_type,
                sensitivity=sensitivity,
                raw_label=raw_label,
                final_label=final_label,
                scores=scores,
                retrieved_targets=target_retrieved,
                margin_threshold=margin_threshold,
                override_reason=override_reason,
            )
            continue

        print("Unknown command. Type help.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--user_id", default="demo_user")
    parser.add_argument("--margin_threshold", type=float, default=0.25)
    parser.add_argument("--retrieval_backend", choices=["lexical", "embedding", "hybrid"], default="hybrid")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    global GLOBAL_RETRIEVAL_BACKEND, GLOBAL_EMBEDDER, GLOBAL_EMBEDDING_MODEL

    GLOBAL_RETRIEVAL_BACKEND = args.retrieval_backend
    GLOBAL_EMBEDDING_MODEL = args.embedding_model

    conn = connect(args.db)
    init_db(conn)

    if args.retrieval_backend in {"embedding", "hybrid"}:
        GLOBAL_EMBEDDER = load_embedder(args.embedding_model)
        if GLOBAL_EMBEDDER is None:
            print("Falling back to lexical retrieval.")
            GLOBAL_RETRIEVAL_BACKEND = "lexical"
        else:
            print(f"Loaded embedding retriever: {args.embedding_model}")

    model, encoder = load_model(args.model)
    interactive_loop(conn, model, encoder, args.user_id, args.margin_threshold)


if __name__ == "__main__":
    main()
