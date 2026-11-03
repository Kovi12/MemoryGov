import argparse
import importlib.util
import json
import os
import sqlite3
import threading
import time
import uuid
import warnings
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn


os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")


MODULE_PATH = Path("memory_governor_cli_v4_2.py")
MODEL_PATH = "processed/operation_classifier_results/non_llm_v1_2_forget_aug_selected/tfidf_sgd_hinge_balanced/model.joblib"
DEFAULT_DB = "processed/demo_memory_store_web_final/memories.sqlite"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

APP_STATE = {
    "mgov": None,
    "model": None,
    "encoder": None,
    "conn": None,
    "user_id": "demo_user",
    "margin_threshold": 0.25,
    "pending": {},
    "retrieval_backend": "hybrid",
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "embedding_device": "auto",
}

LOCK = threading.Lock()


def load_module(path):
    spec = importlib.util.spec_from_file_location("mgov", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_embedder(model_name, device="auto"):
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        print(f"sentence_transformers unavailable: {exc}")
        return None

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    try:
        print(f"Loading embedding model: {model_name} on {device}")
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except TypeError:
        try:
            print(f"Loading embedding model: {model_name} on {device}")
            return SentenceTransformer(model_name, device=device)
        except Exception as exc:
            print(f"Could not load embedding model: {exc}")
            return None
    except Exception as exc:
        print(f"Could not load embedding model: {exc}")
        return None


def rowdicts(conn, sql, params=()):
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params)]


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(v) for v in obj]
    return obj


def memory_rows(status="active"):
    conn = APP_STATE["conn"]

    if status == "all":
        return rowdicts(
            conn,
            """
            SELECT memory_id, status, memory_type, sensitivity, text, supersedes_id, created_at, updated_at
            FROM memories
            ORDER BY created_at
            """,
        )

    return rowdicts(
        conn,
        """
        SELECT memory_id, status, memory_type, sensitivity, text, supersedes_id, created_at, updated_at
        FROM memories
        WHERE status = ?
        ORDER BY created_at
        """,
        (status,),
    )


def operation_log_rows():
    return rowdicts(
        APP_STATE["conn"],
        """
        SELECT created_at, raw_predicted_label, final_policy_label, final_action,
               margin, selected_target_id, candidate_memory, notes
        FROM operation_log
        ORDER BY created_at DESC
        LIMIT 200
        """,
    )


def retrieval_log_rows():
    return rowdicts(
        APP_STATE["conn"],
        """
        SELECT created_at, user_message, retrieved_memory_ids
        FROM retrieval_log
        ORDER BY created_at DESC
        LIMIT 200
        """,
    )


def simple_answer_preview(user_message, retrieved):
    if not retrieved:
        return "No relevant active memories were retrieved. A real answer model would answer without memory context."

    lines = ["Retrieved memory context for answer generation:"]
    for idx, mem in enumerate(retrieved, 1):
        lines.append(f"{idx}. {mem.get('text')}")
    lines.append("")
    lines.append("This is a memory-context preview, not a full LLM answer. The future answer model should receive this context and the user message.")
    return "\n".join(lines)


def process_message(user_message):
    mgov = APP_STATE["mgov"]
    model = APP_STATE["model"]
    encoder = APP_STATE["encoder"]
    conn = APP_STATE["conn"]
    user_id = APP_STATE["user_id"]
    margin_threshold = APP_STATE["margin_threshold"]

    t0 = time.perf_counter()

    answer_retrieved = mgov.retrieve_for_answer_context(conn, user_id, user_message)
    mgov.log_retrieval(conn, user_id, user_message, answer_retrieved)

    t_retrieval = time.perf_counter()

    answer_context = mgov.build_answer_context(user_message, answer_retrieved)
    answer_preview = simple_answer_preview(user_message, answer_retrieved)

    candidate = mgov.extract_candidate_memory(user_message)

    result = {
        "user_message": user_message,
        "answer_context": answer_context,
        "answer_preview": answer_preview,
        "answer_retrieved": answer_retrieved,
        "candidate": candidate,
        "mode": "retrieval_only" if not candidate else "operation_candidate",
        "applied": False,
        "pending_id": None,
        "latency": {
            "retrieval_sec": t_retrieval - t0,
            "operation_sec": 0.0,
            "total_sec": time.perf_counter() - t0,
        },
    }

    if not candidate:
        return result

    memory_type = mgov.infer_memory_type(user_message)
    sensitivity = mgov.infer_sensitivity(user_message)

    target_for_model = mgov.retrieve_for_target(conn, user_id, candidate)

    op_input = mgov.build_operation_input(
        user_message,
        candidate,
        target_for_model,
        memory_type,
        sensitivity,
    )

    raw_label, scores = mgov.predict_operation(model, encoder, op_input)

    final_label, override_reason = mgov.policy_override(
        raw_label,
        user_message,
        candidate,
        memory_type,
        sensitivity,
    )

    target_retrieved = mgov.refine_targets_for_final_action(
        conn,
        user_id,
        final_label,
        user_message,
        candidate,
        memory_type,
        target_for_model,
    )

    margin = mgov.score_margin(scores)

    target = None
    target_reason = "not_targeted"

    if final_label in {"UPDATE", "MERGE", "FORGET"}:
        target, target_reason = mgov.choose_target_automatically(target_retrieved)

    confirm, confirm_reason = mgov.requires_confirmation(
        final_label,
        margin,
        margin_threshold,
        target,
        target_reason,
    )

    t_operation = time.perf_counter()

    result.update({
        "memory_type": memory_type,
        "sensitivity": sensitivity,
        "raw_label": raw_label,
        "final_label": final_label,
        "override_reason": override_reason,
        "scores": scores,
        "scores_sorted": [
            {"label": k, "score": float(v)}
            for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ],
        "margin": float(margin),
        "target_retrieved": target_retrieved,
        "target": target,
        "target_reason": target_reason,
        "confirm": bool(confirm),
        "confirm_reason": confirm_reason,
        "latency": {
            "retrieval_sec": t_retrieval - t0,
            "operation_sec": t_operation - t_retrieval,
            "total_sec": t_operation - t0,
        },
    })

    if confirm:
        pending_id = uuid.uuid4().hex[:12]
        APP_STATE["pending"][pending_id] = result
        result["pending_id"] = pending_id
        result["applied"] = False
        result["apply_reason"] = "pending_confirmation"
        return result

    mgov.apply_final_action(
        conn=conn,
        user_id=user_id,
        final_label=final_label,
        user_message=user_message,
        candidate=candidate,
        memory_type=memory_type,
        sensitivity=sensitivity,
        raw_label=raw_label,
        scores=scores,
        margin=margin,
        target=target,
        notes=f"web_auto; {override_reason}; {confirm_reason}; {target_reason}",
    )

    result["applied"] = True
    result["apply_reason"] = "auto_applied"
    return result


def apply_pending(pending_id, final_label, target_id):
    mgov = APP_STATE["mgov"]
    conn = APP_STATE["conn"]
    user_id = APP_STATE["user_id"]

    pending = APP_STATE["pending"].get(pending_id)
    if not pending:
        return {"ok": False, "error": "pending_id not found"}

    target = None

    if final_label in {"UPDATE", "MERGE", "FORGET"}:
        for mem in pending.get("target_retrieved", []):
            if mem["memory_id"] == target_id:
                target = mem
                break

        if target is None:
            return {"ok": False, "error": "target_id required or not found"}

    mgov.apply_final_action(
        conn=conn,
        user_id=user_id,
        final_label=final_label,
        user_message=pending["user_message"],
        candidate=pending["candidate"],
        memory_type=pending["memory_type"],
        sensitivity=pending["sensitivity"],
        raw_label=pending["raw_label"],
        scores=pending["scores"],
        margin=pending["margin"],
        target=target,
        notes=(
            f"web_user_confirmed; original_final={pending['final_label']}; "
            f"{pending['override_reason']}; {pending['confirm_reason']}; {pending['target_reason']}"
        ),
    )

    del APP_STATE["pending"][pending_id]
    return {"ok": True}


def decline_pending(pending_id):
    mgov = APP_STATE["mgov"]
    conn = APP_STATE["conn"]
    user_id = APP_STATE["user_id"]

    pending = APP_STATE["pending"].get(pending_id)
    if not pending:
        return {"ok": False, "error": "pending_id not found"}

    mgov.log_operation(
        conn=conn,
        user_id=user_id,
        user_message=pending["user_message"],
        candidate_memory=pending["candidate"],
        raw_label=pending["raw_label"],
        final_label=pending["final_label"],
        final_action="USER_DECLINED",
        scores=pending["scores"],
        margin=pending["margin"],
        target_id=None,
        notes="web_user_declined",
    )

    del APP_STATE["pending"][pending_id]
    return {"ok": True}


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Memory Governor</title>
  <style>
    :root {
      --bg: #0f1117;
      --panel: #181b24;
      --panel2: #11141b;
      --border: #303442;
      --text: #e8e8e8;
      --muted: #aeb7d0;
      --blue: #2b63d9;
      --green: #72d672;
      --yellow: #ffd166;
      --red: #ff7b72;
    }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 12px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 16px;
    }
    header strong { font-size: 18px; }
    main {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 12px;
      padding: 12px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      min-width: 0;
    }
    h3, h4 { margin: 8px 0; }
    textarea {
      width: 100%;
      height: 95px;
      box-sizing: border-box;
      background: var(--panel2);
      color: var(--text);
      border: 1px solid #444b5c;
      border-radius: 8px;
      padding: 10px;
      resize: vertical;
    }
    button, select {
      background: var(--blue);
      color: white;
      border: none;
      border-radius: 7px;
      padding: 7px 10px;
      margin: 4px 4px 4px 0;
      cursor: pointer;
      font-size: 13px;
    }
    button.secondary { background: #444b5c; }
    button.danger { background: #a33; }
    button:disabled { opacity: 0.5; cursor: default; }
    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 5px;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      text-align: left;
      position: sticky;
      top: 0;
      background: var(--panel);
    }
    pre {
      white-space: pre-wrap;
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px;
      max-height: 210px;
      overflow: auto;
      font-size: 12px;
    }
    .muted { color: var(--muted); }
    .ok { color: var(--green); }
    .warn { color: var(--yellow); }
    .bad { color: var(--red); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .pill {
      display:inline-block;
      padding: 2px 7px;
      background: #303442;
      border-radius: 999px;
      margin: 2px 4px 2px 0;
      font-size: 12px;
      color: var(--muted);
    }
    .scroll {
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .decision {
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px;
      font-size: 13px;
    }
    .latency {
      font-size: 12px;
      color: var(--muted);
      margin-top: 6px;
    }
  </style>
</head>
<body>
<header>
  <strong>Memory Governor</strong>
  <span class="muted">FastAPI browser app - loaded once on FEP</span>
  <span id="health" class="muted"></span>
</header>

<main>
  <section>
    <h3>User message</h3>
    <textarea id="msg" placeholder="Type a user message..."></textarea><br>
    <button onclick="processMessage()">Process</button>
    <button class="secondary" onclick="seed()">Seed demo memories</button>
    <button class="danger" onclick="resetDb()">Reset DB</button>

    <h3>Decision</h3>
    <div id="decision" class="decision muted">No message processed yet.</div>

    <h3>Answer preview</h3>
    <pre id="answerPreview"></pre>

    <h3>Answer context sent to future answer model</h3>
    <pre id="context"></pre>

    <div id="confirmBox" style="display:none;">
      <h3 class="warn">Confirmation required</h3>
      <div>
        <label>Final action:</label>
        <select id="labelSelect">
          <option>WRITE</option>
          <option>IGNORE</option>
          <option>UPDATE</option>
          <option>MERGE</option>
          <option>FORGET</option>
          <option>ASK_USER</option>
          <option>QUARANTINE</option>
        </select>
      </div>
      <div>
        <label>Target:</label>
        <select id="targetSelect"></select>
      </div>
      <button onclick="applyPending()">Apply selected action</button>
      <button class="secondary" onclick="declinePending()">Ignore / do nothing</button>
    </div>
  </section>

  <section>
    <h3>Retrieved memories and scores</h3>
    <div class="grid2">
      <div>
        <h4>Answer-context retrieval</h4>
        <div id="answerRetrieved" class="scroll"></div>
      </div>
      <div>
        <h4>Target-memory retrieval</h4>
        <div id="targetRetrieved" class="scroll"></div>
      </div>
    </div>

    <h4>Operation scores</h4>
    <div id="scores" class="scroll"></div>
  </section>

  <section>
    <h3>Memory store</h3>
    <button class="secondary" onclick="loadMemories('active')">Active</button>
    <button class="secondary" onclick="loadMemories('all')">All</button>
    <button class="secondary" onclick="loadMemories('superseded')">Superseded</button>
    <button class="secondary" onclick="loadMemories('deleted')">Deleted</button>
    <button class="secondary" onclick="loadMemories('quarantined')">Quarantined</button>
    <div id="memories" class="scroll"></div>
  </section>

  <section>
    <h3>Logs</h3>
    <button class="secondary" onclick="loadLogs()">Refresh logs</button>
    <h4>Operation log</h4>
    <div id="oplog" class="scroll"></div>
    <h4>Retrieval log</h4>
    <div id="rlog" class="scroll"></div>
  </section>
</main>

<script>
let lastResult = null;

function esc(x) {
  if (x === null || x === undefined) return "";
  return String(x).replace(/[&<>"']/g, c => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;"
  }[c]));
}

function fmt(x, n=3) {
  if (x === null || x === undefined || x === "") return "";
  const y = Number(x);
  if (Number.isNaN(y)) return esc(x);
  return y.toFixed(n);
}

function table(rows, cols) {
  if (!rows || rows.length === 0) return "<p class='muted'>None</p>";
  let h = "<table><thead><tr>";
  for (const c of cols) h += `<th>${esc(c)}</th>`;
  h += "</tr></thead><tbody>";
  for (const r of rows) {
    h += "<tr>";
    for (const c of cols) h += `<td>${esc(r[c])}</td>`;
    h += "</tr>";
  }
  h += "</tbody></table>";
  return h;
}

function memoryTable(rows) {
  return table(rows || [], ["memory_id", "status", "memory_type", "sensitivity", "score", "backend", "text"]);
}

async function post(url, data={}) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });
  return await res.json();
}

async function get(url) {
  const res = await fetch(url);
  return await res.json();
}

async function health() {
  const out = await get("/api/health");
  document.getElementById("health").textContent =
    `backend=${out.retrieval_backend} threshold=${out.margin_threshold} user=${out.user_id}`;
}

async function processMessage() {
  const msg = document.getElementById("msg").value.trim();
  if (!msg) return;

  document.getElementById("decision").innerHTML = "<span class='muted'>Processing...</span>";

  const out = await post("/api/process", {message: msg});
  if (out.error) {
    document.getElementById("decision").innerHTML = `<span class='bad'>${esc(out.error)}</span>`;
    return;
  }

  lastResult = out;

  document.getElementById("context").textContent = out.answer_context || "";
  document.getElementById("answerPreview").textContent = out.answer_preview || "";

  document.getElementById("answerRetrieved").innerHTML = memoryTable(
    (out.answer_retrieved || []).map(m => ({
      memory_id: m.memory_id,
      status: m.status,
      memory_type: m.memory_type,
      sensitivity: m.sensitivity,
      score: fmt(m.retrieval_score),
      backend: m.retrieval_backend || "",
      text: m.text
    }))
  );

  document.getElementById("targetRetrieved").innerHTML = memoryTable(
    (out.target_retrieved || []).map(m => ({
      memory_id: m.memory_id,
      status: m.status,
      memory_type: m.memory_type,
      sensitivity: m.sensitivity,
      score: fmt(m.retrieval_score),
      backend: m.retrieval_backend || "",
      text: m.text
    }))
  );

  document.getElementById("scores").innerHTML = table(
    (out.scores_sorted || []).map(x => ({label: x.label, score: fmt(x.score, 4)})),
    ["label", "score"]
  );

  const lat = out.latency || {};
  const latHtml = `<div class='latency'>retrieval=${fmt(lat.retrieval_sec, 3)}s operation=${fmt(lat.operation_sec, 3)}s total=${fmt(lat.total_sec, 3)}s</div>`;

  if (out.mode === "retrieval_only") {
    document.getElementById("decision").innerHTML =
      `<span class='ok'>Retrieval-only turn.</span><br>No durable candidate memory extracted.${latHtml}`;
    document.getElementById("confirmBox").style.display = "none";
  } else {
    document.getElementById("decision").innerHTML = `
      <div><span class='pill'>candidate</span>${esc(out.candidate)}</div>
      <div><span class='pill'>type</span>${esc(out.memory_type)} <span class='pill'>sensitivity</span>${esc(out.sensitivity)}</div>
      <div><span class='pill'>raw</span>${esc(out.raw_label)} <span class='pill'>final</span>${esc(out.final_label)}</div>
      <div><span class='pill'>margin</span>${fmt(out.margin, 4)} <span class='pill'>confirm</span>${esc(out.confirm)}</div>
      <div class='muted'>${esc(out.override_reason || "")}</div>
      <div class='muted'>${esc(out.confirm_reason || "")}</div>
      <div class='muted'>${esc(out.apply_reason || "")}</div>
      ${latHtml}
    `;

    if (out.confirm) {
      setupConfirm(out);
    } else {
      document.getElementById("confirmBox").style.display = "none";
    }
  }

  await loadMemories("active");
  await loadLogs();
}

function setupConfirm(out) {
  document.getElementById("confirmBox").style.display = "block";

  const labelSelect = document.getElementById("labelSelect");
  labelSelect.value = out.final_label;

  const targetSelect = document.getElementById("targetSelect");
  targetSelect.innerHTML = "";

  const targets = out.target_retrieved || [];
  for (const m of targets) {
    const opt = document.createElement("option");
    opt.value = m.memory_id;
    opt.textContent = `${m.memory_id} | ${m.text}`;
    targetSelect.appendChild(opt);
  }

  if (out.target && out.target.memory_id) {
    targetSelect.value = out.target.memory_id;
  }
}

async function applyPending() {
  if (!lastResult || !lastResult.pending_id) return;

  const label = document.getElementById("labelSelect").value;
  const target = document.getElementById("targetSelect").value || null;

  const out = await post("/api/apply", {
    pending_id: lastResult.pending_id,
    final_label: label,
    target_id: target
  });

  if (!out.ok) {
    alert(out.error || "Apply failed");
    return;
  }

  document.getElementById("confirmBox").style.display = "none";
  document.getElementById("decision").innerHTML += "<div class='ok'>Applied.</div>";
  await loadMemories("active");
  await loadLogs();
}

async function declinePending() {
  if (!lastResult || !lastResult.pending_id) return;

  const out = await post("/api/decline", {pending_id: lastResult.pending_id});
  if (!out.ok) {
    alert(out.error || "Decline failed");
    return;
  }

  document.getElementById("confirmBox").style.display = "none";
  document.getElementById("decision").innerHTML += "<div class='warn'>Ignored.</div>";
  await loadLogs();
}

async function seed() {
  await post("/api/seed", {});
  await loadMemories("active");
  await loadLogs();
}

async function resetDb() {
  if (!confirm("Reset database?")) return;

  await post("/api/reset", {});
  lastResult = null;
  document.getElementById("decision").innerHTML = "DB reset.";
  document.getElementById("context").textContent = "";
  document.getElementById("answerPreview").textContent = "";
  document.getElementById("answerRetrieved").innerHTML = "";
  document.getElementById("targetRetrieved").innerHTML = "";
  document.getElementById("scores").innerHTML = "";
  document.getElementById("confirmBox").style.display = "none";
  await loadMemories("active");
  await loadLogs();
}

async function loadMemories(status="active") {
  const out = await get(`/api/memories?status=${status}`);
  document.getElementById("memories").innerHTML = table(out.memories || [], [
    "memory_id", "status", "memory_type", "sensitivity", "text", "supersedes_id"
  ]);
}

async function loadLogs() {
  const out = await get("/api/logs");
  document.getElementById("oplog").innerHTML = table(out.operation_log || [], [
    "created_at", "raw_predicted_label", "final_policy_label", "final_action", "margin", "selected_target_id", "candidate_memory"
  ]);
  document.getElementById("rlog").innerHTML = table(out.retrieval_log || [], [
    "created_at", "user_message", "retrieved_memory_ids"
  ]);
}

health();
loadMemories("active");
loadLogs();
</script>
</body>
</html>
"""

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/health")
def api_health():
    mgov = APP_STATE["mgov"]
    return {
        "ok": True,
        "retrieval_backend": mgov.GLOBAL_RETRIEVAL_BACKEND,
        "user_id": APP_STATE["user_id"],
        "margin_threshold": APP_STATE["margin_threshold"],
        "embedding_model": APP_STATE["embedding_model"],
        "embedding_device": APP_STATE["embedding_device"],
    }


@app.post("/api/process")
async def api_process(request: Request):
    data = await request.json()
    message = (data.get("message") or "").strip()

    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)

    with LOCK:
        out = process_message(message)

    return make_jsonable(out)


@app.post("/api/apply")
async def api_apply(request: Request):
    data = await request.json()

    pending_id = data.get("pending_id")
    final_label = data.get("final_label")
    target_id = data.get("target_id")

    with LOCK:
        out = apply_pending(pending_id, final_label, target_id)

    return out


@app.post("/api/decline")
async def api_decline(request: Request):
    data = await request.json()
    pending_id = data.get("pending_id")

    with LOCK:
        out = decline_pending(pending_id)

    return out


@app.post("/api/seed")
def api_seed():
    with LOCK:
        APP_STATE["mgov"].seed_demo(APP_STATE["conn"], APP_STATE["user_id"])
    return {"ok": True}


@app.post("/api/reset")
def api_reset():
    conn = APP_STATE["conn"]

    with LOCK:
        conn.execute("DELETE FROM memory_embeddings")
        conn.execute("DELETE FROM retrieval_log")
        conn.execute("DELETE FROM operation_log")
        conn.execute("DELETE FROM memories")
        conn.commit()
        APP_STATE["pending"].clear()

    return {"ok": True}


@app.get("/api/memories")
def api_memories(status: str = "active"):
    with LOCK:
        return {"memories": memory_rows(status)}


@app.get("/api/logs")
def api_logs():
    with LOCK:
        return {
            "operation_log": operation_log_rows(),
            "retrieval_log": retrieval_log_rows(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--user_id", default="demo_user")
    parser.add_argument("--margin_threshold", type=float, default=0.25)
    parser.add_argument("--retrieval_backend", choices=["lexical", "embedding", "hybrid"], default="hybrid")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()

    mgov = load_module(MODULE_PATH)

    mgov.GLOBAL_RETRIEVAL_BACKEND = args.retrieval_backend
    mgov.GLOBAL_EMBEDDING_MODEL = args.embedding_model

    if args.retrieval_backend in {"embedding", "hybrid"}:
        mgov.GLOBAL_EMBEDDER = load_embedder(args.embedding_model, args.embedding_device)
        if mgov.GLOBAL_EMBEDDER is None:
            print("Embedding unavailable. Falling back to lexical retrieval.")
            mgov.GLOBAL_RETRIEVAL_BACKEND = "lexical"

    print("Loading Operation Governor...")
    model, encoder = mgov.load_model(MODEL_PATH)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    mgov.init_db(conn)

    APP_STATE.update({
        "mgov": mgov,
        "model": model,
        "encoder": encoder,
        "conn": conn,
        "user_id": args.user_id,
        "margin_threshold": args.margin_threshold,
        "pending": {},
        "retrieval_backend": mgov.GLOBAL_RETRIEVAL_BACKEND,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
    })

    print()
    print("Memory Governor Web Final")
    print(f"db={args.db}")
    print(f"user_id={args.user_id}")
    print(f"retrieval_backend={mgov.GLOBAL_RETRIEVAL_BACKEND}")
    print(f"margin_threshold={args.margin_threshold}")
    print(f"url=http://{args.host}:{args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
