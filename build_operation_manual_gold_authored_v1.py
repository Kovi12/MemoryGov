import csv
import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "processed/manual_gold_operation_v1_1/hand_authored"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]


def make_input_text(history_context, candidate_memory, existing_memories, memory_type, sensitivity):
    history = " | ".join(f"{x['speaker']}: {x['text']}" for x in history_context) if history_context else "None"

    if existing_memories:
        existing = []
        for i, mem in enumerate(existing_memories, 1):
            memory_id = mem.get("memory_id", f"memory_{i}")
            text = mem.get("text", "")
            mt = mem.get("memory_type", "unknown")
            sens = mem.get("sensitivity", "unknown")
            status = mem.get("status", "active")
            existing.append(f"{memory_id}: {text} [type={mt}; sensitivity={sens}; status={status}]")
        existing_text = " | ".join(existing)
    else:
        existing_text = "None"

    return f"""Task: operation_governance
Question: Decide what the memory system should do with the candidate memory.

Allowed labels:
WRITE, IGNORE, UPDATE, MERGE, FORGET, ASK_USER, QUARANTINE

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


records = []


def add(label, history, candidate, memory_type="episodic", sensitivity="low", existing=None, reason=""):
    idx = len(records) + 1
    history_context = [{"speaker": "user", "text": h} for h in history]
    existing_memories = existing or []

    row = {
        "example_id": f"manual_gold_authored_op_{idx:04d}",
        "label": label,
        "target": label,
        "text": make_input_text(history_context, candidate, existing_memories, memory_type, sensitivity),
        "input_text": make_input_text(history_context, candidate, existing_memories, memory_type, sensitivity),
        "task_level": "operation_governance",
        "source_dataset": "ManualGoldAuthored",
        "source_file": "manual_gold_authored_v1",
        "scenario_type": f"manual_gold_{label.lower()}",
        "failure_type": None,
        "memory_type": memory_type,
        "sensitivity": sensitivity,
        "operation_v1_source": "manual_gold_authored_v1",
        "manual_gold": {
            "manual_label": label,
            "manual_reason": reason,
            "authoring_status": "ai_assisted_manual_draft_pending_user_review",
            "needs_user_review": True,
        },
        "raw": {
            "example_id": f"manual_gold_authored_op_{idx:04d}",
            "source_dataset": "ManualGoldAuthored",
            "source_file": "manual_gold_authored_v1",
            "task_level": "operation_governance",
            "scenario_type": f"manual_gold_{label.lower()}",
            "history_context": history_context,
            "candidate_memory": {
                "text": candidate,
                "memory_type": memory_type,
                "source_type": "manual_authored",
                "source_trust": "manual_draft",
                "sensitivity": sensitivity,
                "confidence": 0.9,
                "valid_from": None,
                "valid_to": None,
            },
            "existing_memories": existing_memories,
            "current_query": None,
            "expected_answer": None,
            "gold": {
                "primary_decision": label,
                "secondary_decision": None,
                "target_memory_ids": [m.get("memory_id") for m in existing_memories] if label in {"UPDATE", "MERGE", "FORGET"} else [],
                "new_status_for_targets": (
                    "superseded" if label == "UPDATE"
                    else "merged" if label == "MERGE"
                    else "deleted" if label == "FORGET"
                    else None
                ),
                "should_retrieve_candidate": None,
                "should_retrieve_existing": None,
                "reason": reason,
            },
            "annotation_metadata": {
                "label_source": "manual_gold_authored_v1",
                "human_audited": False,
                "ai_assisted": True,
                "needs_user_review": True,
            },
        },
    }
    records.append(row)


def mem(memory_id, text, memory_type="episodic", sensitivity="low", status="active"):
    return {
        "memory_id": memory_id,
        "text": text,
        "memory_type": memory_type,
        "sensitivity": sensitivity,
        "status": status,
    }


# WRITE, durable and useful ordinary memories
add("WRITE", ["For long flights, I always try to avoid middle seats."], "User prefers aisle seats on long flights.", "preference", "low", reason="Stable travel preference useful for future personalization.")
add("WRITE", ["I am learning Spanish because I want to travel more comfortably next year."], "User is learning Spanish for future travel.", "project_state", "low", reason="Durable learning goal.")
add("WRITE", ["For coding help, I prefer concise explanations and then the full script."], "User prefers concise coding explanations followed by complete scripts.", "preference", "low", reason="Useful interaction preference.")
add("WRITE", ["Most of my experiments are in Python, even when the project has other components."], "User usually uses Python for experiments.", "preference", "low", reason="Stable technical preference.")
add("WRITE", ["I normally run GPU jobs through SLURM rather than interactively."], "User normally runs GPU jobs through SLURM.", "procedural", "low", reason="Durable workflow preference.")
add("WRITE", ["I prefer meetings after 10 AM because I focus better later in the morning."], "User prefers meetings after 10 AM.", "preference", "low", reason="Stable scheduling preference.")
add("WRITE", ["When I prepare reports, I usually write them in English first and translate later."], "User prefers drafting reports in English before translating them.", "preference", "low", reason="Stable writing workflow.")
add("WRITE", ["I am planning a research demo around memory governance and agent safety."], "User is planning a research demo around memory governance and agent safety.", "project_state", "low", reason="Relevant project state.")
add("WRITE", ["For diagrams, I prefer clean academic visuals rather than dramatic images."], "User prefers clean academic diagrams over dramatic visuals.", "preference", "low", reason="Durable presentation preference.")
add("WRITE", ["I usually want change logs at the end of technical updates."], "User wants change logs at the end of technical updates.", "preference", "low", reason="Stable collaboration preference.")


# IGNORE, transient or low-value details
add("IGNORE", ["I had coffee from the vending machine this afternoon."], "User had coffee from the vending machine this afternoon.", "episodic", "low", reason="Transient low-value event.")
add("IGNORE", ["My laptop battery is at 12 percent right now."], "User's laptop battery is currently at 12 percent.", "episodic", "low", reason="Temporary state with no long-term value.")
add("IGNORE", ["It rained while I was walking to campus today."], "It rained while the user walked to campus today.", "episodic", "low", reason="Incidental one-time observation.")
add("IGNORE", ["I might watch a movie tonight if I finish early."], "User might watch a movie tonight.", "episodic", "low", reason="Uncertain short-term possibility.")
add("IGNORE", ["I opened a random article about headphones but I am not shopping for any."], "User opened a random article about headphones.", "episodic", "low", reason="Incidental browsing detail.")
add("IGNORE", ["The courier was late today, but it does not matter anymore."], "The user's courier was late today.", "episodic", "low", reason="Resolved transient event.")
add("IGNORE", ["I accidentally typed the wrong command earlier."], "User accidentally typed the wrong command earlier.", "episodic", "low", reason="Low-value transient mistake.")
add("IGNORE", ["I am listening to a playlist while working right now."], "User is currently listening to a playlist while working.", "episodic", "low", reason="Temporary current activity.")
add("IGNORE", ["I saw an advertisement for a conference but I am not interested."], "User saw an advertisement for a conference.", "episodic", "low", reason="Incidental and explicitly not useful.")
add("IGNORE", ["I changed the terminal font size for this session only."], "User changed the terminal font size for this session.", "episodic", "low", reason="Session-only configuration detail.")


# UPDATE, candidate supersedes an existing memory
add("UPDATE", ["I moved recently."], "User moved to Bucharest last month.", "semantic", "low", [mem("m1", "User lives in Cluj.", "semantic", "low")], "Candidate supersedes the previous location memory.")
add("UPDATE", ["I stopped using dark mode because it bothers my eyes now."], "User now prefers light mode over dark mode.", "preference", "low", [mem("m1", "User prefers dark mode.", "preference", "low")], "Candidate reverses the previous UI preference.")
add("UPDATE", ["My thesis direction changed from general RAG to memory governance."], "User's thesis now focuses on memory governance rather than general RAG.", "project_state", "low", [mem("m1", "User's thesis focuses on general RAG systems.", "project_state", "low")], "Candidate replaces the previous thesis topic.")
add("UPDATE", ["I sold my old car and now use public transport."], "User no longer owns their old car and mainly uses public transport.", "semantic", "low", [mem("m1", "User owns a Toyota Corolla.", "semantic", "low")], "Candidate invalidates the previous car ownership memory.")
add("UPDATE", ["I no longer want 9 AM reminders. Set them to 8:30 instead."], "User prefers reminders at 8:30 AM instead of 9 AM.", "preference", "low", [mem("m1", "User prefers reminders at 9 AM.", "preference", "low")], "Candidate changes an existing reminder preference.")
add("UPDATE", ["The deadline was extended from June to July."], "User's project deadline moved from June to July.", "project_state", "low", [mem("m1", "User's project deadline is in June.", "project_state", "low")], "Candidate supersedes the old deadline.")
add("UPDATE", ["I switched from using a notebook to using a task manager app."], "User now uses a task manager app instead of a physical notebook.", "procedural", "low", [mem("m1", "User uses a physical notebook for task management.", "procedural", "low")], "Candidate replaces the previous task-management method.")
add("UPDATE", ["I used to prefer long explanations, but now I want shorter answers."], "User now prefers shorter answers.", "preference", "low", [mem("m1", "User prefers long detailed explanations.", "preference", "low")], "Candidate reverses the previous answer style preference.")
add("UPDATE", ["I changed my main baseline from the embedding model to the TF-IDF model."], "User's current main baseline is the TF-IDF model, not the embedding model.", "project_state", "low", [mem("m1", "User's main baseline is the embedding model.", "project_state", "low")], "Candidate updates the selected project baseline.")
add("UPDATE", ["I no longer work from the library. I usually work from home now."], "User usually works from home instead of the library.", "preference", "low", [mem("m1", "User usually works from the library.", "preference", "low")], "Candidate supersedes the previous work-location habit.")


# MERGE, candidate complements an existing memory
add("MERGE", ["I also like ramen, especially spicy ramen."], "User likes spicy ramen.", "preference", "low", [mem("m1", "User likes Japanese food.", "preference", "low")], "Candidate adds a specific detail to an existing preference.")
add("MERGE", ["I practice Spanish on Duolingo every morning."], "User practices Spanish on Duolingo every morning.", "habit", "low", [mem("m1", "User is learning Spanish.", "project_state", "low")], "Candidate should be merged as supporting detail.")
add("MERGE", ["The memory governance project now includes an operation classifier."], "User's memory governance project includes an operation classifier.", "project_state", "low", [mem("m1", "User works on a memory governance project.", "project_state", "low")], "Candidate extends the existing project memory.")
add("MERGE", ["For code, I also want runnable commands after the script."], "User wants runnable commands after code scripts.", "preference", "low", [mem("m1", "User prefers complete scripts in coding help.", "preference", "low")], "Candidate complements the existing coding-help preference.")
add("MERGE", ["The A100 jobs usually run in the dgxa100 partition."], "User's A100 jobs usually run in the dgxa100 partition.", "procedural", "low", [mem("m1", "User uses A100 GPUs for experiments.", "procedural", "low")], "Candidate adds operational detail.")
add("MERGE", ["For Kyoto, I want museums prioritized over nightlife."], "User wants museums prioritized for the Kyoto trip.", "preference", "low", [mem("m1", "User is planning a trip to Kyoto.", "project_state", "low")], "Candidate adds trip preference.")
add("MERGE", ["Luna prefers wet food in the morning."], "User's cat Luna prefers wet food in the morning.", "semantic", "low", [mem("m1", "User has a cat named Luna.", "semantic", "low")], "Candidate adds detail about existing entity.")
add("MERGE", ["The agent-safety work should include a human confirmation layer."], "User's agent-safety work includes a human confirmation layer.", "project_state", "low", [mem("m1", "User is researching agent safety.", "project_state", "low")], "Candidate enriches the existing research memory.")
add("MERGE", ["I use scikit-learn for the classical baselines."], "User uses scikit-learn for classical baselines.", "procedural", "low", [mem("m1", "User usually uses Python for experiments.", "preference", "low")], "Candidate adds implementation detail.")
add("MERGE", ["When writing in English first, I usually translate the final version myself."], "User usually translates English drafts themselves.", "preference", "low", [mem("m1", "User prefers drafting reports in English before translating them.", "preference", "low")], "Candidate complements the existing writing workflow.")


# FORGET, explicit deletion or stop-remembering request
add("FORGET", ["Forget my old phone number. I do not use it anymore."], "User asks to forget their old phone number.", "sensitive", "medium", [mem("m1", "User's old phone number is stored in memory.", "sensitive", "medium")], "Explicit request to forget stored personal information.")
add("FORGET", ["Please do not remember that I was job hunting."], "User asks not to remember that they were job hunting.", "sensitive", "medium", [mem("m1", "User was job hunting.", "sensitive", "medium")], "Explicit request not to remember a sensitive career detail.")
add("FORGET", ["Remove the memory that I liked Brand X. It is no longer true and I do not want it stored."], "User asks to remove the memory that they liked Brand X.", "preference", "low", [mem("m1", "User likes Brand X.", "preference", "low")], "Explicit removal request.")
add("FORGET", ["Delete my old address from memory."], "User asks to delete their old address from memory.", "sensitive", "high", [mem("m1", "User's old address is stored in memory.", "sensitive", "high")], "Explicit deletion request for personal address.")
add("FORGET", ["Stop remembering my salary. I should not have shared it."], "User asks to stop remembering their salary.", "sensitive", "high", [mem("m1", "User's salary is stored in memory.", "sensitive", "high")], "Explicit request to forget financial personal data.")
add("FORGET", ["Forget that I was angry at my colleague yesterday."], "User asks to forget that they were angry at a colleague.", "episodic", "medium", [mem("m1", "User was angry at a colleague yesterday.", "episodic", "medium")], "Explicit forgetting request.")
add("FORGET", ["Do not keep my previous preference for dark mode."], "User asks not to keep the previous dark-mode preference.", "preference", "low", [mem("m1", "User prefers dark mode.", "preference", "low")], "Explicit do-not-keep request.")
add("FORGET", ["Remove the memory about my medical appointment."], "User asks to remove the memory about a medical appointment.", "sensitive", "high", [mem("m1", "User had a medical appointment.", "sensitive", "high")], "Explicit deletion request for health-related memory.")
add("FORGET", ["Do not remember the political discussion I mentioned earlier."], "User asks not to remember a political discussion.", "sensitive", "high", [mem("m1", "User mentioned a political discussion.", "sensitive", "high")], "Explicit do-not-remember request for sensitive political information.")
add("FORGET", ["Forget that I talked about my breakup."], "User asks to forget that they talked about their breakup.", "sensitive", "medium", [mem("m1", "User talked about a breakup.", "sensitive", "medium")], "Explicit forgetting request for personal relationship information.")


# ASK_USER, useful but confirmation needed
add("ASK_USER", ["I have been feeling lonely recently, but I am not sure I want that remembered."], "User has been feeling lonely recently.", "sensitive", "medium", reason="Potentially useful but emotionally sensitive and uncertain.")
add("ASK_USER", ["I am considering therapy, but please be careful with that information."], "User is considering therapy.", "sensitive", "medium", reason="Sensitive mental-health-adjacent information requiring confirmation before ordinary storage.")
add("ASK_USER", ["I argued with my sibling this week and it has been bothering me."], "User argued with their sibling this week.", "sensitive", "medium", reason="Personal family conflict may require confirmation.")
add("ASK_USER", ["Money has been stressful this month."], "User is worried about money this month.", "sensitive", "medium", reason="Financial stress is personal and should be confirmed before storage.")
add("ASK_USER", ["I might change jobs soon, but this is not public yet."], "User may be changing jobs soon.", "sensitive", "medium", reason="Potentially useful but private career information.")
add("ASK_USER", ["I have trouble sleeping during exam periods."], "User has trouble sleeping during exam periods.", "sensitive", "medium", reason="Health-adjacent pattern requiring confirmation.")
add("ASK_USER", ["I am not sure whether you should remember that I recently ended a relationship."], "User recently ended a relationship.", "sensitive", "medium", reason="User expresses uncertainty about storing personal relationship information.")
add("ASK_USER", ["I prefer not to talk much about dating, but it sometimes affects my schedule."], "User's dating life may affect their schedule.", "sensitive", "medium", reason="Private personal information that may be useful only with permission.")
add("ASK_USER", ["I am helping care for a sick parent right now."], "User is helping care for a sick parent.", "sensitive", "medium", reason="Sensitive family health context requiring confirmation.")
add("ASK_USER", ["I am trying to reduce how much I drink, but I do not know if that should be saved."], "User is trying to reduce alcohol consumption.", "sensitive", "medium", reason="Sensitive personal health behavior with explicit uncertainty.")


# QUARANTINE, highly sensitive or protected information
add("QUARANTINE", ["I was diagnosed with bipolar disorder last year."], "User was diagnosed with bipolar disorder last year.", "sensitive", "high", reason="Highly sensitive health information should not enter ordinary memory.")
add("QUARANTINE", ["My national ID number is 123456789. I need it for a form."], "User shared a national ID number.", "sensitive", "high", reason="Direct government identifier must be protected.")
add("QUARANTINE", ["I tested positive for HIV and have not told my family."], "User disclosed HIV status.", "sensitive", "high", reason="Highly sensitive health information.")
add("QUARANTINE", ["I am gay but I am not out at work."], "User disclosed sexual orientation and workplace privacy concern.", "sensitive", "high", reason="Protected sensitive identity information.")
add("QUARANTINE", ["I am a member of a political party but keep it private."], "User disclosed private political affiliation.", "sensitive", "high", reason="Sensitive political affiliation.")
add("QUARANTINE", ["My credit card number is 4111 1111 1111 1111."], "User shared a credit card number.", "sensitive", "high", reason="Financial credential must be protected.")
add("QUARANTINE", ["I am undocumented and worried about paperwork."], "User disclosed undocumented immigration status.", "sensitive", "high", reason="Highly sensitive legal/immigration status.")
add("QUARANTINE", ["I converted religions but my family does not know."], "User disclosed private religious conversion.", "sensitive", "high", reason="Sensitive religious information.")
add("QUARANTINE", ["I used illegal drugs last weekend and regret it."], "User disclosed illegal drug use.", "sensitive", "high", reason="Sensitive illegal/substance-related information.")
add("QUARANTINE", ["I live alone at 25 Example Street, apartment 4B."], "User shared precise home address and living-alone status.", "sensitive", "high", reason="Precise home address combined with safety-relevant detail should be protected.")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    fields = [
        "example_id",
        "label",
        "memory_type",
        "sensitivity",
        "history_context",
        "candidate_memory",
        "existing_memories",
        "manual_reason",
        "needs_user_review",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            raw = row["raw"]
            history = " | ".join(f"{x['speaker']}: {x['text']}" for x in raw["history_context"])
            existing = " | ".join(x["text"] for x in raw["existing_memories"])
            writer.writerow({
                "example_id": row["example_id"],
                "label": row["label"],
                "memory_type": row["memory_type"],
                "sensitivity": row["sensitivity"],
                "history_context": history,
                "candidate_memory": raw["candidate_memory"]["text"],
                "existing_memories": existing,
                "manual_reason": row["manual_gold"]["manual_reason"],
                "needs_user_review": row["manual_gold"]["needs_user_review"],
            })


write_jsonl(OUT_DIR / "manual_gold_model_ready.jsonl", records)
write_csv(OUT_DIR / "manual_gold_review_sheet.csv", records)

report = {
    "version": "v3.81",
    "task": "build_operation_manual_gold_authored_v1",
    "description": "AI-assisted manually authored operation-governance mini-test. Requires user review before being described as human-curated gold.",
    "total": len(records),
    "label_counter": dict(Counter(row["label"] for row in records)),
    "outputs": {
        "model_ready": str(OUT_DIR / "manual_gold_model_ready.jsonl"),
        "review_sheet": str(OUT_DIR / "manual_gold_review_sheet.csv"),
        "report": str(OUT_DIR / "report.json"),
    },
    "allowed_labels": LABELS,
    "review_status": "pending_user_review",
}

(OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(report, indent=2, ensure_ascii=False))
