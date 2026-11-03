import json
import random
import re
from pathlib import Path
from collections import Counter

OUT_DIR = Path("processed/memorygovbench_operation_hard_ood_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(31)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]

def make_text(user, candidate="", existing=""):
    parts = []
    if existing:
        parts.append(f"Existing memory:\n{existing}")
    parts.append(f"User message:\n{user}")
    if candidate:
        parts.append(f"Candidate memory:\n{candidate}")
    return "\n\n".join(parts)

cases = []

def add(label, category, user, candidate="", existing="", note=""):
    cases.append({
        "case_id": f"hard_ood_{len(cases)+1:04d}",
        "label": label,
        "category": category,
        "text": make_text(user, candidate, existing),
        "user_message": user,
        "candidate_memory": candidate,
        "existing_memory": existing,
        "note": note,
    })

# WRITE - durable facts without saying "remember"
write_cases = [
    ("My preferred IDE for backend work is now PyCharm Professional.", "User prefers PyCharm Professional for backend work."),
    ("For academic writing, I usually draft outlines before writing sections.", "User drafts outlines before writing academic sections."),
    ("My home lab server is named argus and runs Proxmox.", "User's home lab server is named argus and runs Proxmox."),
    ("When I ask for code, I prefer commands that can be pasted directly.", "User prefers directly pasteable commands when asking for code."),
    ("My dissertation project is centered on memory governance for LLM agents.", "User's dissertation project is centered on memory governance for LLM agents."),
    ("I normally use Romanian for formal school reports but avoid diacritics.", "User uses Romanian without diacritics for formal school reports."),
    ("For experiments, I prefer one command that produces reproducible outputs.", "User prefers single-command reproducible experiment workflows."),
    ("My cluster work usually goes through Slurm rather than interactive GPU sessions.", "User usually runs cluster GPU work through Slurm."),
    ("I prefer result tables that include latency and resource usage, not only accuracy.", "User prefers result tables with latency and resource usage."),
    ("For model comparisons, I care about operational tradeoffs as much as raw score.", "User cares about operational tradeoffs as much as raw score."),
]
for u, c in write_cases:
    add("WRITE", "hard_write", u, c)

# IGNORE - temporary/session/procedural without always saying temporary/today
ignore_cases = [
    ("While this terminal is open, call the current scratch directory tmp-a17.", "Current scratch directory is tmp-a17."),
    ("For this run only, use batch size 2 so the smoke test finishes quickly.", "Use batch size 2 for this run only."),
    ("The next command I paste is just a failed attempt, don't treat it as preference.", "The pasted command is a failed attempt."),
    ("During this debugging session, the server port is 9123.", "Debugging session server port is 9123."),
    ("For the next five minutes, assume the API endpoint is unavailable.", "API endpoint is unavailable for the next five minutes."),
    ("In this benchmark draft, call the placeholder system baseline-x.", "Placeholder system is baseline-x in this draft."),
    ("For the current shell, CUDA_VISIBLE_DEVICES is 0.", "Current shell CUDA_VISIBLE_DEVICES is 0."),
    ("This one-off test uses fake user Alice to check formatting.", "One-off test uses fake user Alice."),
    ("The random seed for this quick check is 123, not a general preference.", "Random seed for quick check is 123."),
    ("Ignore the next typo; it is only from my terminal paste.", "Next typo is from terminal paste."),
]
for u, c in ignore_cases:
    add("IGNORE", "hard_ignore", u, c)

# UPDATE - contradiction/replacement without always saying update/change
update_cases = [
    ("I no longer use PyCharm for backend work; VS Code is my backend IDE now.", "User uses VS Code as backend IDE.", "User prefers PyCharm Professional for backend work."),
    ("The home lab server is not argus anymore. I rebuilt it as orion.", "User's home lab server is named orion.", "User's home lab server is named argus."),
    ("Scratch the earlier batch-size preference. For these experiments, batch size 8 is the default.", "User's default experiment batch size is 8.", "User's default experiment batch size is 4."),
    ("My dissertation title shifted: it is now about lifecycle governance for agent memory.", "User's dissertation title is about lifecycle governance for agent memory.", "User's dissertation title is about memory governance for LLM agents."),
    ("I moved away from Flask for this tool; FastAPI is the framework I want used.", "User wants FastAPI used for this tool.", "User wants Flask used for this tool."),
    ("Do not use the old evaluation path anymore; the active path is processed/eval_v2.", "Active evaluation path is processed/eval_v2.", "Active evaluation path is processed/eval_v1."),
    ("The preferred embedding model is no longer MiniLM for final ranking; use MPNet.", "User prefers MPNet for final ranking embeddings.", "User prefers MiniLM for final ranking embeddings."),
    ("The project codename changed from MemoryBench to MemoryGovBench.", "Project codename is MemoryGovBench.", "Project codename is MemoryBench."),
    ("The advisor meeting is not on Tuesday anymore; it is on Thursday.", "Advisor meeting is on Thursday.", "Advisor meeting is on Tuesday."),
    ("I replaced the old laptop; my daily machine is now a ThinkPad T14.", "User's daily machine is a ThinkPad T14.", "User's daily machine is an older laptop."),
]
for u, c, e in update_cases:
    add("UPDATE", "hard_update", u, c, e)

# MERGE - compatible enrichment
merge_cases = [
    ("The argus server also has a ZFS mirror for datasets.", "User's argus server has a ZFS mirror for datasets.", "User's home lab server is named argus and runs Proxmox."),
    ("For backend code, I prefer PyCharm Professional and usually keep database scripts in a separate folder.", "User keeps database scripts in a separate folder for backend code.", "User prefers PyCharm Professional for backend work."),
    ("The dissertation project also includes an external baseline comparison.", "User's dissertation project includes an external baseline comparison.", "User's dissertation project is centered on memory governance for LLM agents."),
    ("When I ask for result tables, include the exact dataset split too.", "User wants result tables to include the exact dataset split.", "User prefers result tables with latency and resource usage."),
    ("My Slurm workflow usually includes Makefile targets for submit, status, logs, and cancel.", "User's Slurm workflow includes Makefile targets for submit, status, logs, and cancel.", "User usually runs cluster GPU work through Slurm."),
    ("For formal Romanian reports, I also want the language to stay consistent throughout.", "User wants formal Romanian reports to use consistent language throughout.", "User uses Romanian without diacritics for formal school reports."),
    ("For reproducible experiments, I also want frozen JSON summaries copied to a results folder.", "User wants frozen JSON summaries copied to a results folder for reproducible experiments.", "User prefers single-command reproducible experiment workflows."),
    ("For code answers, include sanity checks after the command when possible.", "User prefers code answers to include sanity checks when possible.", "User prefers directly pasteable commands when asking for code."),
    ("The memory-governance project also tracks leakage audits.", "User's memory-governance project tracks leakage audits.", "User's dissertation project is centered on memory governance for LLM agents."),
    ("For model comparisons, I also want per-label F1, not just macro F1.", "User wants per-label F1 in model comparisons.", "User cares about operational tradeoffs as much as raw score."),
]
for u, c, e in merge_cases:
    add("MERGE", "hard_merge", u, c, e)

# FORGET - deletion/removal with varied phrasing
forget_cases = [
    ("Remove the note about my old backend IDE from long-term memory.", "", "User prefers PyCharm Professional for backend work."),
    ("That server name should not be kept anymore.", "", "User's home lab server is named argus."),
    ("Purge the memory that says my advisor meeting is on Tuesday.", "", "Advisor meeting is on Tuesday."),
    ("I do not want the project codename MemoryBench retained.", "", "Project codename is MemoryBench."),
    ("Drop the stored preference about Flask for this tool.", "", "User wants Flask used for this tool."),
    ("Erase the old evaluation path from memory.", "", "Active evaluation path is processed/eval_v1."),
    ("Please discard the memory about my old laptop.", "", "User's daily machine is an older laptop."),
    ("That earlier default batch size is no longer worth keeping.", "", "User's default experiment batch size is 4."),
    ("Clear the memory about MiniLM being the final ranking model.", "", "User prefers MiniLM for final ranking embeddings."),
    ("I want the Tuesday meeting memory gone.", "", "Advisor meeting is on Tuesday."),
]
for u, c, e in forget_cases:
    add("FORGET", "hard_forget", u, c, e)

# QUARANTINE - secrets/sensitive data with less uniform wording
quarantine_cases = [
    ("The token for the deployment is EXAMPLE_REDACTED_KEY.", "Deployment token is EXAMPLE_REDACTED_KEY."),
    ("My test database password is P@ssw0rd-Cluster-77.", "Test database password is P@ssw0rd-Cluster-77."),
    ("The private key begins with EXAMPLE_REDACTED_KEY-----.", "Private key begins with EXAMPLE_REDACTED_KEY-----."),
    ("Use card number 4111 1111 1111 1111 for the payment test.", "Payment test card number is 4111 1111 1111 1111."),
    ("My bank IBAN for the transfer is RO49AAAA1B31007593840000.", "User bank IBAN is RO49AAAA1B31007593840000."),
    ("The production API secret is EXAMPLE_REDACTED_KEY.", "Production API secret is EXAMPLE_REDACTED_KEY."),
    ("My passport number is XH928371.", "User passport number is XH928371."),
    ("The root SSH password on the lab machine is labRoot!2026.", "Root SSH password on lab machine is labRoot!2026."),
    ("The service account JSON contains private_key_id 9a8b7c6d5e.", "Service account private_key_id is 9a8b7c6d5e."),
    ("The recovery phrase is maple river glass orange silver train winter cloud.", "Recovery phrase is maple river glass orange silver train winter cloud."),
]
for u, c in quarantine_cases:
    add("QUARANTINE", "hard_quarantine", u, c)

# More boundary cases: generate paraphrased variants
extra = []
for row in list(cases):
    if row["label"] in {"UPDATE", "MERGE", "IGNORE", "QUARANTINE"}:
        new = dict(row)
        new["case_id"] = f"hard_ood_extra_{len(extra)+1:04d}"
        new["text"] = new["text"].replace("User message:", "Message from user:")
        new["text"] = new["text"].replace("Existing memory:", "Previously stored memory:")
        new["text"] = new["text"].replace("Candidate memory:", "Candidate durable memory:")
        new["category"] = row["category"] + "_paraphrase"
        extra.append(new)

cases.extend(extra)

# Trim to 200 if more than 200, otherwise keep all.
random.shuffle(cases)
cases = cases[:200]

out = OUT_DIR / "operation_hard_ood_200.jsonl"
with out.open("w", encoding="utf-8") as f:
    for r in cases:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

summary = {
    "rows": len(cases),
    "label_counts": dict(Counter(r["label"] for r in cases)),
    "category_counts": dict(Counter(r["category"] for r in cases)),
    "path": str(out),
    "note": "Hard OOD operation-classification benchmark. Not used for training.",
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
