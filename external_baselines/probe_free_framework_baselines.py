import importlib.util
import json
import shutil
import subprocess
import sys

MODULES = [
    "langgraph",
    "langchain",
    "letta",
    "memgpt",
    "graphiti_core",
    "graphiti",
    "cognee",
    "chromadb",
    "faiss",
    "sentence_transformers",
]

COMMANDS = [
    "python",
    "pip",
    "git",
    "docker",
    "neo4j",
]

def mod_status(name):
    spec = importlib.util.find_spec(name)
    return {
        "module": name,
        "installed": spec is not None,
        "origin": getattr(spec, "origin", None) if spec else None,
    }

def cmd_status(name):
    path = shutil.which(name)
    return {
        "command": name,
        "available": path is not None,
        "path": path,
    }

def pip_show(pkg):
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", pkg],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return out.strip()
    except Exception:
        return None

report = {
    "modules": [mod_status(m) for m in MODULES],
    "commands": [cmd_status(c) for c in COMMANDS],
    "pip_show": {
        m: pip_show(m)
        for m in MODULES
        if mod_status(m)["installed"]
    },
}

print(json.dumps(report, indent=2))
