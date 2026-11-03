import json
from pathlib import Path

CANDIDATES = [
    ("MPNet", "processed/operation_classifier_results/transformer_mpnet_base_v1_3_2_no_confirmation/summary.json"),
    ("DeBERTa", "processed/operation_classifier_results/transformer_deberta_v1_3_2_no_confirmation/summary.json"),
    ("Qwen32 zero-shot", "processed/operation_llm_eval/qwen32_zero_fewshot_v1/qwen32_zero_summary.json"),
    ("Qwen32 few-shot", "processed/operation_llm_eval/qwen32_zero_fewshot_v1/qwen32_few_summary.json"),
    ("Qwen7B LoRA", "processed/operation_classifier_results/qwen2_5_7b_lora_opgov_v1_3_2_no_confirmation/summary.json"),
    ("Qwen14B LoRA", "processed/operation_classifier_results/qwen2_5_14b_lora_opgov_v1_3_2_no_confirmation/summary.json"),
    ("Qwen32B LoRA smoke", "processed/operation_classifier_results/qwen2_5_32b_lora_opgov_smoke_v1_3_2_no_confirmation/summary.json"),
    ("Qwen32B LoRA full", "processed/operation_classifier_results/qwen2_5_32b_lora_opgov_full_v1_3_2_no_confirmation/summary.json"),
]

def load(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": repr(e)}

def extract(summary):
    if summary is None:
        return None

    if "_error" in summary:
        return {"error": summary["_error"]}

    # Transformer classifier schema
    if "accuracy" in summary and "macro_f1" in summary:
        return {
            "accuracy": summary.get("accuracy"),
            "macro_f1": summary.get("macro_f1"),
            "train_seconds": summary.get("train_seconds"),
            "class_f1": summary.get("class_f1", {}),
        }

    # Qwen zero/few-shot schema
    if "accuracy_all" in summary or "macro_f1_valid_only" in summary:
        return {
            "accuracy": summary.get("accuracy_all") or summary.get("accuracy_valid_only"),
            "macro_f1": summary.get("macro_f1_valid_only"),
            "valid_predictions": summary.get("valid_predictions"),
            "parse_or_request_errors": summary.get("parse_or_request_errors"),
            "mean_latency_sec": summary.get("mean_latency_sec"),
            "median_latency_sec": summary.get("median_latency_sec"),
        }

    # Qwen LoRA schema
    if "eval" in summary:
        ev = summary["eval"]
        return {
            "accuracy": ev.get("accuracy_all") or ev.get("accuracy_valid_only"),
            "macro_f1": ev.get("macro_f1_valid_only"),
            "valid_predictions": ev.get("valid_predictions"),
            "parse_errors": ev.get("parse_errors"),
            "mean_latency_sec": ev.get("mean_latency_sec"),
            "median_latency_sec": ev.get("median_latency_sec"),
            "train_seconds": summary.get("train_seconds"),
        }

    return {"unknown_schema_keys": sorted(summary.keys())}

print("\nOperation Governor model-family results\n")
print("| Model | Status | Accuracy | Macro F1 | Train sec | Mean latency | Notes |")
print("|---|---|---:|---:|---:|---:|---|")

for name, path in CANDIDATES:
    s = load(path)
    if s is None:
        print(f"| {name} | missing |  |  |  |  | {path} |")
        continue

    e = extract(s)
    if "error" in e:
        print(f"| {name} | invalid json |  |  |  |  | {e['error']} |")
        continue

    acc = e.get("accuracy")
    mf1 = e.get("macro_f1")
    tr = e.get("train_seconds")
    lat = e.get("mean_latency_sec")
    notes = []
    for k in ["valid_predictions", "parse_or_request_errors", "parse_errors"]:
        if e.get(k) is not None:
            notes.append(f"{k}={e[k]}")

    print(
        f"| {name} | done | "
        f"{acc:.4f} | " if isinstance(acc, (int, float)) else f"| {name} | done |  | "
    , end="")

    print(
        f"{mf1:.4f} | " if isinstance(mf1, (int, float)) else " | ",
        end=""
    )
    print(
        f"{tr:.1f} | " if isinstance(tr, (int, float)) else " | ",
        end=""
    )
    print(
        f"{lat:.3f} | " if isinstance(lat, (int, float)) else " | ",
        end=""
    )
    print("; ".join(notes) + " |")

print("\nExisting summaries:")
for name, path in CANDIDATES:
    if Path(path).exists():
        print(f"  {name}: {path}")
