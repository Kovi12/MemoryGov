import json
import time
from pathlib import Path

import joblib
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_DIR = Path("processed/memorygovbench_operation_v1_3_model_ready_noleak")
OUT_DIR = Path("processed/operation_classifier_results/non_llm_v1_3_no_ask_user_selected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "QUARANTINE"]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def get_label(row):
    for key in ["operation_label_v1_3", "label", "operation_label", "target", "class_label", "y"]:
        if key in row and isinstance(row[key], str):
            return row[key]
    raise KeyError(f"No label field found in row keys: {sorted(row.keys())}")


def get_text(row):
    for key in ["text", "input", "prompt"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]

    parts = []
    for key in ["user_message", "message", "candidate", "candidate_memory", "target_memory", "context"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(f"{key}: {v}")
    if parts:
        return "\n".join(parts)

    raise KeyError(f"No text/input field found in row keys: {sorted(row.keys())}")


def resolve_split_path(name):
    candidates = {
        "train": [
            "operation_train_balanced_model_ready.jsonl",
            "operation_train_full_model_ready.jsonl",
            "train.jsonl",
            "train_balanced.jsonl",
            "train_combined.jsonl",
            "train_all.jsonl",
        ],
        "val": [
            "operation_val_model_ready.jsonl",
            "val.jsonl",
            "validation.jsonl",
            "val_balanced.jsonl",
        ],
        "test_synthetic": [
            "operation_test_synthetic_model_ready.jsonl",
            "test_synthetic.jsonl",
            "synthetic_test.jsonl",
        ],
        "test_benchmark": [
            "operation_test_benchmark_model_ready.jsonl",
            "test_benchmark.jsonl",
            "benchmark_test.jsonl",
        ],
        "test_combined": [
            "operation_test_combined_model_ready.jsonl",
            "test_combined.jsonl",
            "combined_test.jsonl",
        ],
        "test": [
            "test.jsonl",
        ],
    }.get(name, [f"{name}.jsonl"])

    for fname in candidates:
        path = DATA_DIR / fname
        if path.exists():
            return path

    return DATA_DIR / f"{name}.jsonl"


def load_split(name):
    path = resolve_split_path(name)
    rows = read_jsonl(path)
    print(f"loaded split {name}: {path} rows={len(rows)}")

    X = [get_text(r) for r in rows]
    y = [get_label(r) for r in rows]
    return rows, X, y


def eval_model(model, X, y, name):
    pred = model.predict(X)
    supported = sorted(set(y) | set(pred))

    return {
        "split": name,
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=supported, average="macro", zero_division=0),
        "supported_macro_f1": f1_score(y, pred, labels=supported, average="macro", zero_division=0),
        "labels_present": sorted(set(y)),
        "classification_report": classification_report(y, pred, labels=LABELS, zero_division=0, output_dict=True),
    }


def build_models():
    return {
        "tfidf_sgd_hinge_balanced": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", SGDClassifier(
                loss="hinge",
                alpha=1e-5,
                class_weight="balanced",
                random_state=17,
                max_iter=5000,
                tol=1e-4,
            )),
        ]),
        "tfidf_sgd_logloss_balanced": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                class_weight="balanced",
                random_state=17,
                max_iter=5000,
                tol=1e-4,
            )),
        ]),
        "tfidf_linear_svc_balanced": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=17,
                max_iter=10000,
            )),
        ]),
        "char_sgd_hinge_balanced": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 6),
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
            )),
            ("clf", SGDClassifier(
                loss="hinge",
                alpha=1e-5,
                class_weight="balanced",
                random_state=17,
                max_iter=5000,
                tol=1e-4,
            )),
        ]),
        "tfidf_logreg_balanced": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=3.0,
                class_weight="balanced",
                random_state=17,
                max_iter=3000,
                n_jobs=1,
            )),
        ]),
    }


def main():
    train_rows, X_train, y_train = load_split("train")
    val_rows, X_val, y_val = load_split("val")

    test_splits = []
    for name in ["test_synthetic", "test_benchmark", "test_combined", "test"]:
        if (DATA_DIR / f"{name}.jsonl").exists():
            test_splits.append((name, *load_split(name)[1:]))

    print("train", len(y_train), {x: y_train.count(x) for x in sorted(set(y_train))})
    print("val", len(y_val), {x: y_val.count(x) for x in sorted(set(y_val))})
    print("test_splits", [x[0] for x in test_splits])

    if not X_train or not y_train:
        raise SystemExit(f"Empty training split in {DATA_DIR}. Check v1.3 data conversion.")
    if not X_val or not y_val:
        raise SystemExit(f"Empty validation split in {DATA_DIR}. Check v1.3 data conversion.")

    results = []

    for name, model in build_models().items():
        print(f"\nTraining {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        model_dir = OUT_DIR / name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")

        model_results = {
            "model": name,
            "train_seconds": elapsed,
            "val": eval_model(model, X_val, y_val, "val"),
        }

        for split_name, X_test, y_test in test_splits:
            model_results[split_name] = eval_model(model, X_test, y_test, split_name)

        (model_dir / "metrics.json").write_text(json.dumps(model_results, indent=2), encoding="utf-8")
        results.append(model_results)

        print("val supported macro F1", model_results["val"]["supported_macro_f1"])

    # Safety-oriented selection:
    # primary: validation supported macro F1
    # tie-breaker: QUARANTINE F1 on val
    def select_key(r):
        val = r["val"]
        q_f1 = val["classification_report"].get("QUARANTINE", {}).get("f1-score", 0.0)
        forget_f1 = val["classification_report"].get("FORGET", {}).get("f1-score", 0.0)
        return (val["supported_macro_f1"], q_f1, forget_f1)

    best = max(results, key=select_key)

    summary = {
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "label_space": LABELS,
        "models": [
            {
                "model": r["model"],
                "val_supported_macro_f1": r["val"]["supported_macro_f1"],
                "val_accuracy": r["val"]["accuracy"],
                "val_quarantine_f1": r["val"]["classification_report"].get("QUARANTINE", {}).get("f1-score", 0.0),
                "val_forget_f1": r["val"]["classification_report"].get("FORGET", {}).get("f1-score", 0.0),
            }
            for r in results
        ],
        "selected_model": best["model"],
        "selected_model_path": str(OUT_DIR / best["model"] / "model.joblib"),
        "selected_val_supported_macro_f1": best["val"]["supported_macro_f1"],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
