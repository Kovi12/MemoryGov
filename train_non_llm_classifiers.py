import argparse
import json
import math
import time
import warnings
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


VERSION = "v3.36"


DEFAULT_PATHS = {
    "retrieval": {
        "train": "processed/memorygovbench_model_ready/retrieval_train_model_ready.jsonl",
        "val": "processed/memorygovbench_model_ready/retrieval_val_model_ready.jsonl",
        "test_balanced": "processed/memorygovbench_model_ready/retrieval_test_balanced_model_ready.jsonl",
        "test_natural": "processed/memorygovbench_model_ready/retrieval_test_natural_model_ready.jsonl",
    },
    "operation": {
        "train": "processed/memorygovbench_model_ready/operation_train_model_ready.jsonl",
        "val": "processed/memorygovbench_model_ready/operation_val_model_ready.jsonl",
        "test": "processed/memorygovbench_model_ready/operation_test_model_ready.jsonl",
    },
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_text(x):
    if x is None:
        return ""
    return str(x)


def truncate_text(text, max_chars):
    text = safe_text(text)
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n[...TRUNCATED...]\n" + text[-tail:]


def candidate_memory_text(row):
    candidate = row.get("candidate_memory", "")
    if isinstance(candidate, dict):
        return safe_text(candidate.get("text", ""))
    return safe_text(candidate)


def build_retrieval_text(row):
    parts = []

    query = safe_text(row.get("current_query", ""))
    candidate = candidate_memory_text(row)

    if query:
        parts.append("QUERY:\n" + query)
    if candidate:
        parts.append("CANDIDATE_MEMORY:\n" + candidate)

    existing = row.get("existing_memories", [])
    if isinstance(existing, list) and existing:
        existing_texts = []
        for mem in existing[:8]:
            if isinstance(mem, dict):
                existing_texts.append(safe_text(mem.get("memory_text") or mem.get("text") or ""))
            else:
                existing_texts.append(safe_text(mem))
        existing_texts = [x for x in existing_texts if x.strip()]
        if existing_texts:
            parts.append("EXISTING_MEMORIES:\n" + "\n\n".join(existing_texts))

    scenario = safe_text(row.get("scenario_type", ""))
    memory_type = ""
    sensitivity = ""

    candidate_obj = row.get("candidate_memory", {})
    if isinstance(candidate_obj, dict):
        memory_type = safe_text(candidate_obj.get("memory_type", ""))
        sensitivity = safe_text(candidate_obj.get("sensitivity", ""))

    meta = []
    if scenario:
        meta.append(f"scenario_type={scenario}")
    if memory_type:
        meta.append(f"memory_type={memory_type}")
    if sensitivity:
        meta.append(f"sensitivity={sensitivity}")

    if meta:
        parts.append("METADATA:\n" + "\n".join(meta))

    return "\n\n".join(parts)


def build_operation_text(row):
    parts = []

    memory_text = safe_text(row.get("memory_text", ""))
    if not memory_text:
        memory_text = candidate_memory_text(row)

    query = safe_text(row.get("current_query", ""))
    evidence = safe_text(row.get("evidence_span", ""))
    reason = safe_text(row.get("reason", ""))

    if memory_text:
        parts.append("MEMORY_CANDIDATE:\n" + memory_text)
    if query:
        parts.append("CURRENT_QUERY:\n" + query)
    if evidence:
        parts.append("EVIDENCE:\n" + evidence)
    if reason:
        parts.append("REASON:\n" + reason)

    scenario = safe_text(row.get("scenario_type", ""))
    memory_type = safe_text(row.get("memory_type", ""))
    sensitivity = safe_text(row.get("sensitivity", ""))
    stability = safe_text(row.get("stability", ""))

    candidate_obj = row.get("candidate_memory", {})
    if isinstance(candidate_obj, dict):
        if not memory_type:
            memory_type = safe_text(candidate_obj.get("memory_type", ""))
        if not sensitivity:
            sensitivity = safe_text(candidate_obj.get("sensitivity", ""))

    meta = []
    if scenario:
        meta.append(f"scenario_type={scenario}")
    if memory_type:
        meta.append(f"memory_type={memory_type}")
    if sensitivity:
        meta.append(f"sensitivity={sensitivity}")
    if stability:
        meta.append(f"stability={stability}")

    if meta:
        parts.append("METADATA:\n" + "\n".join(meta))

    return "\n\n".join(parts)


def extract_text_from_row(row):
    direct_fields = [
        "text",
        "input",
        "prompt",
        "model_input",
        "classifier_input",
        "classification_input",
    ]

    for field in direct_fields:
        value = safe_text(row.get(field, ""))
        if value.strip():
            return value

    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        message_texts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = safe_text(msg.get("role", ""))
                content = safe_text(msg.get("content", ""))
                if content.strip():
                    message_texts.append(f"{role}: {content}" if role else content)
            else:
                msg_text = safe_text(msg)
                if msg_text.strip():
                    message_texts.append(msg_text)
        if message_texts:
            return "\n".join(message_texts)

    if row.get("task_level") == "retrieval_governance":
        return build_retrieval_text(row)

    if row.get("task_level") == "operation_governance":
        return build_operation_text(row)

    if row.get("current_query") or row.get("candidate_memory"):
        return build_retrieval_text(row)

    if row.get("memory_text") or row.get("evidence_span"):
        return build_operation_text(row)

    return ""


def get_label(row):
    for field in ["label", "target", "decision", "gold_label"]:
        value = row.get(field)
        if value is not None and safe_text(value).strip():
            return safe_text(value)

    gold = row.get("gold", {})
    if isinstance(gold, dict):
        value = gold.get("primary_decision")
        if value is not None and safe_text(value).strip():
            return safe_text(value)

    raise ValueError(f"Could not find label in row with keys: {sorted(row.keys())}")


def extract_xy(rows, max_text_chars):
    ids = []
    texts = []
    labels = []
    empty_text_count = 0

    for idx, row in enumerate(rows):
        example_id = row.get("example_id") or row.get("task_id") or f"row_{idx}"
        text = truncate_text(extract_text_from_row(row), max_text_chars)
        label = get_label(row)

        if not text.strip():
            empty_text_count += 1

        ids.append(example_id)
        texts.append(text)
        labels.append(label)

    if empty_text_count == len(rows):
        sample_keys = sorted(rows[0].keys()) if rows else []
        raise ValueError(
            "All extracted texts are empty. "
            f"Sample row keys: {sample_keys}. "
            "The input schema is not supported by extract_text_from_row()."
        )

    if empty_text_count > 0:
        print(f"Warning: {empty_text_count}/{len(rows)} rows have empty extracted text.", flush=True)

    return ids, np.array(texts, dtype=object), np.array(labels, dtype=object)


def build_tfidf_word(max_features, ngram_min, ngram_max):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )


def build_tfidf_char(max_features):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )


def maybe_add_xgboost(models, random_state):
    try:
        from xgboost import XGBClassifier

        models["svd_xgboost"] = Pipeline([
            ("tfidf", build_tfidf_word(80000, 1, 2)),
            ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=8,
            )),
        ])
    except Exception:
        pass


def maybe_add_lightgbm(models, random_state):
    try:
        from lightgbm import LGBMClassifier

        models["svd_lightgbm"] = Pipeline([
            ("tfidf", build_tfidf_word(80000, 1, 2)),
            ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=8,
            )),
        ])
    except Exception:
        pass


def maybe_add_catboost(models, random_state):
    try:
        from catboost import CatBoostClassifier

        models["svd_catboost"] = Pipeline([
            ("tfidf", build_tfidf_word(80000, 1, 2)),
            ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", CatBoostClassifier(
                iterations=400,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )),
        ])
    except Exception:
        pass


def get_model_registry(random_state, max_features):
    models = {}

    models["tfidf_logreg_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["tfidf_logreg_l2"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", LogisticRegression(
            max_iter=3000,
            solver="saga",
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["tfidf_linear_svc_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000,
        )),
    ])

    models["tfidf_sgd_logloss_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["tfidf_sgd_hinge_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", SGDClassifier(
            loss="hinge",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["tfidf_ridge_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", RidgeClassifier(
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    models["tfidf_passive_aggressive_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", PassiveAggressiveClassifier(
            class_weight="balanced",
            max_iter=2000,
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["tfidf_complement_nb"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", ComplementNB(alpha=0.2)),
    ])

    models["tfidf_multinomial_nb"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("clf", MultinomialNB(alpha=0.2)),
    ])

    models["char_logreg_balanced"] = Pipeline([
        ("tfidf", build_tfidf_char(max_features)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["char_sgd_logloss_balanced"] = Pipeline([
        ("tfidf", build_tfidf_char(max_features)),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["svd_logreg_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["svd_linear_svc_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000,
        )),
    ])

    models["svd_random_forest_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["svd_extra_trees_balanced"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
        ("clf", ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=8,
            random_state=random_state,
        )),
    ])

    models["svd_hist_gradient_boosting"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
        )),
    ])

    models["svd_mlp"] = Pipeline([
        ("tfidf", build_tfidf_word(max_features, 1, 2)),
        ("svd", TruncatedSVD(n_components=256, random_state=random_state)),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=40,
            early_stopping=False,
            random_state=random_state,
        )),
    ])

    maybe_add_xgboost(models, random_state)
    maybe_add_lightgbm(models, random_state)
    maybe_add_catboost(models, random_state)

    return models


def get_scores(model, x, positive_label, label_encoder):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            classes = list(model.classes_)
            if positive_label in classes:
                idx = classes.index(positive_label)
            else:
                idx = list(label_encoder.classes_).index(positive_label)
            return proba[:, idx].astype(float).tolist()
    except Exception:
        pass

    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x)
            if len(np.shape(scores)) == 1:
                return np.asarray(scores, dtype=float).tolist()
            classes = list(model.classes_)
            if positive_label in classes:
                idx = classes.index(positive_label)
            else:
                idx = list(label_encoder.classes_).index(positive_label)
            return np.asarray(scores[:, idx], dtype=float).tolist()
    except Exception:
        pass

    preds = model.predict(x)
    return [1.0 if p == positive_label else 0.0 for p in preds]


def binary_metrics(y_true, y_pred, scores, positive_label):
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    pos_true = np.array([1 if y == positive_label else 0 for y in y_true])
    pos_pred = np.array([1 if y == positive_label else 0 for y in y_pred])

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "positive_label": positive_label,
        "positive_precision": float(precision_score(pos_true, pos_pred, zero_division=0)),
        "positive_recall": float(recall_score(pos_true, pos_pred, zero_division=0)),
        "positive_f1": float(f1_score(pos_true, pos_pred, zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        "label_counter_true": dict(Counter(y_true.tolist())),
        "label_counter_pred": dict(Counter(y_pred.tolist())),
    }

    try:
        result["roc_auc"] = float(roc_auc_score(pos_true, scores))
    except Exception:
        result["roc_auc"] = None

    try:
        result["average_precision"] = float(average_precision_score(pos_true, scores))
    except Exception:
        result["average_precision"] = None

    return result


def multiclass_metrics(y_true, y_pred):
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        "label_counter_true": dict(Counter(y_true.tolist())),
        "label_counter_pred": dict(Counter(y_pred.tolist())),
    }


def evaluate_model(model, split_name, ids, x, y, positive_label, label_encoder, out_dir, model_name):
    start = time.time()
    pred = model.predict(x)
    elapsed = time.time() - start

    if positive_label and len(set(y.tolist())) == 2:
        scores = get_scores(model, x, positive_label, label_encoder)
        metrics = binary_metrics(y, pred, scores, positive_label)
    else:
        scores = None
        metrics = multiclass_metrics(y, pred)

    metrics["eval_seconds"] = elapsed
    metrics["examples_per_second"] = len(y) / elapsed if elapsed > 0 else None

    pred_rows = []
    for i, example_id in enumerate(ids):
        row = {
            "example_id": example_id,
            "split": split_name,
            "label": str(y[i]),
            "prediction": str(pred[i]),
            "correct": bool(y[i] == pred[i]),
        }
        if scores is not None:
            row["score"] = float(scores[i])
        pred_rows.append(row)

    pred_path = out_dir / model_name / f"predictions_{split_name}.jsonl"
    write_jsonl(pred_path, pred_rows)

    return metrics


def select_model_names(requested, registry):
    if requested == ["all"]:
        return list(registry.keys())

    selected = []
    missing = []
    for name in requested:
        if name in registry:
            selected.append(name)
        else:
            missing.append(name)

    if missing:
        raise ValueError(f"Unknown model names: {missing}. Available: {list(registry.keys())}")

    return selected


def load_splits(args):
    paths = DEFAULT_PATHS[args.task].copy()

    if args.train:
        paths["train"] = args.train
    if args.val:
        paths["val"] = args.val
    if args.test_balanced:
        paths["test_balanced"] = args.test_balanced
    if args.test_natural:
        paths["test_natural"] = args.test_natural
    if args.test:
        paths["test"] = args.test

    loaded = {}
    for split, path in paths.items():
        path = Path(path)
        if path.exists():
            rows = read_jsonl(path)
            loaded[split] = {
                "path": str(path),
                "rows": rows,
            }

    return loaded


def train_and_evaluate(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_splits(args)
    if "train" not in loaded or "val" not in loaded:
        raise FileNotFoundError("Train and validation files are required.")

    train_ids, x_train, y_train_raw = extract_xy(loaded["train"]["rows"], args.max_text_chars)
    val_ids, x_val, y_val_raw = extract_xy(loaded["val"]["rows"], args.max_text_chars)

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_val_raw]))

    positive_label = args.positive_label
    if positive_label == "auto":
        if args.task == "retrieval":
            positive_label = "RETRIEVE"
        else:
            positive_label = None

    registry = get_model_registry(args.random_state, args.max_features)
    selected_names = select_model_names(args.models.split(","), registry)

    report = {
        "version": VERSION,
        "task": args.task,
        "output_dir": str(out_dir),
        "selected_models": selected_names,
        "available_models": list(registry.keys()),
        "positive_label": positive_label,
        "max_features": args.max_features,
        "max_text_chars": args.max_text_chars,
        "splits": {
            split: {
                "path": data["path"],
                "total": len(data["rows"]),
                "label_counter": dict(Counter(row.get("label") for row in data["rows"])),
            }
            for split, data in loaded.items()
        },
        "models": {},
    }

    save_json(out_dir / "run_config.json", report)

    best_model_name = None
    best_score = -math.inf

    for model_name in selected_names:
        print(f"\n=== Training {model_name} ===", flush=True)
        model_start = time.time()
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model = clone(registry[model_name])
        model.fit(x_train, y_train_raw)

        model_report = {
            "model_name": model_name,
            "train_seconds": time.time() - model_start,
            "metrics": {},
        }

        split_eval_data = {
            "val": (val_ids, x_val, y_val_raw),
        }

        for split, data in loaded.items():
            if split in ["train", "val"]:
                continue
            ids, x, y = extract_xy(data["rows"], args.max_text_chars)
            split_eval_data[split] = (ids, x, y)

        if args.evaluate_train:
            split_eval_data["train"] = (train_ids, x_train, y_train_raw)

        for split_name, (ids, x, y) in split_eval_data.items():
            metrics = evaluate_model(
                model=model,
                split_name=split_name,
                ids=ids,
                x=x,
                y=y,
                positive_label=positive_label,
                label_encoder=label_encoder,
                out_dir=out_dir,
                model_name=model_name,
            )
            model_report["metrics"][split_name] = metrics
            print(
                f"{model_name} | {split_name} | "
                f"acc={metrics.get('accuracy'):.4f} "
                f"macro_f1={metrics.get('macro_f1'):.4f} "
                f"pos_f1={metrics.get('positive_f1', None)}",
                flush=True,
            )

        joblib.dump(model, model_dir / "model.joblib")
        save_json(model_dir / "metrics.json", model_report)

        val_metrics = model_report["metrics"]["val"]
        score = val_metrics.get(args.selection_metric)

        if score is None:
            score = val_metrics.get("macro_f1", -math.inf)

        if score is not None and score > best_score:
            best_score = score
            best_model_name = model_name

        report["models"][model_name] = model_report
        report["best_model_so_far"] = best_model_name
        report["best_score_so_far"] = best_score
        save_json(out_dir / "summary_report.json", report)

    report["best_model"] = best_model_name
    report["best_score"] = best_score
    save_json(out_dir / "summary_report.json", report)

    rows = []
    for model_name, model_report in report["models"].items():
        row = {
            "model_name": model_name,
            "train_seconds": model_report["train_seconds"],
        }
        for split_name, metrics in model_report["metrics"].items():
            row[f"{split_name}_accuracy"] = metrics.get("accuracy")
            row[f"{split_name}_macro_f1"] = metrics.get("macro_f1")
            row[f"{split_name}_weighted_f1"] = metrics.get("weighted_f1")
            row[f"{split_name}_positive_precision"] = metrics.get("positive_precision")
            row[f"{split_name}_positive_recall"] = metrics.get("positive_recall")
            row[f"{split_name}_positive_f1"] = metrics.get("positive_f1")
            row[f"{split_name}_roc_auc"] = metrics.get("roc_auc")
            row[f"{split_name}_average_precision"] = metrics.get("average_precision")
        rows.append(row)

    write_jsonl(out_dir / "model_comparison.jsonl", rows)

    print("\nFinished.")
    print(f"Best model: {best_model_name}")
    print(f"Best {args.selection_metric}: {best_score}")
    print(f"Report: {out_dir / 'summary_report.json'}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", choices=["retrieval", "operation"], default="retrieval")
    parser.add_argument("--output_dir", default="processed/non_llm_classifier_results/retrieval")
    parser.add_argument("--models", default="all")
    parser.add_argument("--selection_metric", default="positive_f1")
    parser.add_argument("--positive_label", default="auto")
    parser.add_argument("--max_features", type=int, default=100000)
    parser.add_argument("--max_text_chars", type=int, default=20000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--evaluate_train", action="store_true")

    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--test_balanced", default=None)
    parser.add_argument("--test_natural", default=None)

    args = parser.parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
