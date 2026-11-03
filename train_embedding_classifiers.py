import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer


VERSION = "v3.42"

BASE_DIR = Path(__file__).resolve().parent

ENCODER_REGISTRY = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "trust_remote_code": False,
        "prefix": "",
    },
    "mpnet": {
        "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "trust_remote_code": False,
        "prefix": "",
    },
    "bge_small": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "trust_remote_code": False,
        "prefix": "",
    },
    "bge_base": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "trust_remote_code": False,
        "prefix": "",
    },
    "bge_large": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "trust_remote_code": False,
        "prefix": "",
    },
    "e5_base": {
        "hf_name": "intfloat/e5-base-v2",
        "trust_remote_code": False,
        "prefix": "query: ",
    },
    "e5_large": {
        "hf_name": "intfloat/e5-large-v2",
        "trust_remote_code": False,
        "prefix": "query: ",
    },
}

ENCODER_GROUPS = {
    "small": ["minilm", "bge_small"],
    "recommended": ["minilm", "bge_base", "mpnet", "e5_base"],
    "large": ["bge_large", "e5_large"],
    "all": ["minilm", "bge_small", "bge_base", "mpnet", "e5_base", "bge_large", "e5_large"],
}

CLASSIFIER_REGISTRY = {
    "logreg_balanced": lambda seed, n_jobs: LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="saga",
        n_jobs=n_jobs,
        random_state=seed,
    ),
    "logreg_l2": lambda seed, n_jobs: LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        n_jobs=n_jobs,
        random_state=seed,
    ),
    "linear_svc_balanced": lambda seed, n_jobs: LinearSVC(
        class_weight="balanced",
        random_state=seed,
        max_iter=10000,
    ),
    "sgd_logloss_balanced": lambda seed, n_jobs: SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        random_state=seed,
        max_iter=3000,
        tol=1e-4,
        n_jobs=n_jobs,
    ),
    "sgd_hinge_balanced": lambda seed, n_jobs: SGDClassifier(
        loss="hinge",
        class_weight="balanced",
        random_state=seed,
        max_iter=3000,
        tol=1e-4,
        n_jobs=n_jobs,
    ),
    "ridge_balanced": lambda seed, n_jobs: RidgeClassifier(
        class_weight="balanced",
        random_state=seed,
    ),
    "random_forest_balanced": lambda seed, n_jobs: RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=n_jobs,
        max_depth=None,
        min_samples_leaf=2,
    ),
    "extra_trees_balanced": lambda seed, n_jobs: ExtraTreesClassifier(
        n_estimators=700,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        max_depth=None,
        min_samples_leaf=2,
    ),
    "hist_gradient_boosting": lambda seed, n_jobs: HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        random_state=seed,
        l2_regularization=0.1,
    ),
    "mlp": lambda seed, n_jobs: MLPClassifier(
        hidden_layer_sizes=(512, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=seed,
    ),
}

CLASSIFIER_GROUPS = {
    "fast": ["logreg_balanced", "sgd_logloss_balanced", "linear_svc_balanced", "ridge_balanced"],
    "tree": ["random_forest_balanced", "extra_trees_balanced", "hist_gradient_boosting"],
    "all": list(CLASSIFIER_REGISTRY.keys()),
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_label(row):
    for key in ["label", "decision", "target", "primary_decision"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    gold = row.get("gold")
    if isinstance(gold, dict):
        value = gold.get("primary_decision")
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("Could not find label in row")


def load_example_id(row, idx):
    value = row.get("example_id") or row.get("task_id") or row.get("id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"row_{idx}"


def load_text(row, max_text_chars):
    for key in ["text", "input_text", "model_input", "classifier_input"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value[:max_text_chars]

    current_query = row.get("current_query") or row.get("query") or row.get("question") or ""
    candidate_text = ""

    candidate_memory = row.get("candidate_memory")
    if isinstance(candidate_memory, dict):
        candidate_text = candidate_memory.get("text") or candidate_memory.get("memory_text") or ""

    if not candidate_text:
        candidate_text = row.get("candidate_memory_text") or row.get("memory_text") or row.get("candidate_text") or ""

    existing = row.get("existing_memories")
    existing_text = ""
    if isinstance(existing, list) and existing:
        parts = []
        for item in existing[:5]:
            if isinstance(item, dict):
                txt = item.get("memory_text") or item.get("text")
                if txt:
                    parts.append(str(txt))
            elif isinstance(item, str):
                parts.append(item)
        existing_text = "\n".join(parts)

    text = (
        "Task: retrieval governance\n"
        f"Query: {current_query}\n"
        f"Candidate memory: {candidate_text}\n"
        f"Existing memories: {existing_text}"
    )
    return text[:max_text_chars]


def maybe_limit(rows, limit, seed):
    if not limit or limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:limit]


def load_split(path, max_text_chars, limit, seed):
    rows = read_jsonl(path)
    rows = maybe_limit(rows, limit, seed)
    texts = []
    labels = []
    ids = []
    raw_meta = []

    for idx, row in enumerate(rows):
        texts.append(load_text(row, max_text_chars))
        labels.append(load_label(row))
        ids.append(load_example_id(row, idx))
        raw_meta.append(
            {
                "example_id": ids[-1],
                "source_dataset": row.get("source_dataset"),
                "source_file": row.get("source_file"),
                "scenario_type": row.get("scenario_type"),
                "memory_type": (row.get("candidate_memory") or {}).get("memory_type")
                if isinstance(row.get("candidate_memory"), dict)
                else row.get("memory_type"),
                "sensitivity": (row.get("candidate_memory") or {}).get("sensitivity")
                if isinstance(row.get("candidate_memory"), dict)
                else row.get("sensitivity"),
            }
        )

    return {
        "rows": rows,
        "texts": texts,
        "labels": labels,
        "ids": ids,
        "meta": raw_meta,
        "label_counter": dict(Counter(labels)),
        "total": len(labels),
    }


def resolve_names(value, registry, groups):
    if value == "all":
        return list(groups["all"])
    if value in groups:
        return list(groups[value])

    names = []
    for part in value.split(","):
        name = part.strip()
        if not name:
            continue
        if name in groups:
            names.extend(groups[name])
        elif name in registry:
            names.append(name)
        else:
            raise ValueError(f"Unknown name: {name}")
    return list(dict.fromkeys(names))


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state):
    return last_hidden_state[:, 0]


def l2_normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def encode_texts(
    texts,
    encoder_key,
    encoder_cfg,
    cache_path,
    batch_size,
    max_length,
    pooling,
    normalize,
    device,
    dtype,
    recompute,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not recompute:
        return np.load(cache_path)

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_cfg["hf_name"],
        trust_remote_code=encoder_cfg.get("trust_remote_code", False),
    )

    torch_dtype = torch.float32
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModel.from_pretrained(
        encoder_cfg["hf_name"],
        trust_remote_code=encoder_cfg.get("trust_remote_code", False),
        torch_dtype=torch_dtype if device.startswith("cuda") else torch.float32,
    )
    model.to(device)
    model.eval()

    prefix = encoder_cfg.get("prefix", "")
    encoded_chunks = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = [prefix + text for text in texts[start : start + batch_size]]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(
                device_type="cuda",
                dtype=torch_dtype,
                enabled=device.startswith("cuda") and dtype in {"float16", "bfloat16"},
            ):
                outputs = model(**batch)
                if pooling == "cls":
                    pooled = cls_pool(outputs.last_hidden_state)
                else:
                    pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])

            arr = pooled.detach().float().cpu().numpy()
            encoded_chunks.append(arr)

    embeddings = np.vstack(encoded_chunks).astype(np.float32)
    if normalize:
        embeddings = l2_normalize(embeddings).astype(np.float32)

    ensure_dir(cache_path.parent)
    np.save(cache_path, embeddings)
    return embeddings


def get_scores(model, x, positive_idx):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(model.classes_)
        if positive_idx in classes:
            pos_col = classes.index(positive_idx)
        else:
            pos_col = positive_idx
        return proba[:, pos_col].astype(float)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            if positive_idx == 1:
                return scores.astype(float)
            return (-scores).astype(float)
        return scores[:, positive_idx].astype(float)

    return None


def metrics_for_predictions(y_true, y_pred, labels, positive_idx, scores=None):
    positive_label = labels[positive_idx]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[positive_idx],
        average="binary",
        pos_label=positive_idx,
        zero_division=0,
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "positive_label": positive_label,
        "positive_precision": float(p),
        "positive_recall": float(r),
        "positive_f1": float(f1),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "label_counter_true": {labels[k]: int(v) for k, v in Counter(y_true).items()},
        "label_counter_pred": {labels[k]: int(v) for k, v in Counter(y_pred).items()},
    }

    if scores is not None and len(set(y_true)) == 2:
        y_bin = np.asarray([1 if y == positive_idx else 0 for y in y_true])
        try:
            result["roc_auc"] = float(roc_auc_score(y_bin, scores))
        except Exception:
            result["roc_auc"] = None
        try:
            result["average_precision"] = float(average_precision_score(y_bin, scores))
        except Exception:
            result["average_precision"] = None

    return result


def threshold_predictions(scores, threshold, positive_idx, negative_idx):
    return np.asarray([positive_idx if score >= threshold else negative_idx for score in scores], dtype=int)


def tune_threshold(y_true, scores, positive_idx, negative_idx, precision_constraints):
    if scores is None:
        return {}

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return {}

    quantiles = np.linspace(0.0, 1.0, 1001)
    thresholds = np.unique(np.quantile(scores, quantiles))
    thresholds = np.concatenate(([scores.min() - 1e-9], thresholds, [scores.max() + 1e-9]))

    best_f1 = None
    constrained = {str(c): None for c in precision_constraints}

    for threshold in thresholds:
        pred = threshold_predictions(scores, threshold, positive_idx, negative_idx)
        precision = precision_score(y_true, pred, pos_label=positive_idx, zero_division=0)
        recall = recall_score(y_true, pred, pos_label=positive_idx, zero_division=0)
        f1 = f1_score(y_true, pred, pos_label=positive_idx, zero_division=0)

        item = {
            "threshold": float(threshold),
            "positive_precision": float(precision),
            "positive_recall": float(recall),
            "positive_f1": float(f1),
        }

        if best_f1 is None or f1 > best_f1["positive_f1"]:
            best_f1 = item

        for c in precision_constraints:
            key = str(c)
            if precision >= c:
                if constrained[key] is None or recall > constrained[key]["positive_recall"]:
                    constrained[key] = item

    return {
        "best_f1": best_f1,
        "precision_constrained": constrained,
    }


def evaluate_split(model, x, y_true, labels, positive_idx, negative_idx, threshold=None):
    start = time.time()
    scores = get_scores(model, x, positive_idx)

    if threshold is None:
        y_pred = model.predict(x)
    else:
        y_pred = threshold_predictions(scores, threshold, positive_idx, negative_idx)

    y_pred = np.asarray(y_pred, dtype=int)

    result = metrics_for_predictions(y_true, y_pred, labels, positive_idx, scores=scores)
    result["eval_seconds"] = float(time.time() - start)
    result["examples_per_second"] = float(len(y_true) / max(result["eval_seconds"], 1e-9))
    if threshold is not None:
        result["threshold"] = float(threshold)
    return result, y_pred, scores


def write_predictions(path, ids, y_true, y_pred, scores, labels, meta, split_name, encoder_name, classifier_name):
    rows = []
    for i in range(len(ids)):
        row = {
            "example_id": ids[i],
            "split": split_name,
            "encoder": encoder_name,
            "classifier": classifier_name,
            "true_label": labels[int(y_true[i])],
            "predicted_label": labels[int(y_pred[i])],
        }
        if scores is not None:
            row["positive_score"] = float(scores[i])
        if meta:
            row.update(meta[i])
        rows.append(row)
    write_jsonl(path, rows)


def train_one_classifier(
    classifier_name,
    model,
    x_train,
    y_train,
    split_embeddings,
    split_data,
    labels,
    positive_idx,
    negative_idx,
    output_dir,
    encoder_name,
    selection_metric,
    threshold_split,
    precision_constraints,
):
    ensure_dir(output_dir)
    metrics_path = Path(output_dir) / "metrics.json"
    model_path = Path(output_dir) / "model.joblib"

    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start

    joblib.dump(model, model_path)

    metrics = {
        "version": VERSION,
        "encoder": encoder_name,
        "classifier": classifier_name,
        "train_seconds": float(train_seconds),
        "metrics": {},
        "threshold_tuning": {},
        "threshold_tuned_metrics": {},
    }

    default_outputs = {}

    for split_name, x in split_embeddings.items():
        y_true = split_data[split_name]["y"]
        result, y_pred, scores = evaluate_split(model, x, y_true, labels, positive_idx, negative_idx)
        metrics["metrics"][split_name] = result
        default_outputs[split_name] = {
            "pred": y_pred,
            "scores": scores,
        }

        write_predictions(
            Path(output_dir) / f"predictions_{split_name}.jsonl",
            split_data[split_name]["ids"],
            y_true,
            y_pred,
            scores,
            labels,
            split_data[split_name]["meta"],
            split_name,
            encoder_name,
            classifier_name,
        )

    if threshold_split in split_embeddings:
        scores = default_outputs[threshold_split]["scores"]
        y_true = split_data[threshold_split]["y"]
        tuning = tune_threshold(y_true, scores, positive_idx, negative_idx, precision_constraints)
        metrics["threshold_tuning"][threshold_split] = tuning

        if tuning.get("best_f1"):
            threshold = tuning["best_f1"]["threshold"]
            for split_name, x in split_embeddings.items():
                tuned_result, tuned_pred, tuned_scores = evaluate_split(
                    model,
                    x,
                    split_data[split_name]["y"],
                    labels,
                    positive_idx,
                    negative_idx,
                    threshold=threshold,
                )
                metrics["threshold_tuned_metrics"][split_name] = tuned_result

                write_predictions(
                    Path(output_dir) / f"predictions_{split_name}_threshold_best_f1.jsonl",
                    split_data[split_name]["ids"],
                    split_data[split_name]["y"],
                    tuned_pred,
                    tuned_scores,
                    labels,
                    split_data[split_name]["meta"],
                    split_name,
                    encoder_name,
                    classifier_name,
                )

    if "val_natural" in metrics["metrics"]:
        score_split = "val_natural"
    else:
        score_split = "val"

    metrics["selection_split"] = score_split
    metrics["selection_metric"] = selection_metric
    metrics["selection_score"] = metrics["metrics"].get(score_split, {}).get(selection_metric)

    save_json(metrics_path, metrics)
    return metrics


def get_default_paths(task):
    if task != "retrieval":
        raise ValueError("This script currently supports retrieval only.")

    root = BASE_DIR / "processed" / "memorygovbench_model_ready"

    paths = {
        "train": root / "retrieval_train_model_ready.jsonl",
        "val": root / "retrieval_val_model_ready.jsonl",
        "test_balanced": root / "retrieval_test_balanced_model_ready.jsonl",
        "test_natural": root / "retrieval_test_natural_model_ready.jsonl",
    }

    optional_val_natural = root / "retrieval_val_natural_model_ready.jsonl"
    if optional_val_natural.exists():
        paths["val_natural"] = optional_val_natural

    return paths


def rebuild_leaderboard(output_root, selection_metric):
    rows = []

    for metrics_path in sorted(Path(output_root).glob("*/*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        row = {
            "encoder": metrics.get("encoder"),
            "classifier": metrics.get("classifier"),
            "model": f"{metrics.get('encoder')}__{metrics.get('classifier')}",
            "selection_split": metrics.get("selection_split"),
            "selection_metric": metrics.get("selection_metric"),
            "selection_score": metrics.get("selection_score"),
            "metrics_file": str(metrics_path),
        }

        for split_name, split_metrics in metrics.get("metrics", {}).items():
            for key in [
                "accuracy",
                "macro_f1",
                "positive_precision",
                "positive_recall",
                "positive_f1",
                "roc_auc",
                "average_precision",
            ]:
                value = split_metrics.get(key)
                row[f"{split_name}_{key}"] = value

        for split_name, split_metrics in metrics.get("threshold_tuned_metrics", {}).items():
            for key in [
                "accuracy",
                "macro_f1",
                "positive_precision",
                "positive_recall",
                "positive_f1",
                "roc_auc",
                "average_precision",
                "threshold",
            ]:
                value = split_metrics.get(key)
                row[f"{split_name}_tuned_{key}"] = value

        rows.append(row)

    rows.sort(
        key=lambda r: (
            -float(r["selection_score"]) if r.get("selection_score") is not None else float("inf"),
            r["model"] or "",
        )
    )

    columns = [
        "rank",
        "model",
        "encoder",
        "classifier",
        "selection_split",
        "selection_metric",
        "selection_score",
        "val_positive_f1",
        "val_macro_f1",
        "val_roc_auc",
        "val_average_precision",
        "val_natural_positive_f1",
        "val_natural_positive_precision",
        "val_natural_positive_recall",
        "test_balanced_positive_f1",
        "test_balanced_positive_precision",
        "test_balanced_positive_recall",
        "test_balanced_roc_auc",
        "test_balanced_average_precision",
        "test_natural_positive_f1",
        "test_natural_positive_precision",
        "test_natural_positive_recall",
        "test_natural_roc_auc",
        "test_natural_average_precision",
        "test_natural_accuracy",
        "test_natural_tuned_positive_f1",
        "test_natural_tuned_positive_precision",
        "test_natural_tuned_positive_recall",
        "test_natural_tuned_threshold",
        "metrics_file",
    ]

    leaderboard_path = Path(output_root) / "leaderboard.tsv"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for idx, row in enumerate(rows, 1):
            out = dict(row)
            out["rank"] = idx
            values = []
            for col in columns:
                value = out.get(col)
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                elif value is None:
                    values.append("")
                else:
                    values.append(str(value))
            f.write("\t".join(values) + "\n")

    comparison_path = Path(output_root) / "model_comparison.jsonl"
    write_jsonl(comparison_path, rows)

    summary = {
        "version": VERSION,
        "total_models": len(rows),
        "selection_metric": selection_metric,
        "best_model": rows[0]["model"] if rows else None,
        "best_score": rows[0]["selection_score"] if rows else None,
        "leaderboard": str(leaderboard_path),
        "model_comparison": str(comparison_path),
    }
    save_json(Path(output_root) / "summary_report.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="retrieval", choices=["retrieval"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoders", default="small")
    parser.add_argument("--models", default="fast")
    parser.add_argument("--positive_label", default="RETRIEVE")
    parser.add_argument("--selection_metric", default="positive_f1")
    parser.add_argument("--threshold_split", default="val_natural")
    parser.add_argument("--precision_constraints", default="0.10,0.20,0.30,0.50")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--no_normalize_embeddings", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_text_chars", type=int, default=20000)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_eval", type=int, default=0)

    parser.add_argument("--recompute_embeddings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", default=True)

    parser.add_argument("--train_path", default=None)
    parser.add_argument("--val_path", default=None)
    parser.add_argument("--val_natural_path", default=None)
    parser.add_argument("--test_balanced_path", default=None)
    parser.add_argument("--test_natural_path", default=None)

    return parser.parse_args()


def train_and_evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    paths = get_default_paths(args.task)

    if args.train_path:
        paths["train"] = Path(args.train_path)
    if args.val_path:
        paths["val"] = Path(args.val_path)
    if args.val_natural_path:
        paths["val_natural"] = Path(args.val_natural_path)
    if args.test_balanced_path:
        paths["test_balanced"] = Path(args.test_balanced_path)
    if args.test_natural_path:
        paths["test_natural"] = Path(args.test_natural_path)

    existing_paths = {}
    for split_name, path in paths.items():
        if Path(path).exists():
            existing_paths[split_name] = Path(path)

    if "train" not in existing_paths or "val" not in existing_paths:
        raise FileNotFoundError(f"Missing train or val paths: {existing_paths}")

    split_data_raw = {}
    for split_name, path in existing_paths.items():
        limit = args.limit_train if split_name == "train" else args.limit_eval
        split_data_raw[split_name] = load_split(path, args.max_text_chars, limit, args.seed)

    all_labels = []
    for split_name in ["train", "val", "val_natural", "test_balanced", "test_natural"]:
        if split_name in split_data_raw:
            all_labels.extend(split_data_raw[split_name]["labels"])

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    labels = list(label_encoder.classes_)

    if args.positive_label not in labels:
        raise ValueError(f"Positive label {args.positive_label} not in labels: {labels}")

    positive_idx = int(label_encoder.transform([args.positive_label])[0])
    negative_candidates = [i for i in range(len(labels)) if i != positive_idx]
    if len(negative_candidates) != 1:
        raise ValueError(f"Expected binary labels, found: {labels}")
    negative_idx = negative_candidates[0]

    split_data = {}
    for split_name, data in split_data_raw.items():
        split_data[split_name] = {
            "texts": data["texts"],
            "y": label_encoder.transform(data["labels"]).astype(int),
            "ids": data["ids"],
            "meta": data["meta"],
            "label_counter": data["label_counter"],
            "total": data["total"],
        }

    encoder_names = resolve_names(args.encoders, ENCODER_REGISTRY, ENCODER_GROUPS)
    classifier_names = resolve_names(args.models, CLASSIFIER_REGISTRY, CLASSIFIER_GROUPS)

    normalize = True
    if args.no_normalize_embeddings:
        normalize = False
    if args.normalize_embeddings:
        normalize = True

    run_config = {
        "version": VERSION,
        "task": args.task,
        "output_dir": str(output_root),
        "encoders": encoder_names,
        "classifiers": classifier_names,
        "labels": labels,
        "positive_label": args.positive_label,
        "positive_idx": positive_idx,
        "negative_idx": negative_idx,
        "paths": {k: str(v) for k, v in existing_paths.items()},
        "split_stats": {
            k: {
                "total": v["total"],
                "label_counter": v["label_counter"],
            }
            for k, v in split_data_raw.items()
        },
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pooling": args.pooling,
        "normalize_embeddings": normalize,
        "dtype": args.dtype,
        "device": device,
        "max_text_chars": args.max_text_chars,
    }
    save_json(output_root / "run_config.json", run_config)

    precision_constraints = [
        float(x.strip())
        for x in args.precision_constraints.split(",")
        if x.strip()
    ]

    for encoder_name in encoder_names:
        encoder_cfg = ENCODER_REGISTRY[encoder_name]
        print(f"\n=== Encoder: {encoder_name} | {encoder_cfg['hf_name']} ===", flush=True)

        encoder_dir = output_root / encoder_name
        embedding_dir = encoder_dir / "embeddings"
        ensure_dir(embedding_dir)

        split_embeddings = {}
        for split_name, data in split_data.items():
            cache_path = embedding_dir / f"{split_name}_embeddings.npy"
            print(f"Encoding/loading {encoder_name} {split_name}...", flush=True)
            split_embeddings[split_name] = encode_texts(
                data["texts"],
                encoder_name,
                encoder_cfg,
                cache_path,
                args.batch_size,
                args.max_length,
                args.pooling,
                normalize,
                device,
                args.dtype,
                args.recompute_embeddings,
            )

        x_train = split_embeddings["train"]
        y_train = split_data["train"]["y"]

        for classifier_name in classifier_names:
            classifier_dir = encoder_dir / classifier_name
            metrics_path = classifier_dir / "metrics.json"

            if metrics_path.exists() and args.skip_existing and not args.overwrite:
                print(f"Skipping existing {encoder_name}__{classifier_name}", flush=True)
                continue

            print(f"\n=== Training {encoder_name}__{classifier_name} ===", flush=True)

            model = CLASSIFIER_REGISTRY[classifier_name](args.seed, args.n_jobs)

            threshold_split = args.threshold_split
            if threshold_split not in split_embeddings:
                threshold_split = "val"

            try:
                metrics = train_one_classifier(
                    classifier_name,
                    model,
                    x_train,
                    y_train,
                    split_embeddings,
                    split_data,
                    labels,
                    positive_idx,
                    negative_idx,
                    classifier_dir,
                    encoder_name,
                    args.selection_metric,
                    threshold_split,
                    precision_constraints,
                )

                for split_name, split_metrics in metrics["metrics"].items():
                    print(
                        f"{encoder_name}__{classifier_name} | {split_name} | "
                        f"acc={split_metrics.get('accuracy'):.4f} "
                        f"macro_f1={split_metrics.get('macro_f1'):.4f} "
                        f"pos_f1={split_metrics.get('positive_f1'):.4f} "
                        f"pos_p={split_metrics.get('positive_precision'):.4f} "
                        f"pos_r={split_metrics.get('positive_recall'):.4f}",
                        flush=True,
                    )

            except Exception as exc:
                ensure_dir(classifier_dir)
                error_report = {
                    "version": VERSION,
                    "encoder": encoder_name,
                    "classifier": classifier_name,
                    "error": repr(exc),
                }
                save_json(classifier_dir / "error.json", error_report)
                print(f"FAILED {encoder_name}__{classifier_name}: {repr(exc)}", flush=True)

    summary = rebuild_leaderboard(output_root, args.selection_metric)
    print("\nFinished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
