import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


BASE = Path(__file__).resolve().parent

LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]
LIFECYCLE_LABELS = ["IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]

DEFAULT_DATA_DIR = BASE / "processed/memorygovbench_operation_v1_1_model_ready_noleak"

ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_base": "BAAI/bge-base-en-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "e5_base": "intfloat/e5-base-v2",
}


def read_jsonl(path, max_text_chars=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("input_text") or row.get("text") or ""
            if max_text_chars and len(text) > max_text_chars:
                text = text[:max_text_chars]
            rows.append({
                "example_id": row.get("example_id"),
                "text": text,
                "label": row.get("label") or row.get("target"),
                "source_dataset": row.get("source_dataset"),
                "scenario_type": row.get("scenario_type"),
                "operation_v1_source": row.get("operation_v1_source"),
                "memory_type": row.get("memory_type"),
                "sensitivity": row.get("sensitivity"),
            })
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows):
    return {
        "total": len(rows),
        "label_counter": dict(Counter(row["label"] for row in rows)),
        "source_dataset_counter": dict(Counter(row.get("source_dataset") for row in rows)),
        "scenario_counter": dict(Counter(row.get("scenario_type") for row in rows)),
        "operation_v1_source_counter": dict(Counter(row.get("operation_v1_source") for row in rows)),
    }


def stable_hash(items):
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def format_texts_for_encoder(encoder_key, texts):
    if encoder_key.startswith("e5"):
        return [f"query: {text}" for text in texts]
    return texts


def load_or_encode(encoder_key, model_name, split_name, rows, cache_dir, batch_size, device, max_seq_length, force_reencode):
    from sentence_transformers import SentenceTransformer

    texts = [row["text"] for row in rows]
    ids = [row["example_id"] for row in rows]

    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = stable_hash([encoder_key, model_name, split_name, len(rows), *ids[:20], *ids[-20:]])
    emb_path = cache_dir / f"{encoder_key}_{split_name}_{fingerprint}.npy"
    meta_path = cache_dir / f"{encoder_key}_{split_name}_{fingerprint}.json"

    if emb_path.exists() and meta_path.exists() and not force_reencode:
        arr = np.load(emb_path)
        return arr

    print(f"Loading encoder {encoder_key}: {model_name}", flush=True)
    encoder = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        encoder.max_seq_length = max_seq_length

    encoded_texts = format_texts_for_encoder(encoder_key, texts)

    print(f"Encoding {encoder_key} {split_name}: {len(encoded_texts)} examples", flush=True)
    embeddings = encoder.encode(
        encoded_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(np.float32)
    np.save(emb_path, embeddings)

    meta = {
        "encoder_key": encoder_key,
        "model_name": model_name,
        "split_name": split_name,
        "rows": len(rows),
        "embedding_shape": list(embeddings.shape),
        "max_seq_length": max_seq_length,
        "ids_hash": stable_hash(ids),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return embeddings


def get_classifier_specs(n_jobs):
    return {
        "logreg_balanced": LogisticRegression(max_iter=2500, class_weight="balanced", n_jobs=n_jobs),
        "logreg_l2": LogisticRegression(max_iter=2500, n_jobs=n_jobs),
        "linear_svc_balanced": LinearSVC(class_weight="balanced", dual="auto", max_iter=5000),
        "sgd_logloss_balanced": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=42,
        ),
        "sgd_hinge_balanced": SGDClassifier(
            loss="hinge",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=42,
        ),
        "ridge_balanced": RidgeClassifier(class_weight="balanced"),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=42,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=0.02,
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            random_state=42,
        ),
    }


def parse_selection(value, available):
    if value == "all":
        return list(available)
    if value == "fast":
        preferred = [
            "logreg_balanced",
            "linear_svc_balanced",
            "sgd_logloss_balanced",
            "sgd_hinge_balanced",
            "ridge_balanced",
            "mlp",
        ]
        return [x for x in preferred if x in available]
    return [x.strip() for x in value.split(",") if x.strip()]


def predict_scores(model, x, encoder):
    clf_classes = getattr(model, "classes_", None)

    def decode_class(c):
        try:
            return encoder.inverse_transform([int(c)])[0]
        except Exception:
            return str(c)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
        classes = [decode_class(c) for c in clf_classes]
        return scores, classes, "probability"

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if len(scores.shape) == 1:
            scores = np.vstack([-scores, scores]).T
        classes = [decode_class(c) for c in clf_classes]
        return scores, classes, "decision"

    return None, [], "none"


def supported_macro_f1(per_label):
    vals = []
    for label in LABELS:
        info = per_label.get(label, {})
        if int(info.get("support", 0) or 0) > 0:
            vals.append(float(info.get("f1", 0.0) or 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def supported_balanced_accuracy(per_label):
    vals = []
    for label in LABELS:
        info = per_label.get(label, {})
        if int(info.get("support", 0) or 0) > 0:
            vals.append(float(info.get("recall", 0.0) or 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def lifecycle_macro_f1(per_label):
    vals = []
    for label in LIFECYCLE_LABELS:
        info = per_label.get(label, {})
        if int(info.get("support", 0) or 0) > 0:
            vals.append(float(info.get("f1", 0.0) or 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_model(model, x, rows, encoder, split_name, model_dir):
    y_true_labels = [row["label"] for row in rows]
    y_true = encoder.transform(y_true_labels)

    start = time.time()
    y_pred = model.predict(x)
    eval_seconds = time.time() - start

    y_pred_labels = encoder.inverse_transform(y_pred)
    label_ids = encoder.transform(LABELS)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )

    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    score_matrix, score_classes, score_type = predict_scores(model, x, encoder)

    predictions = []
    for i, row in enumerate(rows):
        item = {
            "example_id": row.get("example_id"),
            "split": split_name,
            "true_label": y_true_labels[i],
            "pred_label": str(y_pred_labels[i]),
            "correct": bool(y_true_labels[i] == y_pred_labels[i]),
            "source_dataset": row.get("source_dataset"),
            "scenario_type": row.get("scenario_type"),
            "operation_v1_source": row.get("operation_v1_source"),
            "memory_type": row.get("memory_type"),
            "sensitivity": row.get("sensitivity"),
        }

        if score_matrix is not None:
            scores = {}
            for j, cls in enumerate(score_classes):
                if j < score_matrix.shape[1]:
                    scores[cls] = float(score_matrix[i, j])
            item["score_type"] = score_type
            item["scores"] = scores
            if scores:
                item["top_score"] = max(scores.values())

        predictions.append(item)

    write_jsonl(model_dir / f"predictions_{split_name}.jsonl", predictions)

    all_macro_f1 = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)

    return {
        "split": split_name,
        "total": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1_all_labels": float(all_macro_f1),
        "weighted_f1": float(weighted_f1),
        "supported_macro_f1": float(supported_macro_f1(per_label)),
        "supported_balanced_accuracy": float(supported_balanced_accuracy(per_label)),
        "lifecycle_macro_f1": float(lifecycle_macro_f1(per_label)),
        "per_label": per_label,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "label_counter_true": dict(Counter(y_true_labels)),
        "label_counter_pred": dict(Counter(str(x) for x in y_pred_labels)),
        "eval_seconds": eval_seconds,
        "examples_per_second": len(rows) / eval_seconds if eval_seconds > 0 else None,
    }


def metric_from_model_metrics(metrics, selection_metric):
    split_metrics = metrics.get("metrics", {})
    parts = selection_metric.split("_", 1)
    if len(parts) != 2:
        return None

    split, metric = parts
    if split == "test":
        return None

    if selection_metric.startswith("test_synthetic_"):
        return split_metrics.get("test_synthetic", {}).get(selection_metric.replace("test_synthetic_", ""))
    if selection_metric.startswith("test_combined_"):
        return split_metrics.get("test_combined", {}).get(selection_metric.replace("test_combined_", ""))
    if selection_metric.startswith("test_benchmark_"):
        return split_metrics.get("test_benchmark", {}).get(selection_metric.replace("test_benchmark_", ""))
    if selection_metric.startswith("val_"):
        return split_metrics.get("val", {}).get(selection_metric.replace("val_", ""))

    return None


def flatten_metrics_row(model_name, encoder_key, clf_name, metrics, metrics_path, selection_metric):
    row = {
        "model": model_name,
        "encoder": encoder_key,
        "classifier": clf_name,
        "train_seconds": metrics.get("train_seconds"),
        "metrics_file": str(metrics_path),
        "selection_metric": selection_metric,
        "selection_score": metric_from_model_metrics(metrics, selection_metric),
    }

    for split_name, split_metrics in metrics.get("metrics", {}).items():
        for key in [
            "accuracy",
            "weighted_f1",
            "macro_f1_all_labels",
            "supported_macro_f1",
            "supported_balanced_accuracy",
            "lifecycle_macro_f1",
        ]:
            row[f"{split_name}_{key}"] = split_metrics.get(key)

        for label in LABELS:
            info = split_metrics.get("per_label", {}).get(label, {})
            row[f"{split_name}_{label}_support"] = info.get("support", 0)
            row[f"{split_name}_{label}_f1"] = info.get("f1", 0.0)
            row[f"{split_name}_{label}_recall"] = info.get("recall", 0.0)
            row[f"{split_name}_{label}_precision"] = info.get("precision", 0.0)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--train_split", choices=["train_balanced", "train_full"], default="train_balanced")
    parser.add_argument("--output_dir", default="processed/operation_classifier_results/embedding_v1_1_balanced_noleak_all")
    parser.add_argument("--encoders", default="all")
    parser.add_argument("--classifiers", default="all")
    parser.add_argument("--selection_metric", default="val_supported_macro_f1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_text_chars", type=int, default=20000)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--force_reencode", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    data_dir = BASE / args.data_dir
    output_dir = BASE / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train_full": data_dir / "operation_train_full_model_ready.jsonl",
        "train_balanced": data_dir / "operation_train_balanced_model_ready.jsonl",
        "val": data_dir / "operation_val_model_ready.jsonl",
        "test_benchmark": data_dir / "operation_test_benchmark_model_ready.jsonl",
        "test_synthetic": data_dir / "operation_test_synthetic_model_ready.jsonl",
        "test_combined": data_dir / "operation_test_combined_model_ready.jsonl",
    }

    train_rows = read_jsonl(split_paths[args.train_split], args.max_text_chars)
    eval_rows = {
        "val": read_jsonl(split_paths["val"], args.max_text_chars),
        "test_benchmark": read_jsonl(split_paths["test_benchmark"], args.max_text_chars),
        "test_synthetic": read_jsonl(split_paths["test_synthetic"], args.max_text_chars),
        "test_combined": read_jsonl(split_paths["test_combined"], args.max_text_chars),
    }

    encoder = LabelEncoder()
    encoder.fit(LABELS)

    y_train = encoder.transform([row["label"] for row in train_rows])

    selected_encoder_keys = parse_selection(args.encoders, ENCODERS.keys())
    unknown_encoders = [x for x in selected_encoder_keys if x not in ENCODERS]
    if unknown_encoders:
        raise ValueError(f"Unknown encoders: {unknown_encoders}")

    classifier_specs = get_classifier_specs(args.n_jobs)
    selected_classifiers = parse_selection(args.classifiers, classifier_specs.keys())
    unknown_classifiers = [x for x in selected_classifiers if x not in classifier_specs]
    if unknown_classifiers:
        raise ValueError(f"Unknown classifiers: {unknown_classifiers}")

    run_config = {
        "version": "v3.73",
        "task": "operation_governance_embedding_classification",
        "data_dir": str(data_dir),
        "train_split": args.train_split,
        "output_dir": str(output_dir),
        "labels": LABELS,
        "lifecycle_labels": LIFECYCLE_LABELS,
        "encoders": {k: ENCODERS[k] for k in selected_encoder_keys},
        "classifiers": selected_classifiers,
        "selection_metric": args.selection_metric,
        "splits": {
            args.train_split: {
                "path": str(split_paths[args.train_split]),
                **summarize_rows(train_rows),
            },
            **{
                name: {
                    "path": str(split_paths[name]),
                    **summarize_rows(rows),
                }
                for name, rows in eval_rows.items()
            },
        },
    }

    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    cache_dir = output_dir / "embedding_cache"
    comparison_rows = []
    summary = {
        **run_config,
        "models": {},
    }

    best_model = None
    best_score = None

    for encoder_key in selected_encoder_keys:
        model_name = ENCODERS[encoder_key]

        print(f"\n=== Encoder: {encoder_key} | {model_name} ===", flush=True)

        x_train = load_or_encode(
            encoder_key,
            model_name,
            args.train_split,
            train_rows,
            cache_dir,
            args.batch_size,
            args.device,
            args.max_seq_length,
            args.force_reencode,
        )

        x_eval = {}
        for split_name, rows in eval_rows.items():
            x_eval[split_name] = load_or_encode(
                encoder_key,
                model_name,
                split_name,
                rows,
                cache_dir,
                args.batch_size,
                args.device,
                args.max_seq_length,
                args.force_reencode,
            )

        for clf_name in selected_classifiers:
            clf = classifier_specs[clf_name]
            full_model_name = f"{encoder_key}__{clf_name}"
            model_dir = output_dir / encoder_key / clf_name
            model_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = model_dir / "metrics.json"

            if args.skip_existing and metrics_path.exists():
                print(f"Skipping existing {full_model_name}", flush=True)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                print(f"\n=== Training {full_model_name} ===", flush=True)

                x_train_fit = x_train
                scaler = None

                if clf_name in {"logreg_balanced", "logreg_l2", "linear_svc_balanced", "sgd_logloss_balanced", "sgd_hinge_balanced", "ridge_balanced", "mlp"}:
                    scaler = StandardScaler()
                    x_train_fit = scaler.fit_transform(x_train)

                start = time.time()
                clf.fit(x_train_fit, y_train)
                train_seconds = time.time() - start

                split_metrics = {}
                for split_name, rows in eval_rows.items():
                    x_split = x_eval[split_name]
                    if scaler is not None:
                        x_split = scaler.transform(x_split)

                    result = evaluate_model(clf, x_split, rows, encoder, split_name, model_dir)
                    split_metrics[split_name] = result

                    print(
                        f"{full_model_name} | {split_name} | "
                        f"acc={result['accuracy']:.4f} "
                        f"supp_macro_f1={result['supported_macro_f1']:.4f} "
                        f"life_macro_f1={result['lifecycle_macro_f1']:.4f}",
                        flush=True,
                    )

                payload = {
                    "model": clf,
                    "scaler": scaler,
                    "label_encoder": encoder,
                    "labels": LABELS,
                    "encoder_key": encoder_key,
                    "encoder_model_name": model_name,
                    "classifier": clf_name,
                    "train_split": args.train_split,
                }
                joblib.dump(payload, model_dir / "model.joblib")

                metrics = {
                    "model_name": full_model_name,
                    "encoder": encoder_key,
                    "encoder_model_name": model_name,
                    "classifier": clf_name,
                    "train_seconds": train_seconds,
                    "metrics": split_metrics,
                }

                metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

            summary["models"][full_model_name] = metrics

            row = flatten_metrics_row(full_model_name, encoder_key, clf_name, metrics, metrics_path, args.selection_metric)
            comparison_rows.append(row)

            score = row.get("selection_score")
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_model = full_model_name

    comparison_rows.sort(
        key=lambda r: (
            float(r.get("selection_score") or -1),
            float(r.get("test_synthetic_supported_macro_f1") or -1),
            float(r.get("test_combined_supported_macro_f1") or -1),
        ),
        reverse=True,
    )

    for i, row in enumerate(comparison_rows, 1):
        row["rank"] = i

    write_tsv(output_dir / "leaderboard.tsv", comparison_rows)
    write_jsonl(output_dir / "model_comparison.jsonl", comparison_rows)

    summary["best_model"] = best_model
    summary["best_score"] = best_score
    summary["leaderboard"] = str(output_dir / "leaderboard.tsv")
    summary["model_comparison"] = str(output_dir / "model_comparison.jsonl")

    (output_dir / "summary_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "version": "v3.73",
        "total_models": len(comparison_rows),
        "best_model": best_model,
        "best_score": best_score,
        "selection_metric": args.selection_metric,
        "leaderboard": str(output_dir / "leaderboard.tsv"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
