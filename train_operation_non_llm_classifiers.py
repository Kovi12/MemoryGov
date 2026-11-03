import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC


LABELS = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]

BASE = Path(__file__).resolve().parent

DEFAULT_SPLITS = {
    "train_full": BASE / "processed/memorygovbench_operation_v1_model_ready/operation_train_full_model_ready.jsonl",
    "train_balanced": BASE / "processed/memorygovbench_operation_v1_model_ready/operation_train_balanced_model_ready.jsonl",
    "val": BASE / "processed/memorygovbench_operation_v1_model_ready/operation_val_model_ready.jsonl",
    "test_benchmark": BASE / "processed/memorygovbench_operation_v1_model_ready/operation_test_benchmark_model_ready.jsonl",
    "test_synthetic": BASE / "processed/memorygovbench_operation_v1_model_ready/operation_test_synthetic_model_ready.jsonl",
    "test_combined": BASE / "processed/memorygovbench_operation_v1_model_ready/operation_test_combined_model_ready.jsonl",
}


def read_jsonl(path, max_text_chars):
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
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize_split(rows):
    return {
        "total": len(rows),
        "label_counter": dict(Counter(row["label"] for row in rows)),
        "source_dataset_counter": dict(Counter(row.get("source_dataset") for row in rows)),
        "scenario_counter": dict(Counter(row.get("scenario_type") for row in rows)),
        "operation_v1_source_counter": dict(Counter(row.get("operation_v1_source") for row in rows)),
    }


def make_word_vectorizer(max_features):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )


def make_char_vectorizer(max_features):
    return TfidfVectorizer(
        max_features=max_features,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
        sublinear_tf=True,
    )


def make_svd_pipeline(max_features, svd_components, clf):
    return Pipeline([
        ("tfidf", make_word_vectorizer(max_features)),
        ("svd", TruncatedSVD(n_components=svd_components, random_state=42)),
        ("norm", Normalizer(copy=False)),
        ("clf", clf),
    ])


def get_model_specs(args):
    n_jobs = args.n_jobs

    specs = {
        "tfidf_logreg_balanced": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", n_jobs=n_jobs)),
        ]),
        "tfidf_logreg_l2": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", LogisticRegression(max_iter=2500, n_jobs=n_jobs)),
        ]),
        "tfidf_linear_svc_balanced": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", LinearSVC(class_weight="balanced", dual="auto", max_iter=5000)),
        ]),
        "tfidf_sgd_logloss_balanced": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-4, random_state=42)),
        ]),
        "tfidf_sgd_hinge_balanced": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", SGDClassifier(loss="hinge", class_weight="balanced", max_iter=2000, tol=1e-4, random_state=42)),
        ]),
        "tfidf_ridge_balanced": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", RidgeClassifier(class_weight="balanced")),
        ]),
        "tfidf_passive_aggressive_balanced": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", PassiveAggressiveClassifier(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "tfidf_complement_nb": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", ComplementNB()),
        ]),
        "tfidf_multinomial_nb": Pipeline([
            ("tfidf", make_word_vectorizer(args.max_features)),
            ("clf", MultinomialNB()),
        ]),
        "char_logreg_balanced": Pipeline([
            ("tfidf", make_char_vectorizer(args.max_features)),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", n_jobs=n_jobs)),
        ]),
        "char_sgd_logloss_balanced": Pipeline([
            ("tfidf", make_char_vectorizer(args.max_features)),
            ("clf", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-4, random_state=42)),
        ]),
        "svd_logreg_balanced": make_svd_pipeline(
            args.max_features,
            args.svd_components,
            LogisticRegression(max_iter=2500, class_weight="balanced", n_jobs=n_jobs),
        ),
        "svd_linear_svc_balanced": make_svd_pipeline(
            args.max_features,
            args.svd_components,
            LinearSVC(class_weight="balanced", dual="auto", max_iter=5000),
        ),
        "svd_random_forest_balanced": make_svd_pipeline(
            args.max_features,
            args.svd_components,
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=42,
            ),
        ),
        "svd_extra_trees_balanced": make_svd_pipeline(
            args.max_features,
            args.svd_components,
            ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=42,
            ),
        ),
        "svd_hist_gradient_boosting": make_svd_pipeline(
            args.max_features,
            args.svd_components,
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                l2_regularization=0.02,
                random_state=42,
            ),
        ),
        "svd_mlp": make_svd_pipeline(
            args.max_features,
            args.svd_components,
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=80,
                early_stopping=True,
                random_state=42,
            ),
        ),
    }

    fast = [
        "tfidf_logreg_balanced",
        "tfidf_linear_svc_balanced",
        "tfidf_sgd_logloss_balanced",
        "tfidf_ridge_balanced",
        "char_sgd_logloss_balanced",
        "svd_logreg_balanced",
        "svd_linear_svc_balanced",
    ]

    if args.models == "all":
        names = list(specs.keys())
    elif args.models == "fast":
        names = fast
    else:
        names = [name.strip() for name in args.models.split(",") if name.strip()]

    missing = [name for name in names if name not in specs]
    if missing:
        raise ValueError(f"Unknown models: {missing}")

    return {name: specs[name] for name in names}, list(specs.keys())


def predict_scores(model, x, encoder):
    clf_classes = getattr(model, "classes_", None)
    if clf_classes is None and hasattr(model, "named_steps"):
        clf_classes = getattr(model.named_steps.get("clf"), "classes_", None)

    def class_label(class_id):
        try:
            return encoder.inverse_transform([int(class_id)])[0]
        except Exception:
            return str(class_id)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
        classes = [class_label(c) for c in clf_classes]
        return scores, classes, "probability"

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if len(scores.shape) == 1:
            scores = np.vstack([-scores, scores]).T
        classes = [class_label(c) for c in clf_classes]
        return scores, classes, "decision"

    return None, [], "none"


def evaluate_model(model, rows, encoder, split_name, output_dir):
    x = [row["text"] for row in rows]
    y_true_labels = [row["label"] for row in rows]
    y_true = encoder.transform(y_true_labels)

    start = time.time()
    y_pred = model.predict(x)
    eval_seconds = time.time() - start

    y_pred_labels = encoder.inverse_transform(y_pred)

    label_ids = encoder.transform(LABELS)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)

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

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

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

    write_jsonl(output_dir / f"predictions_{split_name}.jsonl", predictions)

    return {
        "split": split_name,
        "total": len(rows),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_label": per_label,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_counter_true": dict(Counter(y_true_labels)),
        "label_counter_pred": dict(Counter(str(x) for x in y_pred_labels)),
        "eval_seconds": eval_seconds,
        "examples_per_second": len(rows) / eval_seconds if eval_seconds > 0 else None,
    }


def metric_value(metrics, key):
    if key in metrics:
        return metrics[key]
    if key.startswith("val_"):
        return metrics.get("val", {}).get(key.replace("val_", ""))
    if key.startswith("test_synthetic_"):
        return metrics.get("test_synthetic", {}).get(key.replace("test_synthetic_", ""))
    if key.startswith("test_combined_"):
        return metrics.get("test_combined", {}).get(key.replace("test_combined_", ""))
    if key.startswith("test_benchmark_"):
        return metrics.get("test_benchmark", {}).get(key.replace("test_benchmark_", ""))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_split", choices=["train_balanced", "train_full"], default="train_balanced")
    parser.add_argument("--output_dir", default="processed/operation_classifier_results/non_llm_v1_balanced")
    parser.add_argument("--models", default="all")
    parser.add_argument("--selection_metric", default="val_macro_f1")
    parser.add_argument("--max_features", type=int, default=100000)
    parser.add_argument("--max_text_chars", type=int, default=20000)
    parser.add_argument("--svd_components", type=int, default=300)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--data_dir", default="processed/memorygovbench_operation_v1_model_ready")
    args = parser.parse_args()

    output_dir = BASE / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = BASE / args.data_dir
    split_paths = {
        "train_full": data_dir / "operation_train_full_model_ready.jsonl",
        "train_balanced": data_dir / "operation_train_balanced_model_ready.jsonl",
        "val": data_dir / "operation_val_model_ready.jsonl",
        "test_benchmark": data_dir / "operation_test_benchmark_model_ready.jsonl",
        "test_synthetic": data_dir / "operation_test_synthetic_model_ready.jsonl",
        "test_combined": data_dir / "operation_test_combined_model_ready.jsonl",
    }
    train_rows = read_jsonl(split_paths[args.train_split], args.max_text_chars)
    eval_splits = {
        "val": read_jsonl(split_paths["val"], args.max_text_chars),
        "test_benchmark": read_jsonl(split_paths["test_benchmark"], args.max_text_chars),
        "test_synthetic": read_jsonl(split_paths["test_synthetic"], args.max_text_chars),
        "test_combined": read_jsonl(split_paths["test_combined"], args.max_text_chars),
    }

    all_labels = sorted(set([row["label"] for row in train_rows] + LABELS))
    encoder = LabelEncoder()
    encoder.fit(all_labels)

    x_train = [row["text"] for row in train_rows]
    y_train = encoder.transform([row["label"] for row in train_rows])

    model_specs, available_models = get_model_specs(args)

    run_config = {
        "version": "v3.64",
        "task": "operation_governance_non_llm_classification",
        "train_split": args.train_split,
        "output_dir": str(output_dir),
        "selected_models": list(model_specs.keys()),
        "available_models": available_models,
        "labels": LABELS,
        "max_features": args.max_features,
        "max_text_chars": args.max_text_chars,
        "svd_components": args.svd_components,
        "splits": {
            args.train_split: {
                "path": str(split_paths[args.train_split]),
                **summarize_split(train_rows),
            },
            **{
                name: {
                    "path": str(split_paths[name]),
                    **summarize_split(rows),
                }
                for name, rows in eval_splits.items()
            },
        },
    }

    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    comparison_rows = []
    summary = {
        **run_config,
        "models": {},
    }

    best_model = None
    best_score = None

    for model_name, model in model_specs.items():
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = model_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"Skipping existing {model_name}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"\n=== Training {model_name} ===", flush=True)
            start = time.time()
            model.fit(x_train, y_train)
            train_seconds = time.time() - start

            joblib.dump({
                "model": model,
                "label_encoder": encoder,
                "labels": LABELS,
                "train_split": args.train_split,
            }, model_dir / "model.joblib")

            split_metrics = {}
            for split_name, rows in eval_splits.items():
                result = evaluate_model(model, rows, encoder, split_name, model_dir)
                split_metrics[split_name] = result
                print(
                    f"{model_name} | {split_name} | "
                    f"acc={result['accuracy']:.4f} "
                    f"bal_acc={result['balanced_accuracy']:.4f} "
                    f"macro_f1={result['macro_f1']:.4f} "
                    f"weighted_f1={result['weighted_f1']:.4f}",
                    flush=True,
                )

            metrics = {
                "model_name": model_name,
                "train_seconds": train_seconds,
                "metrics": split_metrics,
            }

            metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        summary["models"][model_name] = metrics

        row = {
            "model": model_name,
            "train_seconds": metrics.get("train_seconds"),
        }

        for split_name, split_metrics in metrics.get("metrics", {}).items():
            row[f"{split_name}_accuracy"] = split_metrics.get("accuracy")
            row[f"{split_name}_balanced_accuracy"] = split_metrics.get("balanced_accuracy")
            row[f"{split_name}_macro_f1"] = split_metrics.get("macro_f1")
            row[f"{split_name}_weighted_f1"] = split_metrics.get("weighted_f1")
            for label in LABELS:
                row[f"{split_name}_{label}_f1"] = split_metrics.get("per_label", {}).get(label, {}).get("f1")
                row[f"{split_name}_{label}_recall"] = split_metrics.get("per_label", {}).get(label, {}).get("recall")
                row[f"{split_name}_{label}_precision"] = split_metrics.get("per_label", {}).get(label, {}).get("precision")

        score = metric_value(metrics.get("metrics", {}), args.selection_metric)
        row["selection_metric"] = args.selection_metric
        row["selection_score"] = score
        row["metrics_file"] = str(metrics_path)

        comparison_rows.append(row)

        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_model = model_name

    comparison_rows.sort(key=lambda r: r["selection_score"] if r["selection_score"] is not None else -1, reverse=True)

    for i, row in enumerate(comparison_rows, 1):
        row["rank"] = i

    write_tsv(output_dir / "leaderboard.tsv", comparison_rows)
    write_jsonl(output_dir / "model_comparison.jsonl", comparison_rows)

    summary["best_model"] = best_model
    summary["best_score"] = best_score
    summary["selection_metric"] = args.selection_metric
    summary["leaderboard"] = str(output_dir / "leaderboard.tsv")
    summary["model_comparison"] = str(output_dir / "model_comparison.jsonl")

    (output_dir / "summary_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "version": "v3.64",
        "best_model": best_model,
        "best_score": best_score,
        "selection_metric": args.selection_metric,
        "leaderboard": str(output_dir / "leaderboard.tsv"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
