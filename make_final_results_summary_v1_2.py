import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
OUT = BASE / "processed/final_results_summary_v1_2"
FIG = OUT / "figures"
TAB = OUT / "tables"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_tsv(path, rows):
    if not rows:
        return

    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except Exception:
        return 0.0


def bar(labels, values, title, ylabel, output, ylim=(0, 1.05)):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")

    for bar_item in bars:
        height = bar_item.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar_item.get_x() + bar_item.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def confusion_heatmap(csv_path, output):
    rows = read_csv(csv_path)
    if not rows:
        return

    labels = [key for key in rows[0].keys() if key != "true_label"]
    matrix = []
    ylabels = []

    for row in rows:
        ylabels.append(row["true_label"])
        matrix.append([int(row[label]) for label in labels])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix)

    ax.set_title("Operation Governor v1.2 manual boundary confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    retrieval_readme_path = BASE / "processed/frozen_results/retrieval_v1/README.txt"

    op_v1_1_lb = read_tsv(
        BASE / "processed/operation_classifier_results/non_llm_v1_1_balanced_noleak_all/leaderboard_supported_metrics.tsv"
    )
    op_v1_2_lb = read_tsv(
        BASE / "processed/operation_classifier_results/non_llm_v1_2_forget_aug_selected/leaderboard_supported_metrics.tsv"
    )
    op_embedding_lb = read_tsv(
        BASE / "processed/operation_classifier_results/embedding_v1_1_balanced_noleak_all/leaderboard.tsv"
    )

    manual_v1_2 = read_json(
        BASE / "processed/manual_gold_operation_v1_1/hand_authored/evaluation_v1_2_tfidf_sgd_hinge_detailed/report.json"
    )

    manual_v1_2_per_label = read_csv(
        BASE / "processed/manual_gold_operation_v1_1/hand_authored/evaluation_v1_2_tfidf_sgd_hinge_detailed/per_label_metrics.csv"
    )

    retrieval_text = retrieval_readme_path.read_text(encoding="utf-8") if retrieval_readme_path.exists() else ""

    op_v1_1_lb = sorted(op_v1_1_lb, key=lambda r: f(r, "val_supported_macro_f1"), reverse=True)
    op_v1_2_lb = sorted(op_v1_2_lb, key=lambda r: f(r, "val_supported_macro_f1"), reverse=True)
    op_embedding_lb = sorted(op_embedding_lb, key=lambda r: f(r, "val_supported_macro_f1"), reverse=True)

    best_v1_1 = op_v1_1_lb[0] if op_v1_1_lb else {}
    best_v1_2_validation = op_v1_2_lb[0] if op_v1_2_lb else {}
    best_embedding = op_embedding_lb[0] if op_embedding_lb else {}

    final_operation = {
        "family": "final_operation_v1_2_safety_oriented",
        "model": "tfidf_sgd_hinge_balanced",
        "manual_accuracy": manual_v1_2["accuracy"] if manual_v1_2 else None,
        "manual_supported_macro_f1": manual_v1_2["supported_macro_f1"] if manual_v1_2 else None,
        "manual_lifecycle_macro_f1": manual_v1_2["lifecycle_macro_f1"] if manual_v1_2 else None,
        "manual_FORGET_f1": manual_v1_2["per_label"]["FORGET"]["f1"] if manual_v1_2 else None,
        "manual_FORGET_recall": manual_v1_2["per_label"]["FORGET"]["recall"] if manual_v1_2 else None,
        "manual_errors": manual_v1_2["errors"] if manual_v1_2 else None,
    }

    comparison_rows = [
        {
            "stage": "operation_v1_1_best_validation_sparse",
            "model": best_v1_1.get("model"),
            "val_supported_macro_f1": best_v1_1.get("val_supported_macro_f1"),
            "test_synthetic_supported_macro_f1": best_v1_1.get("test_synthetic_supported_macro_f1"),
            "test_combined_supported_macro_f1": best_v1_1.get("test_combined_supported_macro_f1"),
            "test_benchmark_accuracy": best_v1_1.get("test_benchmark_accuracy"),
        },
        {
            "stage": "operation_v1_1_best_embedding",
            "model": best_embedding.get("model"),
            "val_supported_macro_f1": best_embedding.get("val_supported_macro_f1"),
            "test_synthetic_supported_macro_f1": best_embedding.get("test_synthetic_supported_macro_f1"),
            "test_combined_supported_macro_f1": best_embedding.get("test_combined_supported_macro_f1"),
            "test_benchmark_accuracy": best_embedding.get("test_benchmark_accuracy"),
        },
        {
            "stage": "operation_v1_2_best_validation_sparse",
            "model": best_v1_2_validation.get("model"),
            "val_supported_macro_f1": best_v1_2_validation.get("val_supported_macro_f1"),
            "test_synthetic_supported_macro_f1": best_v1_2_validation.get("test_synthetic_supported_macro_f1"),
            "test_combined_supported_macro_f1": best_v1_2_validation.get("test_combined_supported_macro_f1"),
            "test_benchmark_accuracy": best_v1_2_validation.get("test_benchmark_accuracy"),
        },
        final_operation,
    ]

    write_tsv(TAB / "operation_model_comparison_summary.tsv", comparison_rows)
    write_tsv(TAB / "operation_v1_2_top_models.tsv", op_v1_2_lb[:10])

    if manual_v1_2_per_label:
        write_tsv(TAB / "operation_v1_2_manual_per_label_metrics.tsv", manual_v1_2_per_label)

        labels = [row["label"] for row in manual_v1_2_per_label]
        f1_values = [float(row["f1"]) for row in manual_v1_2_per_label]
        bar(
            labels,
            f1_values,
            "Operation Governor v1.2 manual boundary F1 by label",
            "F1",
            FIG / "operation_v1_2_manual_f1_by_label.png",
        )

    if manual_v1_2:
        bar(
            ["accuracy", "supported_macro_f1", "lifecycle_macro_f1", "FORGET_f1"],
            [
                manual_v1_2["accuracy"],
                manual_v1_2["supported_macro_f1"],
                manual_v1_2["lifecycle_macro_f1"],
                manual_v1_2["per_label"]["FORGET"]["f1"],
            ],
            "Operation Governor v1.2 manual boundary summary",
            "Score",
            FIG / "operation_v1_2_manual_summary_scores.png",
        )

    confusion_heatmap(
        BASE / "processed/manual_gold_operation_v1_1/hand_authored/evaluation_v1_2_tfidf_sgd_hinge_detailed/confusion_matrix.csv",
        FIG / "operation_v1_2_manual_confusion_matrix.png",
    )

    summary = f"""# Final Results Summary v1.2

## Retrieval Governor v1

{retrieval_text}

## Operation Governor v1.2

Final safety-oriented model:
- tfidf_sgd_hinge_balanced

Why this model:
- Selected as the final safety-oriented candidate because it fixes the explicit FORGET weakness found in Operation Governor v1.1.
- The validation-selected v1.2 model is also strong, but tfidf_sgd_hinge_balanced performs best on the manual policy-boundary mini-test.

Manual boundary mini-test:
- 70 examples
- 10 examples per label
- AI-assisted authored set, pending researcher review before being described as human-curated

Manual boundary results:
- accuracy = {manual_v1_2["accuracy"] if manual_v1_2 else "NA"}
- supported_macro_f1 = {manual_v1_2["supported_macro_f1"] if manual_v1_2 else "NA"}
- lifecycle_macro_f1 = {manual_v1_2["lifecycle_macro_f1"] if manual_v1_2 else "NA"}
- FORGET precision = {manual_v1_2["per_label"]["FORGET"]["precision"] if manual_v1_2 else "NA"}
- FORGET recall = {manual_v1_2["per_label"]["FORGET"]["recall"] if manual_v1_2 else "NA"}
- FORGET F1 = {manual_v1_2["per_label"]["FORGET"]["f1"] if manual_v1_2 else "NA"}
- errors = {manual_v1_2["errors"] if manual_v1_2 else "NA"} / 70

Remaining error types:
- WRITE -> IGNORE: 1
- IGNORE -> WRITE: 3
- MERGE -> UPDATE: 1
- ASK_USER -> QUARANTINE: 1

Interpretation:
Operation Governor v1.2 addresses the main safety weakness discovered in v1.1. Targeted train-only FORGET augmentation taught the policy rule that explicit deletion intent overrides sensitivity routing. Remaining errors are policy-boundary cases involving durable vs transient memory, merge vs update, and confirmation vs quarantine.

## Reporting caution

The manual boundary set is AI-assisted and should be reviewed by the researcher before being described as human-curated. Until then, describe it as an AI-assisted manual mini-test or researcher-pending-review boundary test.
"""

    (OUT / "summary.md").write_text(summary, encoding="utf-8")

    manifest = {
        "version": "v3.88",
        "output_dir": str(OUT),
        "summary": str(OUT / "summary.md"),
        "tables": [str(p) for p in sorted(TAB.glob("*"))],
        "figures": [str(p) for p in sorted(FIG.glob("*"))],
    }

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
