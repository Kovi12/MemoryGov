import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
OUT = BASE / "processed/final_results_summary_v1_1"
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


def write_tsv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except Exception:
        return 0.0


def bar(labels, values, title, ylabel, output, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}" if isinstance(h, float) else str(h),
                    (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    non_llm_lb = read_tsv("processed/operation_classifier_results/non_llm_v1_1_balanced_noleak_all/leaderboard_supported_metrics.tsv")
    emb_lb = read_tsv("processed/operation_classifier_results/embedding_v1_1_balanced_noleak_all/leaderboard.tsv")
    op_analysis = read_json("processed/operation_final_analysis_v1_1/final_model_error_analysis_report.json")
    retrieval_readme = Path("processed/frozen_results/retrieval_v1/README.txt")

    non_llm_lb = sorted(non_llm_lb, key=lambda r: f(r, "val_supported_macro_f1"), reverse=True)
    emb_lb = sorted(emb_lb, key=lambda r: f(r, "val_supported_macro_f1"), reverse=True)

    best_sparse = non_llm_lb[0]
    best_embedding = emb_lb[0]

    comparison = [
        {
            "family": "sparse_tfidf_svd",
            "model": best_sparse["model"],
            "val_supported_macro_f1": best_sparse.get("val_supported_macro_f1"),
            "test_synthetic_supported_macro_f1": best_sparse.get("test_synthetic_supported_macro_f1"),
            "test_combined_supported_macro_f1": best_sparse.get("test_combined_supported_macro_f1"),
            "test_benchmark_accuracy": best_sparse.get("test_benchmark_accuracy"),
        },
        {
            "family": "embedding",
            "model": best_embedding["model"],
            "val_supported_macro_f1": best_embedding.get("val_supported_macro_f1"),
            "test_synthetic_supported_macro_f1": best_embedding.get("test_synthetic_supported_macro_f1"),
            "test_combined_supported_macro_f1": best_embedding.get("test_combined_supported_macro_f1"),
            "test_benchmark_accuracy": best_embedding.get("test_benchmark_accuracy"),
        },
    ]

    write_tsv(TAB / "best_operation_models_by_family.tsv", comparison)
    write_tsv(TAB / "operation_non_llm_top10.tsv", non_llm_lb[:10])
    write_tsv(TAB / "operation_embedding_top10.tsv", emb_lb[:10])

    labels = [r["family"] for r in comparison]
    bar(
        labels,
        [f(r, "val_supported_macro_f1") for r in comparison],
        "Best operation models by family: validation supported macro F1",
        "Supported macro F1",
        FIG / "operation_best_family_val_supported_macro_f1.png",
        ylim=(0, 1.05),
    )
    bar(
        labels,
        [f(r, "test_synthetic_supported_macro_f1") for r in comparison],
        "Best operation models by family: synthetic lifecycle test",
        "Supported macro F1",
        FIG / "operation_best_family_test_synthetic.png",
        ylim=(0, 1.05),
    )

    if op_analysis:
        synth = op_analysis["splits"]["test_synthetic"]
        confusions = synth["top_confusions"]
        write_tsv(TAB / "operation_test_synthetic_top_confusions.tsv", confusions)

        per_label = synth["per_true_label_error_rate"]
        lab = list(per_label.keys())
        vals = [per_label[x]["error_rate"] for x in lab]
        bar(
            lab,
            vals,
            "Operation Governor v1.1: synthetic test error rate by true label",
            "Error rate",
            FIG / "operation_test_synthetic_error_rate_by_label.png",
            ylim=(0, 1.05),
        )

    retrieval_text = retrieval_readme.read_text(encoding="utf-8") if retrieval_readme.exists() else ""

    summary = f"""# Final Results Summary v1.1

## Retrieval Governor v1

{retrieval_text}

## Operation Governor v1.1

Best sparse/non-LLM model:
- {best_sparse["model"]}
- val_supported_macro_f1 = {best_sparse.get("val_supported_macro_f1")}
- test_synthetic_supported_macro_f1 = {best_sparse.get("test_synthetic_supported_macro_f1")}
- test_combined_supported_macro_f1 = {best_sparse.get("test_combined_supported_macro_f1")}
- test_benchmark_accuracy = {best_sparse.get("test_benchmark_accuracy")}

Best embedding model:
- {best_embedding["model"]}
- val_supported_macro_f1 = {best_embedding.get("val_supported_macro_f1")}
- test_synthetic_supported_macro_f1 = {best_embedding.get("test_synthetic_supported_macro_f1")}
- test_combined_supported_macro_f1 = {best_embedding.get("test_combined_supported_macro_f1")}
- test_benchmark_accuracy = {best_embedding.get("test_benchmark_accuracy")}

Conclusion:
The sparse TF-IDF/SGD model remains the strongest operation-governance baseline. Dense embedding models perform strongly, but do not outperform the sparse no-leak baseline.

## Error Analysis

The final selected operation model makes very few synthetic lifecycle test errors. The remaining mistakes are mostly policy-boundary cases:
- ASK_USER vs QUARANTINE for sensitive health/private information.
- IGNORE vs WRITE for habits or potentially useful preferences.
- FORGET vs UPDATE when deletion/removal wording overlaps with memory mutation.

## Evaluation Interpretation

The benchmark test split is WRITE-only after label audit and should be interpreted only as a WRITE sanity check. Full lifecycle-governance evaluation is based on the synthetic lifecycle test split and the combined split.
"""

    (OUT / "summary.md").write_text(summary, encoding="utf-8")

    print(json.dumps({
        "output_dir": str(OUT),
        "summary": str(OUT / "summary.md"),
        "figures": [str(p) for p in sorted(FIG.glob("*.png"))],
        "tables": [str(p) for p in sorted(TAB.glob("*.tsv"))],
    }, indent=2))


if __name__ == "__main__":
    main()
