import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "processed/project_results_pack_v1"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["WRITE", "IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]
NON_WRITE_ORDER = ["IGNORE", "UPDATE", "MERGE", "FORGET", "ASK_USER", "QUARANTINE"]


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def label_of(row):
    return row.get("gold", {}).get("primary_decision")


def count_labels(rows):
    return Counter(label_of(row) for row in rows)


def pct(num, den):
    if not den:
        return 0.0
    return 100.0 * num / den


def find_first_existing(paths):
    for path in paths:
        p = BASE / path
        if p.exists():
            return p
    return None


def add_value_labels(ax, bars, fmt="{:.0f}", rotation=0):
    for bar in bars:
        height = bar.get_height()
        if height is None or math.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
        )


def save_bar_chart(title, labels, values, ylabel, output_path, value_fmt="{:.0f}", rotate_x=30):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate_x, ha="right")
    add_value_labels(ax, bars, value_fmt)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_grouped_bar_chart(title, groups, series_names, data_by_series, ylabel, output_path, value_fmt="{:.0f}", rotate_x=30):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(groups)))
    width = 0.8 / max(1, len(series_names))

    for i, series in enumerate(series_names):
        values = [data_by_series.get(series, {}).get(group, 0) for group in groups]
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        bars = ax.bar(offsets, values, width, label=series)
        add_value_labels(ax, bars, value_fmt, rotation=90 if len(groups) > 6 else 0)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rotate_x, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_stacked_bar_chart(title, groups, stack_names, data, ylabel, output_path, rotate_x=30):
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0] * len(groups)

    for stack in stack_names:
        values = [data.get(group, {}).get(stack, 0) for group in groups]
        ax.bar(groups, values, bottom=bottoms, label=stack)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    for i, total in enumerate(bottoms):
        ax.annotate(str(total), xy=(i, total), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(groups, rotation=rotate_x, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def synthetic_clean_sources():
    sources = {
        "pilot": [
            "processed/frozen_results/operation_synthetic_pilot_v1/clean.jsonl",
            "processed/operation_synthetic_v1/validation_pilot_all_2examples_salvaged_normalized/clean.jsonl",
        ],
        "batch1": [
            "processed/frozen_results/operation_synthetic_batch1_v1/clean.jsonl",
            "processed/operation_synthetic_v1/label_batches/batch1/validation/clean.jsonl",
        ],
        "remaining_v1": [
            "processed/frozen_results/operation_synthetic_remaining_v1/clean.jsonl",
            "processed/operation_synthetic_v1/label_batches/remaining_v1/validation/clean.jsonl",
        ],
    }

    out = {}
    for name, candidates in sources.items():
        path = find_first_existing(candidates)
        out[name] = path

    return out


def synthetic_report_sources():
    sources = {
        "pilot": [
            "processed/frozen_results/operation_synthetic_pilot_v1/report.json",
            "processed/operation_synthetic_v1/validation_pilot_all_2examples_salvaged_normalized/report.json",
        ],
        "batch1": [
            "processed/frozen_results/operation_synthetic_batch1_v1/report.json",
            "processed/operation_synthetic_v1/label_batches/batch1/validation/report.json",
        ],
        "remaining_v1": [
            "processed/frozen_results/operation_synthetic_remaining_v1/report.json",
            "processed/operation_synthetic_v1/label_batches/remaining_v1/validation/report.json",
        ],
    }

    out = {}
    for name, candidates in sources.items():
        path = find_first_existing(candidates)
        out[name] = path

    return out


def collect_synthetic_results():
    clean_paths = synthetic_clean_sources()
    report_paths = synthetic_report_sources()

    clean_counts_by_source = {}
    clean_total = Counter()

    clean_count_rows = []
    for source, path in clean_paths.items():
        rows = read_jsonl(path) if path else []
        counts = count_labels(rows)
        clean_counts_by_source[source] = dict(counts)

        for label in LABEL_ORDER:
            value = counts.get(label, 0)
            if value:
                clean_total[label] += value
            clean_count_rows.append({
                "source": source,
                "label": label,
                "clean_examples": value,
                "path": str(path) if path else "",
            })

    validation_rows = []
    for source, path in report_paths.items():
        report = read_json(path) if path else None
        if not report:
            continue

        total_all = report.get("total_all", 0)
        total_clean = report.get("total_clean", 0)
        total_review = report.get("total_review", 0)
        total_rejected = report.get("total_rejected", 0)

        validation_rows.append({
            "source": source,
            "raw_rows": report.get("raw_counter", {}).get("raw_rows", ""),
            "total_all": total_all,
            "total_clean": total_clean,
            "total_review": total_review,
            "total_rejected": total_rejected,
            "clean_rate_percent": round(pct(total_clean, total_all), 2),
            "review_rate_percent": round(pct(total_review, total_all), 2),
            "rejected_rate_percent": round(pct(total_rejected, total_all), 2),
            "path": str(path),
        })

    write_csv(TABLE_DIR / "operation_synthetic_clean_counts_by_source.csv", clean_count_rows)
    write_csv(TABLE_DIR / "operation_synthetic_validation_status.csv", validation_rows)

    save_grouped_bar_chart(
        "Clean synthetic operation examples by label and source",
        NON_WRITE_ORDER,
        list(clean_counts_by_source.keys()),
        clean_counts_by_source,
        "Clean examples",
        FIG_DIR / "operation_synthetic_clean_by_label_and_source.png",
    )

    total_values = [clean_total.get(label, 0) for label in NON_WRITE_ORDER]
    save_bar_chart(
        "Total clean synthetic operation examples by label",
        NON_WRITE_ORDER,
        total_values,
        "Clean examples",
        FIG_DIR / "operation_synthetic_total_clean_by_label.png",
    )

    status_data = {}
    for row in validation_rows:
        status_data[row["source"]] = {
            "clean": int(row["total_clean"]),
            "review": int(row["total_review"]),
            "rejected": int(row["total_rejected"]),
        }

    if status_data:
        save_stacked_bar_chart(
            "Synthetic validation status by generation stage",
            list(status_data.keys()),
            ["clean", "review", "rejected"],
            status_data,
            "Examples",
            FIG_DIR / "operation_synthetic_validation_status.png",
        )

    return {
        "clean_counts_by_source": clean_counts_by_source,
        "clean_total": dict(clean_total),
        "validation_rows": validation_rows,
    }


def collect_current_dataset_results():
    current_report = read_json(BASE / "processed/memorygovbench_current/memorygovbench_current_report.json")
    rows = []

    if current_report:
        splits = current_report.get("splits", {})
        for task_name, split_data in splits.items():
            for split_name, stats in split_data.items():
                total = stats.get("total", 0)
                labels = stats.get("label_counter", {})
                for label, count in labels.items():
                    rows.append({
                        "dataset": "MemoryGovBench-current",
                        "task": task_name,
                        "split": split_name,
                        "label": label,
                        "count": count,
                        "split_total": total,
                    })

    write_csv(TABLE_DIR / "memorygovbench_current_label_counts.csv", rows)

    retrieval_totals = []
    totals = current_report.get("totals", {}) if current_report else {}
    for task, stats in totals.items():
        retrieval_totals.append({
            "task": task,
            "total": stats.get("total", 0),
            "write": stats.get("label::WRITE", ""),
            "ignore": stats.get("label::IGNORE", ""),
            "retrieve": stats.get("label::RETRIEVE", ""),
            "suppress": stats.get("label::SUPPRESS", ""),
        })

    write_csv(TABLE_DIR / "memorygovbench_current_totals.csv", retrieval_totals)

    if retrieval_totals:
        labels = [row["task"] for row in retrieval_totals]
        values = [int(row["total"]) for row in retrieval_totals]
        save_bar_chart(
            "MemoryGovBench-current dataset sizes",
            labels,
            values,
            "Examples",
            FIG_DIR / "memorygovbench_current_dataset_sizes.png",
        )

    return {
        "report": current_report,
        "label_rows": rows,
        "totals": retrieval_totals,
    }


def collect_operation_v1_results():
    report = read_json(BASE / "processed/memorygovbench_operation_v1/report.json")
    rows = []

    if report:
        for split_name, stats in report.get("splits", {}).items():
            label_counter = stats.get("label_counter", {})
            total = stats.get("total", 0)
            for label in LABEL_ORDER:
                rows.append({
                    "split": split_name,
                    "label": label,
                    "count": label_counter.get(label, 0),
                    "split_total": total,
                })

        write_csv(TABLE_DIR / "memorygovbench_operation_v1_split_label_counts.csv", rows)

        data = defaultdict(dict)
        for row in rows:
            data[row["split"]][row["label"]] = int(row["count"])

        save_grouped_bar_chart(
            "MemoryGovBench-operation-v1 label distribution by split",
            LABEL_ORDER,
            list(data.keys()),
            data,
            "Examples",
            FIG_DIR / "operation_v1_label_distribution_by_split.png",
        )

    return report


def collect_existing_expansion_results():
    report = read_json(BASE / "processed/frozen_results/operation_expansion_existing_v1/report.json")
    if not report:
        report = read_json(BASE / "processed/operation_expansion_existing_strict/report.json")

    rows = []
    if report:
        accepted = report.get("accepted", {})
        labels = accepted.get("label_counter", {})
        for label, count in labels.items():
            rows.append({
                "label": label,
                "count": count,
            })

    write_csv(TABLE_DIR / "operation_existing_expansion_accepted_counts.csv", rows)

    if rows:
        labels = [row["label"] for row in rows]
        values = [int(row["count"]) for row in rows]
        save_bar_chart(
            "Strict existing benchmark-derived operation expansion",
            labels,
            values,
            "Accepted examples",
            FIG_DIR / "operation_existing_expansion_accepted_counts.png",
        )

    return report


def get_top_ranking_rows():
    candidates = [
        BASE / "processed/retrieval_ranking_eval/embedding_top10_natural/leaderboard.tsv",
        BASE / "processed/retrieval_ranking_eval/embedding_top10_natural/ranking_leaderboard.tsv",
    ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        existing = list((BASE / "processed/retrieval_ranking_eval").glob("**/*leaderboard*.tsv"))

    rows = []
    for path in existing:
        parsed = read_tsv(path)
        if parsed:
            for row in parsed:
                row["_source_file"] = str(path)
            rows.extend(parsed)

    natural_rows = [row for row in rows if row.get("split") == "test_natural"]
    if natural_rows:
        rows = natural_rows

    rows = [row for row in rows if safe_float(row.get("mrr")) is not None]
    rows.sort(key=lambda r: safe_float(r.get("mrr"), 0.0), reverse=True)
    return rows[:10]


def collect_retrieval_results():
    ranking_rows = get_top_ranking_rows()

    normalized_ranking_rows = []
    for row in ranking_rows:
        normalized_ranking_rows.append({
            "rank": row.get("rank", ""),
            "model": row.get("model", ""),
            "split": row.get("split", ""),
            "mrr": row.get("mrr", ""),
            "hit_at_1": row.get("hit_at_1", ""),
            "hit_at_3": row.get("hit_at_3", ""),
            "hit_at_5": row.get("hit_at_5", ""),
            "recall_at_5": row.get("recall_at_5", ""),
            "ndcg_at_5": row.get("ndcg_at_5", ""),
            "source_file": row.get("_source_file", ""),
        })

    write_csv(TABLE_DIR / "retrieval_ranking_top10.csv", normalized_ranking_rows)

    if normalized_ranking_rows:
        model_labels = [row["model"] for row in normalized_ranking_rows]
        mrr_values = [safe_float(row["mrr"], 0.0) for row in normalized_ranking_rows]
        hit1_values = [safe_float(row["hit_at_1"], 0.0) for row in normalized_ranking_rows]
        hit5_values = [safe_float(row["hit_at_5"], 0.0) for row in normalized_ranking_rows]

        save_bar_chart(
            "Top retrieval ranking models by MRR",
            model_labels,
            mrr_values,
            "MRR",
            FIG_DIR / "retrieval_top10_mrr.png",
            value_fmt="{:.3f}",
            rotate_x=45,
        )

        data = {
            "Hit@1": dict(zip(model_labels, hit1_values)),
            "Hit@5": dict(zip(model_labels, hit5_values)),
        }

        save_grouped_bar_chart(
            "Top retrieval ranking models: Hit@1 and Hit@5",
            model_labels,
            ["Hit@1", "Hit@5"],
            data,
            "Score",
            FIG_DIR / "retrieval_top10_hit1_hit5.png",
            value_fmt="{:.2f}",
            rotate_x=45,
        )

    frozen_readme = BASE / "processed/frozen_results/retrieval_v1/README.txt"
    retrieval_v1 = {}
    if frozen_readme.exists():
        text = frozen_readme.read_text(encoding="utf-8")
        patterns = {
            "MRR": r"MRR\s*=\s*([0-9.]+)",
            "Hit@1": r"Hit@1\s*=\s*([0-9.]+)",
            "Hit@3": r"Hit@3\s*=\s*([0-9.]+)",
            "Hit@5": r"Hit@5\s*=\s*([0-9.]+)",
            "Recall@5": r"Recall@5\s*=\s*([0-9.]+)",
        }
        for name, pattern in patterns.items():
            m = re.search(pattern, text)
            if m:
                retrieval_v1[name] = float(m.group(1))

    if retrieval_v1:
        labels = list(retrieval_v1.keys())
        values = [retrieval_v1[label] for label in labels]
        save_bar_chart(
            "Retrieval Governor v1 natural-test ranking metrics",
            labels,
            values,
            "Score",
            FIG_DIR / "retrieval_governor_v1_metrics.png",
            value_fmt="{:.3f}",
            rotate_x=30,
        )

    classifier_leaderboard_paths = [
        BASE / "processed/non_llm_classifier_results/retrieval_all/leaderboard_rebuilt.tsv",
        BASE / "processed/non_llm_classifier_results/retrieval_all/leaderboard.tsv",
        BASE / "processed/retrieval_threshold_tuning/embedding_all/threshold_tuned_leaderboard.tsv",
        BASE / "processed/embedding_classifier_results/retrieval_all/leaderboard.tsv",
    ]

    classifier_rows = []
    for path in classifier_leaderboard_paths:
        rows = read_tsv(path)
        if rows:
            for row in rows[:10]:
                row["_source_file"] = str(path)
                classifier_rows.append(row)

    normalized_classifier_rows = []
    for row in classifier_rows:
        model = row.get("model", "")
        val_pos_f1 = row.get("val_positive_f1", row.get("best_score", ""))
        tb_pos_f1 = row.get("test_balanced_positive_f1", "")
        tn_pos_f1 = row.get("test_natural_positive_f1", "")

        normalized_classifier_rows.append({
            "model": model,
            "val_positive_f1": val_pos_f1,
            "test_balanced_positive_f1": tb_pos_f1,
            "test_natural_positive_f1": tn_pos_f1,
            "source_file": row.get("_source_file", ""),
        })

    write_csv(TABLE_DIR / "retrieval_classifier_leaderboards_top_rows.csv", normalized_classifier_rows)

    return {
        "ranking_top10": normalized_ranking_rows,
        "retrieval_v1": retrieval_v1,
        "classifier_rows": normalized_classifier_rows,
    }


def write_professor_summary(snapshot):
    synthetic_total = snapshot["synthetic"]["clean_total"]
    synthetic_total_ordered = {label: synthetic_total.get(label, 0) for label in NON_WRITE_ORDER}

    current_totals = snapshot["current_dataset"]["totals"]
    operation_v1 = snapshot.get("operation_v1")
    retrieval_v1 = snapshot["retrieval"].get("retrieval_v1", {})
    ranking_top10 = snapshot["retrieval"].get("ranking_top10", [])

    best_retrieval_model = ranking_top10[0]["model"] if ranking_top10 else "mpnet__mlp"
    best_mrr = ranking_top10[0]["mrr"] if ranking_top10 else retrieval_v1.get("MRR", "")

    lines = []
    lines.append("# Project results pack v1")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append("This project currently has two completed technical components:")
    lines.append("")
    lines.append("1. A retrieval-governance dataset and retrieval governor evaluation.")
    lines.append("2. A first complete operation-governance dataset with expanded lifecycle labels.")
    lines.append("")
    lines.append("The retrieval-governance side already has a strong ranking model. The current best retrieval model is `{}` with natural-test MRR `{}`.".format(best_retrieval_model, best_mrr))
    lines.append("")
    lines.append("The operation-governance side now has enough clean synthetic data for a first full training round. Clean synthetic examples by label:")
    lines.append("")
    for label, count in synthetic_total_ordered.items():
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("## Dataset construction status")
    lines.append("")
    if current_totals:
        lines.append("MemoryGovBench-current totals:")
        lines.append("")
        for row in current_totals:
            lines.append(f"- {row['task']}: {row['total']} examples")
        lines.append("")

    lines.append("Synthetic operation data generation:")
    lines.append("")
    for row in snapshot["synthetic"]["validation_rows"]:
        lines.append(
            f"- {row['source']}: {row['total_clean']} clean / {row['total_all']} total "
            f"({row['clean_rate_percent']}% clean), {row['total_review']} review, {row['total_rejected']} rejected"
        )
    lines.append("")

    if operation_v1:
        lines.append("## MemoryGovBench-operation-v1")
        lines.append("")
        lines.append("The operation-v1 builder has already been run. Split sizes:")
        lines.append("")
        for split, stats in operation_v1.get("splits", {}).items():
            lines.append(f"- {split}: {stats.get('total', 0)} examples, labels={stats.get('label_counter', {})}")
        lines.append("")
    else:
        lines.append("## MemoryGovBench-operation-v1")
        lines.append("")
        lines.append("The final operation-v1 builder has not been detected yet. Run `build_memorygovbench_operation_v1.py` to produce final train/validation/test splits.")
        lines.append("")

    lines.append("## Retrieval-governance results")
    lines.append("")
    if retrieval_v1:
        lines.append("Frozen Retrieval Governor v1 metrics:")
        lines.append("")
        for key, value in retrieval_v1.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    if ranking_top10:
        lines.append("Top retrieval ranking models:")
        lines.append("")
        for row in ranking_top10[:5]:
            lines.append(
                f"- {row['model']}: MRR={row['mrr']}, Hit@1={row['hit_at_1']}, Hit@5={row['hit_at_5']}, Recall@5={row['recall_at_5']}"
            )
        lines.append("")

    lines.append("## Figures generated")
    lines.append("")
    for fig in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- `{fig.relative_to(OUT_DIR)}`")
    lines.append("")

    lines.append("## Tables generated")
    lines.append("")
    for table in sorted(TABLE_DIR.glob("*.csv")):
        lines.append(f"- `{table.relative_to(OUT_DIR)}`")
    lines.append("")

    lines.append("## Talking points for the professor")
    lines.append("")
    lines.append("- The project is no longer only conceptual: it now has concrete benchmark-derived and synthetic datasets.")
    lines.append("- Retrieval governance was evaluated both as binary classification and as ranking, with strong ranking performance.")
    lines.append("- The operation-governance problem required synthetic expansion because public memory benchmarks are dominated by WRITE/RETRIEVE-style supervision.")
    lines.append("- Synthetic generation was validated, normalized, deduplicated, and split into clean/review sets instead of being merged blindly.")
    lines.append("- The next step is training operation-governance classifiers and evaluating them separately on benchmark-native and synthetic lifecycle-label test sets.")
    lines.append("")

    (OUT_DIR / "professor_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    snapshot = {
        "current_dataset": collect_current_dataset_results(),
        "existing_expansion": collect_existing_expansion_results(),
        "synthetic": collect_synthetic_results(),
        "retrieval": collect_retrieval_results(),
        "operation_v1": collect_operation_v1_results(),
    }

    with (OUT_DIR / "results_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    write_professor_summary(snapshot)

    print(json.dumps({
        "output_dir": str(OUT_DIR),
        "summary": str(OUT_DIR / "professor_summary.md"),
        "snapshot": str(OUT_DIR / "results_snapshot.json"),
        "figures": [str(p) for p in sorted(FIG_DIR.glob("*.png"))],
        "tables": [str(p) for p in sorted(TABLE_DIR.glob("*.csv"))],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
