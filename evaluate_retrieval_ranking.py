import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


VERSION = "v3.49"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stable_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def get_nested(row, path):
    cur = row
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def first_present(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def get_example_id(row):
    return first_present(row, [
        "example_id",
        "id",
        "uid",
        "task_id",
        "source_example_id",
        "row_id",
    ])


def get_true_label(row, positive_label, negative_label):
    value = first_present(row, [
        "label",
        "gold_label",
        "target",
        "true_label",
        "y_true",
        "decision",
        "primary_decision",
    ])

    if value is None:
        value = get_nested(row, ["gold", "primary_decision"])

    if isinstance(value, bool):
        return positive_label if value else negative_label

    if isinstance(value, int):
        return positive_label if value == 1 else negative_label

    if isinstance(value, str):
        value = value.strip()
        if value in {positive_label, negative_label}:
            return value
        upper = value.upper()
        if upper in {positive_label, negative_label}:
            return upper

    return None


def get_pred_label(row, positive_label, negative_label):
    value = first_present(row, [
        "pred_label",
        "predicted_label",
        "prediction",
        "y_pred",
        "pred",
    ])

    if isinstance(value, bool):
        return positive_label if value else negative_label

    if isinstance(value, int):
        return positive_label if value == 1 else negative_label

    if isinstance(value, str):
        value = value.strip()
        if value in {positive_label, negative_label}:
            return value
        upper = value.upper()
        if upper in {positive_label, negative_label}:
            return upper

    return None


def get_score(row, positive_label):
    numeric_keys = [
        "positive_score",
        "positive_probability",
        "positive_prob",
        "prob_positive",
        "p_positive",
        "retrieve_score",
        "retrieve_probability",
        "retrieval_score",
        "score",
        "y_score",
        "decision_score",
        "probability",
    ]

    for key in numeric_keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    dict_keys = [
        "probabilities",
        "probas",
        "label_probabilities",
        "label_probs",
        "scores",
        "label_scores",
        "decision_scores",
    ]

    for key in dict_keys:
        value = row.get(key)
        if isinstance(value, dict):
            for label_key in [positive_label, positive_label.lower(), positive_label.upper()]:
                if label_key in value and isinstance(value[label_key], (int, float)):
                    return float(value[label_key])

    labels = row.get("labels")
    probs = row.get("probabilities")
    if isinstance(labels, list) and isinstance(probs, list) and len(labels) == len(probs):
        for label, prob in zip(labels, probs):
            if str(label).upper() == positive_label and isinstance(prob, (int, float)):
                return float(prob)

    pred_label = get_pred_label(row, positive_label, "SUPPRESS")
    if pred_label == positive_label:
        return 1.0
    if pred_label is not None:
        return 0.0

    return None


def get_group_key(row):
    split_group = row.get("split_group")
    if isinstance(split_group, dict):
        return stable_json(split_group)

    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("split_group"), dict):
        return stable_json(metadata["split_group"])

    annotation = row.get("annotation_metadata")
    if isinstance(annotation, dict):
        for keys in [
            ["sample_id", "qa_idx"],
            ["question_id"],
            ["qa_pair_id"],
            ["task_id"],
        ]:
            found = {}
            for key in keys:
                if key in annotation:
                    found[key] = annotation[key]
            if len(found) == len(keys):
                return stable_json(found)

    for keys in [
        ["source_dataset", "source_file", "question_id"],
        ["source_dataset", "source_file", "qa_pair_id"],
        ["source_dataset", "sample_id", "qa_idx"],
        ["source_dataset", "row_idx", "qa_idx"],
    ]:
        found = {}
        for key in keys:
            if key in row:
                found[key] = row[key]
        if len(found) == len(keys):
            return stable_json(found)

    current_query = row.get("current_query") or row.get("query")
    expected_answer = row.get("expected_answer") or row.get("expected_answers")
    source_dataset = row.get("source_dataset")
    source_file = row.get("source_file")

    if current_query is not None:
        return stable_json({
            "source_dataset": source_dataset,
            "source_file": source_file,
            "current_query": current_query,
            "expected_answer": expected_answer,
        })

    return get_example_id(row)


def get_source_dataset(row):
    return row.get("source_dataset") or get_nested(row, ["metadata", "source_dataset"]) or "unknown"


def get_source_file(row):
    return row.get("source_file") or get_nested(row, ["metadata", "source_file"]) or "unknown"


def get_scenario(row):
    return row.get("scenario_type") or get_nested(row, ["metadata", "scenario_type"]) or "unknown"


def get_candidate_text(row):
    candidate = row.get("candidate_memory")
    if isinstance(candidate, dict) and isinstance(candidate.get("text"), str):
        return candidate["text"]

    for key in ["candidate_text", "memory_text", "text", "input", "prompt"]:
        value = row.get(key)
        if isinstance(value, str):
            return value

    return ""


def truncate_text(text, max_chars=500):
    if text is None:
        return ""
    text = str(text).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def find_prediction_path(results_dir, threshold_dir, model_name, split):
    candidates = []

    if "__" in model_name:
        encoder, classifier = model_name.split("__", 1)
        candidates.extend([
            threshold_dir / encoder / classifier / f"predictions_{split}_threshold_tuned.jsonl",
            threshold_dir / encoder / classifier / f"predictions_{split}.jsonl",
            threshold_dir / model_name / f"predictions_{split}_threshold_tuned.jsonl",
            threshold_dir / model_name / f"predictions_{split}.jsonl",
            results_dir / encoder / classifier / f"predictions_{split}_threshold_tuned.jsonl",
            results_dir / encoder / classifier / f"predictions_{split}.jsonl",
            results_dir / model_name / f"predictions_{split}_threshold_tuned.jsonl",
            results_dir / model_name / f"predictions_{split}.jsonl",
        ])
    else:
        candidates.extend([
            threshold_dir / model_name / f"predictions_{split}_threshold_tuned.jsonl",
            threshold_dir / model_name / f"predictions_{split}.jsonl",
            results_dir / model_name / f"predictions_{split}_threshold_tuned.jsonl",
            results_dir / model_name / f"predictions_{split}.jsonl",
        ])

    for path in candidates:
        if path.exists():
            return path

    matches = list(results_dir.rglob(f"predictions_{split}_threshold_tuned.jsonl"))
    matches += list(results_dir.rglob(f"predictions_{split}.jsonl"))
    for path in matches:
        parts = set(path.parts)
        if "__" in model_name:
            encoder, classifier = model_name.split("__", 1)
            if encoder in parts and classifier in parts:
                return path
        elif model_name in parts:
            return path

    return None


def load_models_from_leaderboard(path, top_n):
    models = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            model = row.get("model") or row.get("model_name")
            if model:
                models.append(model)

    if top_n and top_n > 0:
        models = models[:top_n]

    return models


def dcg_at_k(relevances, k):
    total = 0.0
    for idx, rel in enumerate(relevances[:k]):
        if rel:
            total += 1.0 / math.log2(idx + 2)
    return total


def evaluate_group_entries(group_entries, ks):
    sorted_entries = sorted(group_entries, key=lambda x: (-x["score"], x["order"]))
    positive_count = sum(1 for item in sorted_entries if item["is_positive"])

    if positive_count == 0:
        return None

    ranks = []
    for idx, item in enumerate(sorted_entries, start=1):
        if item["is_positive"]:
            ranks.append(idx)

    first_positive_rank = ranks[0] if ranks else None
    relevances = [1 if item["is_positive"] else 0 for item in sorted_entries]

    metrics = {
        "candidate_count": len(sorted_entries),
        "positive_count": positive_count,
        "first_positive_rank": first_positive_rank,
        "mrr": 1.0 / first_positive_rank if first_positive_rank else 0.0,
    }

    for k in ks:
        top = sorted_entries[:k]
        top_positive_count = sum(1 for item in top if item["is_positive"])
        denom_precision = min(k, len(sorted_entries))
        ideal = [1] * min(positive_count, k)
        idcg = dcg_at_k(ideal, k)

        metrics[f"hit_at_{k}"] = 1.0 if top_positive_count > 0 else 0.0
        metrics[f"recall_at_{k}"] = top_positive_count / positive_count
        metrics[f"precision_at_{k}"] = top_positive_count / denom_precision if denom_precision else 0.0
        metrics[f"ndcg_at_{k}"] = dcg_at_k(relevances, k) / idcg if idcg > 0 else 0.0

    return metrics


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def summarize_group_metrics(group_metrics, ks):
    summary = {
        "groups_with_positive": len(group_metrics),
        "candidate_count": {
            "min": min((m["candidate_count"] for m in group_metrics), default=0),
            "p50": percentile([m["candidate_count"] for m in group_metrics], 0.50),
            "p90": percentile([m["candidate_count"] for m in group_metrics], 0.90),
            "max": max((m["candidate_count"] for m in group_metrics), default=0),
        },
        "positive_count": {
            "min": min((m["positive_count"] for m in group_metrics), default=0),
            "p50": percentile([m["positive_count"] for m in group_metrics], 0.50),
            "p90": percentile([m["positive_count"] for m in group_metrics], 0.90),
            "max": max((m["positive_count"] for m in group_metrics), default=0),
        },
        "mrr": mean(m["mrr"] for m in group_metrics),
        "mean_first_positive_rank": mean(m["first_positive_rank"] for m in group_metrics if m["first_positive_rank"]),
    }

    for k in ks:
        summary[f"hit_at_{k}"] = mean(m[f"hit_at_{k}"] for m in group_metrics)
        summary[f"recall_at_{k}"] = mean(m[f"recall_at_{k}"] for m in group_metrics)
        summary[f"precision_at_{k}"] = mean(m[f"precision_at_{k}"] for m in group_metrics)
        summary[f"ndcg_at_{k}"] = mean(m[f"ndcg_at_{k}"] for m in group_metrics)

    return summary


def align_examples_and_predictions(examples, predictions, positive_label, negative_label):
    rows = []

    if len(examples) == len(predictions):
        for idx, (example, prediction) in enumerate(zip(examples, predictions)):
            score = get_score(prediction, positive_label)
            if score is None:
                score = get_score(example, positive_label)
            if score is None:
                continue

            true_label = get_true_label(example, positive_label, negative_label)
            if true_label is None:
                true_label = get_true_label(prediction, positive_label, negative_label)
            if true_label is None:
                continue

            rows.append({
                "example": example,
                "prediction": prediction,
                "score": float(score),
                "true_label": true_label,
                "order": idx,
            })

        return rows

    pred_by_id = {}
    for prediction in predictions:
        example_id = get_example_id(prediction)
        if example_id is not None:
            pred_by_id[str(example_id)] = prediction

    for idx, example in enumerate(examples):
        example_id = get_example_id(example)
        if example_id is None:
            continue

        prediction = pred_by_id.get(str(example_id))
        if prediction is None:
            continue

        score = get_score(prediction, positive_label)
        if score is None:
            score = get_score(example, positive_label)
        if score is None:
            continue

        true_label = get_true_label(example, positive_label, negative_label)
        if true_label is None:
            true_label = get_true_label(prediction, positive_label, negative_label)
        if true_label is None:
            continue

        rows.append({
            "example": example,
            "prediction": prediction,
            "score": float(score),
            "true_label": true_label,
            "order": idx,
        })

    return rows


def evaluate_model(model_name, pred_path, model_ready_path, output_dir, split, positive_label, negative_label, ks):
    examples = read_jsonl(model_ready_path)
    predictions = read_jsonl(pred_path)
    aligned_rows = align_examples_and_predictions(examples, predictions, positive_label, negative_label)

    groups = defaultdict(list)

    for row in aligned_rows:
        example = row["example"]
        group_key = get_group_key(example)
        if group_key is None:
            continue

        groups[group_key].append({
            "example_id": get_example_id(example),
            "score": row["score"],
            "true_label": row["true_label"],
            "is_positive": row["true_label"] == positive_label,
            "order": row["order"],
            "source_dataset": get_source_dataset(example),
            "source_file": get_source_file(example),
            "scenario_type": get_scenario(example),
            "candidate_text": get_candidate_text(example),
        })

    group_details = []
    group_metrics = []
    missed_top10 = []

    by_source_groups = defaultdict(list)
    by_scenario_groups = defaultdict(list)

    for group_key, entries in groups.items():
        metrics = evaluate_group_entries(entries, ks)
        if metrics is None:
            continue

        sorted_entries = sorted(entries, key=lambda x: (-x["score"], x["order"]))
        source_dataset = sorted_entries[0]["source_dataset"]
        scenario_type = sorted_entries[0]["scenario_type"]

        detail = {
            "model": model_name,
            "split": split,
            "group_key": group_key,
            "source_dataset": source_dataset,
            "source_file": sorted_entries[0]["source_file"],
            "scenario_type": scenario_type,
            **metrics,
            "top_10": [
                {
                    "rank": idx + 1,
                    "example_id": item["example_id"],
                    "score": item["score"],
                    "true_label": item["true_label"],
                    "candidate_text": truncate_text(item["candidate_text"]),
                }
                for idx, item in enumerate(sorted_entries[:10])
            ],
        }

        group_details.append(detail)
        group_metrics.append(metrics)
        by_source_groups[source_dataset].append(metrics)
        by_scenario_groups[scenario_type].append(metrics)

        if metrics.get("hit_at_10", 0.0) == 0.0:
            missed_top10.append(detail)

    summary = summarize_group_metrics(group_metrics, ks)

    by_source = {
        source: summarize_group_metrics(metrics, ks)
        for source, metrics in sorted(by_source_groups.items())
    }

    by_scenario = {
        scenario: summarize_group_metrics(metrics, ks)
        for scenario, metrics in sorted(by_scenario_groups.items())
    }

    label_counter = Counter(row["true_label"] for row in aligned_rows)
    source_counter = Counter(get_source_dataset(row["example"]) for row in aligned_rows)
    scenario_counter = Counter(get_scenario(row["example"]) for row in aligned_rows)

    report = {
        "version": VERSION,
        "task": "retrieval_ranking_evaluation",
        "model": model_name,
        "split": split,
        "prediction_path": str(pred_path),
        "model_ready_path": str(model_ready_path),
        "output_dir": str(output_dir),
        "positive_label": positive_label,
        "negative_label": negative_label,
        "ks": ks,
        "total_model_ready_examples": len(examples),
        "total_prediction_rows": len(predictions),
        "aligned_rows": len(aligned_rows),
        "groups_total": len(groups),
        "groups_with_positive": len(group_metrics),
        "label_counter": dict(label_counter),
        "source_dataset_counter": dict(source_counter),
        "scenario_counter": dict(scenario_counter),
        "overall": summary,
        "by_source_dataset": by_source,
        "by_scenario_type": by_scenario,
        "missed_top10_count": len(missed_top10),
    }

    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    write_json(model_out / f"ranking_report_{split}.json", report)
    write_jsonl(model_out / f"group_ranking_details_{split}.jsonl", group_details)
    write_jsonl(model_out / f"missed_top10_{split}.jsonl", missed_top10[:500])

    return report


def write_ranking_leaderboard(path, reports, ks):
    fieldnames = [
        "rank",
        "model",
        "split",
        "groups_with_positive",
        "aligned_rows",
        "mrr",
        "mean_first_positive_rank",
    ]

    for k in ks:
        fieldnames.extend([
            f"hit_at_{k}",
            f"recall_at_{k}",
            f"precision_at_{k}",
            f"ndcg_at_{k}",
        ])

    fieldnames.extend([
        "prediction_path",
        "report_path",
    ])

    rows = []
    for report in reports:
        overall = report["overall"]
        row = {
            "model": report["model"],
            "split": report["split"],
            "groups_with_positive": report["groups_with_positive"],
            "aligned_rows": report["aligned_rows"],
            "mrr": overall.get("mrr", 0.0),
            "mean_first_positive_rank": overall.get("mean_first_positive_rank", 0.0),
            "prediction_path": report["prediction_path"],
            "report_path": str(Path(report["output_dir"]) / report["model"] / f"ranking_report_{report['split']}.json"),
        }

        for k in ks:
            row[f"hit_at_{k}"] = overall.get(f"hit_at_{k}", 0.0)
            row[f"recall_at_{k}"] = overall.get(f"recall_at_{k}", 0.0)
            row[f"precision_at_{k}"] = overall.get(f"precision_at_{k}", 0.0)
            row[f"ndcg_at_{k}"] = overall.get(f"ndcg_at_{k}", 0.0)

        rows.append(row)

    rows.sort(key=lambda x: (x.get("mrr", 0.0), x.get("recall_at_5", 0.0)), reverse=True)

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", default="processed/embedding_classifier_results/retrieval_all")
    parser.add_argument("--threshold_dir", default="processed/retrieval_threshold_tuning/embedding_all")
    parser.add_argument("--leaderboard", default="processed/retrieval_threshold_tuning/embedding_all/threshold_tuned_leaderboard.tsv")
    parser.add_argument("--model_ready_path", default="processed/memorygovbench_model_ready/retrieval_test_natural_model_ready.jsonl")
    parser.add_argument("--output_dir", default="processed/retrieval_ranking_eval/embedding_all")
    parser.add_argument("--split", default="test_natural")
    parser.add_argument("--models", default="")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--positive_label", default="RETRIEVE")
    parser.add_argument("--negative_label", default="SUPPRESS")
    parser.add_argument("--ks", default="1,3,5,10,20")

    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    threshold_dir = Path(args.threshold_dir)
    leaderboard = Path(args.leaderboard)
    model_ready_path = Path(args.model_ready_path)
    output_dir = Path(args.output_dir)
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    if args.models.strip():
        models = [x.strip() for x in args.models.split(",") if x.strip()]
    else:
        models = load_models_from_leaderboard(leaderboard, args.top_n)

    reports = []
    missing = []

    for model_name in models:
        pred_path = find_prediction_path(results_dir, threshold_dir, model_name, args.split)

        if pred_path is None:
            missing.append(model_name)
            print(f"Missing predictions for {model_name}")
            continue

        print(f"Evaluating {model_name}")
        report = evaluate_model(
            model_name=model_name,
            pred_path=pred_path,
            model_ready_path=model_ready_path,
            output_dir=output_dir,
            split=args.split,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            ks=ks,
        )
        reports.append(report)

    write_ranking_leaderboard(output_dir / f"ranking_leaderboard_{args.split}.tsv", reports, ks)

    final_report = {
        "version": VERSION,
        "task": "retrieval_ranking_evaluation_batch",
        "results_dir": str(results_dir),
        "threshold_dir": str(threshold_dir),
        "leaderboard": str(leaderboard),
        "model_ready_path": str(model_ready_path),
        "output_dir": str(output_dir),
        "split": args.split,
        "models_requested": models,
        "models_evaluated": [r["model"] for r in reports],
        "missing_prediction_models": missing,
        "ranking_leaderboard": str(output_dir / f"ranking_leaderboard_{args.split}.tsv"),
    }

    write_json(output_dir / f"ranking_eval_batch_report_{args.split}.json", final_report)
    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
