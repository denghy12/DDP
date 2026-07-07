import argparse
import csv
import json
import os

import torch

from src.helper_functions.detail_report import binary_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select one global EMOTIC threshold from pure-val sweeps "
            "across all continual-learning tasks."
        )
    )
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def load_task_sweeps(sweep_dir, num_tasks):
    sweeps = []
    for task_id in range(num_tasks):
        path = os.path.join(
            sweep_dir, f"task{task_id}", "threshold_sweep.json"
        )
        with open(path, "r", encoding="utf-8") as fp:
            sweep = json.load(fp)
        metadata = sweep["metadata"]
        if metadata.get("eval_splits") != ["val"]:
            raise ValueError(
                f"Task {task_id} is not a pure-val sweep: "
                f"eval_splits={metadata.get('eval_splits')}"
            )
        if int(metadata["task"]) != task_id:
            raise ValueError(
                f"Expected task {task_id}, found task {metadata['task']}"
            )
        score_path = os.path.join(sweep_dir, f"task{task_id}", "task_scores.pt")
        score_data = torch.load(score_path, map_location="cpu")
        sweeps.append(
            {
                "metadata": metadata,
                "scores": score_data["scores"].float(),
                "targets": score_data["targets"].bool(),
            }
        )
    return sweeps


def thresholds(start, end, step):
    count = int(round((end - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def f1_metrics(scores, targets, threshold):
    predictions = scores.gt(threshold)
    overall = binary_counts(predictions, targets)
    class_f1 = []
    for class_id in range(scores.shape[1]):
        counts = binary_counts(
            predictions[:, class_id], targets[:, class_id]
        )
        class_f1.append(counts["f1"])
    return 100.0 * overall["f1"], 100.0 * sum(class_f1) / len(class_f1)


def aggregate_rows(sweeps, threshold_values):
    task_metrics_by_threshold = []
    aggregated = []
    for threshold in threshold_values:
        task_rows = []
        for task_id, sweep in enumerate(sweeps):
            of1, cf1 = f1_metrics(
                sweep["scores"], sweep["targets"], threshold
            )
            task_rows.append(
                {"task": task_id, "oF1": of1, "cF1": cf1}
            )

        mean_of1 = sum(row["oF1"] for row in task_rows) / len(task_rows)
        mean_cf1 = sum(row["cF1"] for row in task_rows) / len(task_rows)
        aggregated.append(
            {
                "threshold": threshold,
                "mean_oF1": mean_of1,
                "mean_cF1": mean_cf1,
                "mean_oF1_cF1": (mean_of1 + mean_cf1) / 2.0,
                "final_task_oF1": task_rows[-1]["oF1"],
                "final_task_cF1": task_rows[-1]["cF1"],
                "final_task_mean_oF1_cF1": (
                    task_rows[-1]["oF1"] + task_rows[-1]["cF1"]
                )
                / 2.0,
            }
        )
        task_metrics_by_threshold.append(
            {"threshold": threshold, "tasks": task_rows}
        )
    return aggregated, task_metrics_by_threshold


def main():
    args = parse_args()
    sweeps = load_task_sweeps(args.sweep_dir, args.num_tasks)
    threshold_values = thresholds(
        args.threshold_start, args.threshold_end, args.threshold_step
    )
    rows, task_metrics_by_threshold = aggregate_rows(
        sweeps, threshold_values
    )
    best_mean_of1 = max(rows, key=lambda row: row["mean_oF1"])
    best_mean_cf1 = max(rows, key=lambda row: row["mean_cF1"])
    best_balanced = max(rows, key=lambda row: row["mean_oF1_cF1"])

    metadata = {
        "selection_split": "val",
        "num_tasks": args.num_tasks,
        "criterion": "mean across tasks of (oF1 + cF1) / 2",
        "threshold_start": args.threshold_start,
        "threshold_end": args.threshold_end,
        "threshold_step": args.threshold_step,
        "selected_threshold": best_balanced["threshold"],
        "best_mean_oF1": best_mean_of1,
        "best_mean_cF1": best_mean_cf1,
        "best_mean_oF1_cF1": best_balanced,
        "task_samples": [
            sweep["metadata"]["samples"] for sweep in sweeps
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(
        args.output_dir, "global_threshold_selection.json"
    )
    csv_path = os.path.join(
        args.output_dir, "global_threshold_selection.csv"
    )
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "metadata": metadata,
                "rows": rows,
                "task_metrics_by_threshold": task_metrics_by_threshold,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
