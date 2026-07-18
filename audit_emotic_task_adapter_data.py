"""Audit sample overlap and partial-label visibility for EMOTIC B5-C3."""

import argparse
import csv
import json
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np

from emotic_task_adapter_bank import (
    TASK_CLASS_RANGES,
    prepare_task_training_subset,
)
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Audit EMOTIC task Adapter data")
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--output_dir", default="./output/emotic_b5c3_task_adapter_audit"
    )
    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    return parser.parse_args()


def sample_id(dataset, index):
    bbox = np.asarray(dataset.body_bboxes[index]).reshape(-1).tolist()
    normalized_bbox = [round(float(value), 4) for value in bbox[:4]]
    return f"{dataset.file_paths[index]}|{normalized_bbox}"


def write_html(path, summary):
    task_rows = []
    for row in summary["tasks"]:
        task_rows.append(
            "<tr>"
            f"<td>{row['task']}</td>"
            f"<td>{escape(str(row['class_range']))}</td>"
            f"<td>{row['full_unique_samples']}</td>"
            f"<td>{row['current_positive_labels']}</td>"
            f"<td>{row['ignored_old_positive_labels']}</td>"
            f"<td>{row['ignored_future_positive_labels']}</td>"
            "</tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Task Adapter Data Audit</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>EMOTIC B5-C3 Task Adapter Data Audit</h1>"
        f"<p>Total train persons: {summary['total_train_persons']}; "
        f"cross-task persons: {summary['cross_task_persons']}.</p>"
        "<table><tr><th>Task</th><th>Classes</th><th>Full persons</th>"
        "<th>Current positives</th><th>Ignored old positives</th>"
        "<th>Ignored future positives</th></tr>"
        + "".join(task_rows)
        + "</table>"
        f"<h2>Membership distribution</h2><pre>{escape(json.dumps(summary['task_membership_count_distribution'], indent=2))}</pre>"
        f"<h2>Overlap matrix</h2><pre>{escape(json.dumps(summary['task_overlap_matrix'], indent=2))}</pre>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EMOTIC(args.data_root, train=True, transform=None, input_mode="full")
    labels = dense_labels(dataset)
    task_sets = []
    task_rows = []
    fewshot = {}

    for task_id, (low, high) in enumerate(TASK_CLASS_RANGES):
        selected, mask, sampling = prepare_task_training_subset(
            labels, task_id, "full", seed=0
        )
        selected_set = set(selected.tolist())
        task_sets.append(selected_set)
        task_rows.append(
            {
                "task": task_id,
                "class_range": [low, high],
                "class_names": dataset.classes[low:high],
                "full_unique_samples": len(selected_set),
                "supervised_entries": int(mask.sum().item()),
                "current_positive_labels": sampling["current_positive_labels"],
                "ignored_old_positive_labels": sampling[
                    "ignored_old_positive_labels"
                ],
                "ignored_future_positive_labels": sampling[
                    "ignored_future_positive_labels"
                ],
            }
        )
        fewshot[str(task_id)] = []
        for seed in args.seeds:
            shot_selected, shot_mask, shot_sampling = prepare_task_training_subset(
                labels,
                task_id,
                "16shot",
                seed=seed,
                shots_per_class=args.shots_per_class,
            )
            fewshot[str(task_id)].append(
                {
                    "seed": seed,
                    "unique_samples": int(shot_selected.numel()),
                    "supervised_entries": int(shot_mask.sum().item()),
                    "sampling": shot_sampling["sampling"],
                }
            )

    overlap = [
        [len(task_sets[left].intersection(task_sets[right])) for right in range(8)]
        for left in range(8)
    ]
    memberships = [sum(index in task_set for task_set in task_sets) for index in range(len(dataset))]
    distribution = Counter(memberships)
    cross_task_indices = [index for index, count in enumerate(memberships) if count > 1]
    summary = {
        "protocol": {
            "dataset": "EMOTIC train",
            "incremental_protocol": "B5-C3",
            "sample_definition": "person instance (image path + body bbox)",
            "full_definition": "all persons with at least one current-task positive",
            "fewshot_definition": "exactly 16 supervised positive anchors per current class",
            "old_and_future_labels_masked": True,
        },
        "total_train_persons": len(dataset),
        "unique_sample_ids": len({sample_id(dataset, index) for index in range(len(dataset))}),
        "cross_task_persons": len(cross_task_indices),
        "task_membership_count_distribution": {
            str(key): value for key, value in sorted(distribution.items())
        },
        "task_overlap_matrix": overlap,
        "tasks": task_rows,
        "fewshot": fewshot,
        "cross_task_examples": [
            {
                "dataset_index": index,
                "sample_id": sample_id(dataset, index),
                "labels": [dataset.classes[class_id] for class_id in dataset.targets[index]],
                "tasks": [task_id for task_id, task_set in enumerate(task_sets) if index in task_set],
            }
            for index in cross_task_indices[:100]
        ],
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(
        output_dir / "task_overlap_matrix.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["task"] + [f"task{task_id}" for task_id in range(8)])
        for task_id, row in enumerate(overlap):
            writer.writerow([f"task{task_id}"] + row)
    write_html(output_dir / "sampling_report.html", summary)
    print(json.dumps({key: summary[key] for key in (
        "total_train_persons", "cross_task_persons", "task_membership_count_distribution"
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

