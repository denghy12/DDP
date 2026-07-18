"""Summarize Full and 16-shot task-routed Adapter Bank experiments."""

import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize task Adapter Banks")
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_task_adapter_bank_comparison",
    )
    return parser.parse_args()


def run_path(output_root, mode, seed):
    return (
        Path(output_root)
        / f"emotic_ddp_task_adapter_bank_{mode}_feature_difference_seed{seed}"
        / "evaluation_summary.json"
    )


def collect_mode(output_root, mode, seeds):
    runs = []
    task_runs = []
    for seed in seeds:
        path = run_path(output_root, mode, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        aggregate = data["aggregate"]
        runs.append(
            {
                "method": f"Task Adapter Bank {mode}",
                "mode": mode,
                "seed": seed,
                "average_mAP": aggregate["average_mAP"],
                "final_mAP": aggregate["final_mAP"],
                "average_mAP_gain": aggregate["average_mAP_gain"],
                "final_mAP_gain": aggregate["final_mAP_gain"],
                "final_cF1": aggregate["final_cF1"],
                "final_oF1": aggregate["final_oF1"],
                "forgetting": aggregate["forgetting"][
                    "average_forgetting_old_classes"
                ],
            }
        )
        task_runs.append(data["tasks"])

    metric_names = (
        "average_mAP",
        "final_mAP",
        "average_mAP_gain",
        "final_mAP_gain",
        "final_cF1",
        "final_oF1",
        "forgetting",
    )
    aggregate = {
        metric: {
            "mean": float(np.mean([run[metric] for run in runs])),
            "std": float(np.std([run[metric] for run in runs])),
        }
        for metric in metric_names
    }
    per_task = []
    for task_id in range(8):
        task_rows = [rows[task_id] for rows in task_runs]
        per_task.append(
            {
                "mode": mode,
                "task": task_id,
                "seen_classes": task_rows[0]["seen_classes"],
                "seen_mAP_mean": float(
                    np.mean([row["test"]["mAP"] for row in task_rows])
                ),
                "seen_mAP_std": float(
                    np.std([row["test"]["mAP"] for row in task_rows])
                ),
                "current_mAP_mean": float(
                    np.mean([row["current_test"]["mAP"] for row in task_rows])
                ),
                "current_mAP_std": float(
                    np.std([row["current_test"]["mAP"] for row in task_rows])
                ),
                "ddp_mAP": task_rows[0]["baseline_ddp_test_mAP"],
                "gain_mean": float(
                    np.mean([row["test_mAP_gain"] for row in task_rows])
                ),
            }
        )
    classnames = list(task_runs[0][-1]["test"]["per_class_ap"])
    per_class = []
    for class_name in classnames:
        values = [
            rows[-1]["test"]["per_class_ap"][class_name] for rows in task_runs
        ]
        per_class.append(
            {
                "mode": mode,
                "class_name": class_name,
                "final_ap_mean": float(np.mean(values)),
                "final_ap_std": float(np.std(values)),
            }
        )
    return runs, aggregate, per_task, per_class


def load_references(output_root):
    output_root = Path(output_root)
    references = {}
    paths = {
        "base5_only_16shot": (
            output_root
            / "emotic_ddp_prompt_free_auxiliary_16shot_cls_feature_difference_summary"
            / "summary.json"
        ),
        "base5_only_full": (
            output_root
            / "emotic_ddp_prompt_free_auxiliary_cls_feature_difference_summary"
            / "summary.json"
        ),
        "ddp": (
            output_root
            / "emotic_ddp_cls_full_base5_ddp_baseline"
            / "evaluation_summary.json"
        ),
    }
    for name, path in paths.items():
        if not path.is_file():
            references[name] = {"status": "missing", "path": str(path)}
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if name == "ddp":
            aggregate = data["aggregate"]
            references[name] = {
                "status": "complete",
                "path": str(path),
                "average_mAP": {"mean": aggregate["average_mAP"], "std": 0.0},
                "final_mAP": {"mean": aggregate["final_mAP"], "std": 0.0},
                "final_cF1": {
                    "mean": data["tasks"][-1]["test"]["cF1"],
                    "std": 0.0,
                },
                "final_oF1": {
                    "mean": data["tasks"][-1]["test"]["oF1"],
                    "std": 0.0,
                },
                "forgetting": {
                    "mean": aggregate["forgetting"][
                        "average_forgetting_old_classes"
                    ],
                    "std": 0.0,
                },
            }
        else:
            references[name] = {
                "status": "complete",
                "path": str(path),
                **data["aggregate"],
            }
    return references


def write_html(path, summary):
    reference_rows = []
    for name, row in summary["references"].items():
        if row["status"] != "complete":
            reference_rows.append(
                f"<tr><td>{escape(name)}</td><td colspan='5'>missing</td></tr>"
            )
            continue
        reference_rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{row['average_mAP']['mean']:.4f}</td>"
            f"<td>{row['final_mAP']['mean']:.4f}</td>"
            f"<td>{row['final_cF1']['mean']:.4f}</td>"
            f"<td>{row['final_oF1']['mean']:.4f}</td>"
            f"<td>{row['forgetting']['mean']:.4f}</td>"
            "</tr>"
        )
    aggregate_rows = []
    for mode in ("16shot", "full"):
        metrics = summary["methods"][mode]["aggregate"]
        aggregate_rows.append(
            "<tr>"
            f"<td>{escape(mode)}</td>"
            f"<td>{metrics['average_mAP']['mean']:.4f} ± {metrics['average_mAP']['std']:.4f}</td>"
            f"<td>{metrics['final_mAP']['mean']:.4f} ± {metrics['final_mAP']['std']:.4f}</td>"
            f"<td>{metrics['final_mAP_gain']['mean']:+.4f}</td>"
            f"<td>{metrics['final_cF1']['mean']:.4f}</td>"
            f"<td>{metrics['final_oF1']['mean']:.4f}</td>"
            f"<td>{metrics['forgetting']['mean']:.4f}</td>"
            "</tr>"
        )
    task_rows = []
    for row in summary["per_task"]:
        task_rows.append(
            "<tr>"
            f"<td>{row['mode']}</td><td>{row['task']}</td>"
            f"<td>{row['seen_mAP_mean']:.4f} ± {row['seen_mAP_std']:.4f}</td>"
            f"<td>{row['current_mAP_mean']:.4f} ± {row['current_mAP_std']:.4f}</td>"
            f"<td>{row['ddp_mAP']:.4f}</td><td>{row['gain_mean']:+.4f}</td>"
            "</tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Task Adapter Bank Comparison</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-bottom:28px}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>EMOTIC B5-C3 Task-routed Adapter Bank</h1>"
        "<p>Prompted CLS · Feature Difference · fixed global α=0.03 · no gates</p>"
        "<h2>Frozen references</h2><table><tr><th>Method</th>"
        "<th>Average mAP</th><th>Final mAP</th><th>cF1</th><th>oF1</th>"
        "<th>Forgetting</th></tr>"
        + "".join(reference_rows)
        + "</table><h2>Task Adapter Banks</h2>"
        "<table><tr><th>Mode</th><th>Average mAP</th><th>Final mAP</th>"
        "<th>vs DDP</th><th>cF1</th><th>oF1</th><th>Forgetting</th></tr>"
        + "".join(aggregate_rows)
        + "</table><h2>Per-task results</h2>"
        "<table><tr><th>Mode</th><th>Task</th><th>Seen mAP</th>"
        "<th>Current mAP</th><th>DDP mAP</th><th>Gain</th></tr>"
        + "".join(task_rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    methods = {}
    all_runs = []
    all_task_rows = []
    all_class_rows = []
    for mode in ("16shot", "full"):
        runs, aggregate, per_task, per_class = collect_mode(
            args.output_root, mode, args.seeds
        )
        methods[mode] = {"runs": runs, "aggregate": aggregate}
        all_runs.extend(runs)
        all_task_rows.extend(per_task)
        all_class_rows.extend(per_class)
    summary = {
        "protocol": {
            "training_modes": ["16shot", "full"],
            "routing": "class introduction task",
            "formula": "feature_difference",
            "inference_alpha": 0.03,
            "seeds": args.seeds,
            "test_used_for_selection": False,
        },
        "references": load_references(args.output_root),
        "methods": methods,
        "per_task": all_task_rows,
        "per_class": all_class_rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(
        output_dir / "comparison_summary.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_runs[0]))
        writer.writeheader()
        writer.writerows(all_runs)
    reference_rows = []
    for name, row in summary["references"].items():
        if row["status"] != "complete":
            continue
        reference_rows.append(
            {
                "method": name,
                "average_mAP": row["average_mAP"]["mean"],
                "final_mAP": row["final_mAP"]["mean"],
                "final_cF1": row["final_cF1"]["mean"],
                "final_oF1": row["final_oF1"]["mean"],
                "forgetting": row["forgetting"]["mean"],
            }
        )
    if reference_rows:
        with open(
            output_dir / "reference_results.csv", "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
            writer.writeheader()
            writer.writerows(reference_rows)
    with open(
        output_dir / "per_task_results.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_task_rows[0]))
        writer.writeheader()
        writer.writerows(all_task_rows)
    with open(
        output_dir / "per_class_results.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_class_rows[0]))
        writer.writeheader()
        writer.writerows(all_class_rows)
    write_html(output_dir / "comparison_summary.html", summary)
    for mode in ("16shot", "full"):
        metrics = methods[mode]["aggregate"]
        print(
            f"{mode:7s}: final mAP "
            f"{metrics['final_mAP']['mean']:.4f}±{metrics['final_mAP']['std']:.4f}, "
            f"gain={metrics['final_mAP_gain']['mean']:+.4f}"
        )


if __name__ == "__main__":
    main()
