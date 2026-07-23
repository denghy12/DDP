"""Summarize Full and 16-shot Final-token Adapter Bank experiments."""

import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize EMOTIC Final-token Adapter Banks"
    )
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_final_token_adapter_bank_comparison",
    )
    return parser.parse_args()


def evaluation_path(output_root, mode, seed):
    return (
        Path(output_root)
        / f"emotic_ddp_final_token_adapter_bank_{mode}_seed{seed}"
        / "evaluation_summary.json"
    )


def collect_mode(output_root, mode, seeds):
    runs = []
    task_sets = []
    for seed in seeds:
        path = evaluation_path(output_root, mode, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = data["protocol"]
        if protocol["decision_threshold"] != 0.5:
            raise ValueError(f"Unlocked threshold in {path}")
        if protocol["checkpoint_selection"] != "fixed last epoch":
            raise ValueError(f"Unlocked checkpoint selection in {path}")
        aggregate = data["aggregate"]
        runs.append(
            {
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
        task_sets.append(data["tasks"])
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
            "mean": float(np.mean([row[metric] for row in runs])),
            "std": float(np.std([row[metric] for row in runs])),
        }
        for metric in metric_names
    }
    per_task = []
    for task_id in range(8):
        rows = [tasks[task_id] for tasks in task_sets]
        per_task.append(
            {
                "mode": mode,
                "task": task_id,
                "seen_classes": rows[0]["seen_classes"],
                "ddp_mAP": rows[0]["baseline_ddp_test_mAP"],
                "seen_mAP_mean": float(
                    np.mean([row["test"]["mAP"] for row in rows])
                ),
                "seen_mAP_std": float(
                    np.std([row["test"]["mAP"] for row in rows])
                ),
                "current_mAP_mean": float(
                    np.mean([row["current_test"]["mAP"] for row in rows])
                ),
                "current_mAP_std": float(
                    np.std([row["current_test"]["mAP"] for row in rows])
                ),
                "gain_mean": float(np.mean([row["test_mAP_gain"] for row in rows])),
            }
        )
    return runs, aggregate, per_task


def load_reference(output_root):
    path = Path(output_root) / "emotic_ddp_task_adapter_bank_comparison" / "comparison_summary.json"
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"status": "complete", "path": str(path), "methods": data.get("methods")}


def load_ddp_reference(output_root):
    path = evaluation_path(output_root, "full", 0)
    data = json.loads(path.read_text(encoding="utf-8"))
    aggregate = data["aggregate"]
    return {
        "average_mAP": aggregate["baseline_average_mAP"],
        "final_mAP": aggregate["baseline_final_mAP"],
        "final_cF1": aggregate["baseline_final_cF1"],
        "final_oF1": aggregate["baseline_final_oF1"],
    }


def write_html(path, summary):
    ddp = summary["ddp_reference"]
    aggregate_rows = [
        "<tr><td>DDP baseline</td>"
        f"<td>{ddp['average_mAP']:.4f}</td>"
        f"<td>{ddp['final_mAP']:.4f}</td><td>—</td>"
        f"<td>{ddp['final_cF1']:.4f}</td>"
        f"<td>{ddp['final_oF1']:.4f}</td><td>—</td></tr>"
    ]
    for mode in ("full", "16shot"):
        metrics = summary["methods"][mode]["aggregate"]
        aggregate_rows.append(
            "<tr>"
            f"<td>{mode}</td>"
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
        "<title>EMOTIC Final-token Adapter Bank Comparison</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-bottom:28px}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>EMOTIC B5-C3 Final-token Adapter Bank</h1>"
        "<p>197 tokens · original DDP pooling · fixed α=0.03 · threshold=0.5 · last epoch</p>"
        "<table><tr><th>Mode</th><th>Average mAP</th><th>Final mAP</th>"
        "<th>vs DDP</th><th>cF1</th><th>oF1</th><th>Forgetting</th></tr>"
        + "".join(aggregate_rows)
        + "</table><h2>Per-task</h2><table><tr><th>Mode</th><th>Task</th>"
        "<th>Seen mAP</th><th>Current mAP</th><th>DDP mAP</th><th>Gain</th></tr>"
        + "".join(task_rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    methods = {}
    all_runs = []
    all_tasks = []
    for mode in ("full", "16shot"):
        runs, aggregate, per_task = collect_mode(
            args.output_root, mode, args.seeds
        )
        methods[mode] = {"runs": runs, "aggregate": aggregate}
        all_runs.extend(runs)
        all_tasks.extend(per_task)
    summary = {
        "protocol": {
            "method": "task-routed Final-token Adapter Bank",
            "token_count": 197,
            "pooling": "exact original DDP token attention",
            "residual_scale": 0.03,
            "decision_threshold": 0.5,
            "checkpoint_selection": "fixed last epoch",
            "validation_role": "reporting only",
            "test_used_for_selection": False,
        },
        "methods": methods,
        "per_task": all_tasks,
        "ddp_reference": load_ddp_reference(args.output_root),
        "previous_cls_task_bank_reference": load_reference(args.output_root),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(output_dir / "runs.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_runs[0]))
        writer.writeheader()
        writer.writerows(all_runs)
    with open(output_dir / "per_task.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_tasks[0]))
        writer.writeheader()
        writer.writerows(all_tasks)
    write_html(output_dir / "comparison_summary.html", summary)
    print(json.dumps({mode: methods[mode]["aggregate"] for mode in methods}, indent=2))
    print(f"Saved {output_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
