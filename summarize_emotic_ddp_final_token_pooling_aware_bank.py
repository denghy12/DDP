"""Compare the complete pooling-aware Final-token Bank with prior baselines."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np


METRICS = (
    "average_mAP",
    "final_mAP",
    "average_mAP_gain",
    "final_mAP_gain",
    "final_cF1",
    "final_oF1",
    "forgetting",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize pooling-aware Final-token Adapter Bank"
    )
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default=(
            "./output/"
            "emotic_ddp_final_token_pooling_aware_bank_comparison"
        ),
    )
    return parser.parse_args()


def _evaluation_path(output_root: Path, method: str, seed: int) -> Path:
    if method == "previous_full":
        name = f"emotic_ddp_final_token_adapter_bank_full_seed{seed}"
    elif method == "pooling_aware_full":
        name = f"emotic_ddp_final_token_pooling_aware_bank_full_seed{seed}"
    else:
        raise ValueError(method)
    return output_root / name / "evaluation_summary.json"


def collect_method(output_root: Path, method: str, seeds):
    runs = []
    task_sets = []
    for seed in seeds:
        path = _evaluation_path(output_root, method, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = data["protocol"]
        if float(protocol["decision_threshold"]) != 0.5:
            raise ValueError(f"Unlocked threshold in {path}")
        if protocol.get("test_used_for_selection") is not False:
            raise ValueError(f"Test-selected result in {path}")
        if method == "pooling_aware_full":
            reg = protocol.get("pooling_aware_regularization", {})
            expected = {
                "optimizer_steps": 300,
                "pooling_weight": 100.0,
                "attention_weight": 100.0,
                "margin_weight": 1.0,
                "margin_beta": 1.0,
            }
            for key, value in expected.items():
                if float(reg.get(key, -1)) != float(value):
                    raise ValueError(f"Unlocked {key} in {path}")
            if reg.get("attention_kl_implementation") != "log_softmax_stable":
                raise ValueError(f"Unstable attention KL in {path}")
            if reg.get("training_precision") != "float32":
                raise ValueError(f"Non-FP32 Adapter training in {path}")
            if protocol.get("checkpoint_selection") != "fixed 300 optimizer steps":
                raise ValueError(f"Unexpected checkpoint rule in {path}")
        aggregate = data["aggregate"]
        run = {
            "method": method,
            "seed": int(seed),
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
        runs.append(run)
        task_sets.append(data["tasks"])
    aggregate = {
        metric: {
            "mean": float(np.mean([run[metric] for run in runs])),
            "std": float(np.std([run[metric] for run in runs])),
        }
        for metric in METRICS
    }
    per_task = []
    for task_id in range(8):
        rows = [task_set[task_id] for task_set in task_sets]
        per_task.append(
            {
                "method": method,
                "task": task_id,
                "seen_classes": rows[0]["seen_classes"],
                "ddp_mAP": rows[0]["baseline_ddp_test_mAP"],
                "mAP_mean": float(np.mean([row["test"]["mAP"] for row in rows])),
                "mAP_std": float(np.std([row["test"]["mAP"] for row in rows])),
                "gain_mean": float(np.mean([row["test_mAP_gain"] for row in rows])),
                "current_mAP_mean": float(
                    np.mean([row["current_test"]["mAP"] for row in rows])
                ),
            }
        )
    return runs, aggregate, per_task


def collect_training_diagnostics(output_root: Path, seeds):
    rows = []
    for seed in seeds:
        for task in range(8):
            path = (
                output_root
                / "emotic_ddp_final_token_pooling_aware_bank_full"
                / f"seed{seed}"
                / f"task{task}"
                / "training_summary.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            drift = data["final_validation_diagnostics"]
            rows.append(
                {
                    "seed": int(seed),
                    "task": task,
                    "val_mAP_gain": data["reporting_val_gain"],
                    "attention_kl": drift["attention_kl"]["mean"],
                    "pooled_cosine_drift": drift[
                        "pooled_feature_cosine_drift"
                    ]["mean"],
                    "path_logit_absolute_drift": drift[
                        "path_logit_absolute_drift"
                    ]["mean"],
                    "all_token_delta_l2": drift["all_token_delta_l2"]["mean"],
                }
            )
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in (
            "val_mAP_gain",
            "attention_kl",
            "pooled_cosine_drift",
            "path_logit_absolute_drift",
            "all_token_delta_l2",
        )
    }
    return rows, aggregate


def _fmt(metric):
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def write_html(path: Path, summary: dict):
    ddp = summary["ddp_reference"]
    methods = summary["methods"]
    rows = [
        "<tr><td>DDP baseline</td>"
        f"<td>{ddp['average_mAP']:.4f}</td>"
        f"<td>{ddp['final_mAP']:.4f}</td><td>—</td>"
        f"<td>{ddp['final_cF1']:.4f}</td>"
        f"<td>{ddp['final_oF1']:.4f}</td><td>—</td></tr>"
    ]
    labels = {
        "previous_full": "Previous Final-token Full",
        "pooling_aware_full": "Pooling + attention + margin Full",
    }
    for method in ("previous_full", "pooling_aware_full"):
        agg = methods[method]["aggregate"]
        rows.append(
            "<tr>"
            f"<td>{escape(labels[method])}</td>"
            f"<td>{_fmt(agg['average_mAP'])}</td>"
            f"<td>{_fmt(agg['final_mAP'])}</td>"
            f"<td>{agg['final_mAP_gain']['mean']:+.4f}</td>"
            f"<td>{_fmt(agg['final_cF1'])}</td>"
            f"<td>{_fmt(agg['final_oF1'])}</td>"
            f"<td>{_fmt(agg['forgetting'])}</td>"
            "</tr>"
        )
    task_rows = []
    previous = {row["task"]: row for row in summary["per_task"] if row["method"] == "previous_full"}
    candidate = {row["task"]: row for row in summary["per_task"] if row["method"] == "pooling_aware_full"}
    for task in range(8):
        old = previous[task]
        new = candidate[task]
        task_rows.append(
            "<tr>"
            f"<td>{task}</td><td>{new['seen_classes']}</td>"
            f"<td>{new['ddp_mAP']:.4f}</td>"
            f"<td>{old['mAP_mean']:.4f} ± {old['mAP_std']:.4f}</td>"
            f"<td>{new['mAP_mean']:.4f} ± {new['mAP_std']:.4f}</td>"
            f"<td>{new['gain_mean']:+.4f}</td>"
            f"<td>{new['mAP_mean'] - old['mAP_mean']:+.4f}</td>"
            "</tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Pooling-aware Final-token Full Bank</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-bottom:28px}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}td:first-child{text-align:left}"
        ".note{background:#f4f6fb;padding:12px;border-left:4px solid #416fbd}"
        "</style><h1>EMOTIC B5-C3 Pooling-aware Final-token Bank</h1>"
        "<p class='note'>Full training data · 3 seeds · Task 0–7 · fixed 300 "
        "optimizer steps · LR=3e-5 · α=0.03 · λpool=100 · λattention=100 · "
        "λmargin=1 · fixed threshold=0.5. Validation/test do not select "
        "weights, gates, α, or thresholds.</p>"
        "<table><tr><th>Method</th><th>Average mAP</th><th>Final mAP</th>"
        "<th>vs DDP</th><th>Final cF1</th><th>Final oF1</th>"
        "<th>Forgetting</th></tr>" + "".join(rows) + "</table>"
        "<h2>Per-task comparison</h2><table><tr><th>Task</th><th>Seen</th>"
        "<th>DDP mAP</th><th>Previous Full</th><th>Pooling-aware Full</th>"
        "<th>vs DDP</th><th>vs Previous</th></tr>" + "".join(task_rows) +
        "</table><h2>Training diagnostics</h2><pre>" +
        escape(json.dumps(summary["training_diagnostics"]["aggregate"], indent=2)) +
        "</pre>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    methods = {}
    all_runs = []
    all_tasks = []
    for method in ("previous_full", "pooling_aware_full"):
        runs, aggregate, tasks = collect_method(output_root, method, args.seeds)
        methods[method] = {"runs": runs, "aggregate": aggregate}
        all_runs.extend(runs)
        all_tasks.extend(tasks)
    diagnostics, diagnostic_aggregate = collect_training_diagnostics(
        output_root, args.seeds
    )
    candidate_first = json.loads(
        _evaluation_path(output_root, "pooling_aware_full", args.seeds[0]).read_text(
            encoding="utf-8"
        )
    )
    aggregate = candidate_first["aggregate"]
    summary = {
        "protocol": {
            "experiment": "complete pooling-aware Final-token Adapter Bank",
            "training_mode": "full",
            "tasks": list(range(8)),
            "seeds": list(args.seeds),
            "optimizer_steps": 300,
            "learning_rate": 3e-5,
            "residual_scale": 0.03,
            "identity_weight": 0.1,
            "pooling_weight": 100.0,
            "attention_weight": 100.0,
            "attention_kl_implementation": "log_softmax_stable",
            "training_precision": "float32",
            "margin_weight": 1.0,
            "margin_beta": 1.0,
            "decision_threshold": 0.5,
            "validation_role": "reporting only",
            "test_used_for_selection": False,
        },
        "ddp_reference": {
            "average_mAP": aggregate["baseline_average_mAP"],
            "final_mAP": aggregate["baseline_final_mAP"],
            "final_cF1": aggregate["baseline_final_cF1"],
            "final_oF1": aggregate["baseline_final_oF1"],
        },
        "methods": methods,
        "per_task": all_tasks,
        "training_diagnostics": {
            "runs": diagnostics,
            "aggregate": diagnostic_aggregate,
        },
        "comparison_note": (
            "Previous Full used its original 50-epoch/LR=1e-3 protocol; the "
            "candidate uses the locked C300 pooling-aware protocol. This is a "
            "complete method-level comparison, not an isolated regularizer ablation."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_runs[0]))
        writer.writeheader()
        writer.writerows(all_runs)
    with (output_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_tasks[0]))
        writer.writeheader()
        writer.writerows(all_tasks)
    with (output_dir / "training_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    write_html(output_dir / "comparison_summary.html", summary)
    print(json.dumps({key: methods[key]["aggregate"] for key in methods}, indent=2))
    print(f"Saved {output_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
