"""Summarize fixed-last ASL/BAL Task-routed Adapter Bank experiments."""

import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="./output")
    parser.add_argument(
        "--training_modes", nargs="+", choices=("full", "16shot"), default=("full",)
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        default=("weighted_bce", "asl", "bal_paper"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_task_adapter_bank_loss_comparison",
    )
    return parser.parse_args()


def evaluation_path(root, loss_name, mode, seed):
    return (
        Path(root)
        / (
            "emotic_ddp_task_adapter_bank_loss_"
            f"{loss_name}_{mode}_feature_difference_seed{seed}"
        )
        / "evaluation_summary.json"
    )


def mean_std(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def collect_group(root, loss_name, mode, seeds):
    summaries = []
    run_rows = []
    for seed in seeds:
        path = evaluation_path(root, loss_name, mode, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = data["protocol"]
        if protocol.get("classification_loss") != loss_name:
            raise ValueError(f"Loss metadata mismatch in {path}")
        if protocol.get("checkpoint_rule") != "last_epoch":
            raise ValueError(f"Non-fixed checkpoint in {path}")
        if protocol.get("decision_threshold") != 0.5:
            raise ValueError(f"Non-fixed decision threshold in {path}")
        summaries.append(data)
        aggregate = data["aggregate"]
        run_rows.append(
            {
                "loss": loss_name,
                "mode": mode,
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
        )

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
        metric: mean_std([row[metric] for row in run_rows])
        for metric in metric_names
    }
    per_task = []
    for task_id in range(8):
        rows = [summary["tasks"][task_id] for summary in summaries]
        per_task.append(
            {
                "loss": loss_name,
                "mode": mode,
                "task": task_id,
                "seen_classes": rows[0]["seen_classes"],
                "seen_mAP_mean": mean_std(
                    [row["test"]["mAP"] for row in rows]
                )["mean"],
                "seen_mAP_std": mean_std(
                    [row["test"]["mAP"] for row in rows]
                )["std"],
                "current_mAP_mean": mean_std(
                    [row["current_test"]["mAP"] for row in rows]
                )["mean"],
                "current_mAP_std": mean_std(
                    [row["current_test"]["mAP"] for row in rows]
                )["std"],
                "ddp_mAP": rows[0]["baseline_ddp_test_mAP"],
                "gain_mean": float(
                    np.mean([row["test_mAP_gain"] for row in rows])
                ),
            }
        )
    classnames = list(summaries[0]["tasks"][-1]["test"]["per_class_ap"])
    per_class = []
    for class_name in classnames:
        values = [
            summary["tasks"][-1]["test"]["per_class_ap"][class_name]
            for summary in summaries
        ]
        per_class.append(
            {
                "loss": loss_name,
                "mode": mode,
                "class_name": class_name,
                "final_ap_mean": float(np.mean(values)),
                "final_ap_std": float(np.std(values)),
            }
        )
    return summaries, run_rows, aggregate, per_task, per_class


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frequency_groups(output_root, loss_name, mode, seed):
    counts = {}
    bank_root = (
        Path(output_root)
        / f"emotic_ddp_task_adapter_bank_loss_{loss_name}_{mode}"
        / f"seed{seed}"
    )
    for task_id in range(8):
        path = bank_root / f"task{task_id}" / "class_distribution.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["classes"]:
            counts[row["class_name"]] = float(row["visible_positives"])
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    partitions = np.array_split(np.array(ordered, dtype=object), 3)
    return {
        group: {
            "classes": partition.tolist(),
            "positive_counts": {
                name: counts[name] for name in partition.tolist()
            },
        }
        for group, partition in zip(("head", "middle", "tail"), partitions)
    }


def bal_component_effects(run_rows, training_modes):
    """Return the locked 2x2 BAL component contrasts when all cells exist."""

    required = (
        "asl",
        "asl_smoothing",
        "asl_positive_weight",
        "bal_paper",
    )
    metrics = (
        "average_mAP",
        "final_mAP",
        "final_cF1",
        "final_oF1",
        "forgetting",
    )
    rows = []
    index = {
        (row["loss"], row["mode"], int(row["seed"])): row
        for row in run_rows
    }
    for mode in training_modes:
        seeds = sorted(
            {
                int(row["seed"])
                for row in run_rows
                if row["mode"] == mode and row["loss"] == "asl"
            }
        )
        if not seeds or not all(
            (loss, mode, seed) in index
            for loss in required
            for seed in seeds
        ):
            continue
        contrasts = {
            "label_smoothing_without_weight": (
                "asl_smoothing",
                "asl",
            ),
            "positive_weight_without_smoothing": (
                "asl_positive_weight",
                "asl",
            ),
            "label_smoothing_with_weight": (
                "bal_paper",
                "asl_positive_weight",
            ),
            "positive_weight_with_smoothing": (
                "bal_paper",
                "asl_smoothing",
            ),
        }
        for effect_name, (high, low) in contrasts.items():
            deltas = {
                metric: [
                    float(index[(high, mode, seed)][metric])
                    - float(index[(low, mode, seed)][metric])
                    for seed in seeds
                ]
                for metric in metrics
            }
            rows.append(
                {
                    "mode": mode,
                    "effect": effect_name,
                    "seeds": ",".join(str(seed) for seed in seeds),
                    **{
                        f"{metric}_{statistic}": float(function(values))
                        for metric in metrics
                        for statistic, function in (
                            ("mean", np.mean),
                            ("std", np.std),
                        )
                        for values in (deltas[metric],)
                    },
                }
            )
        interaction_deltas = {
            metric: [
                float(index[("bal_paper", mode, seed)][metric])
                - float(index[("asl_smoothing", mode, seed)][metric])
                - float(index[("asl_positive_weight", mode, seed)][metric])
                + float(index[("asl", mode, seed)][metric])
                for seed in seeds
            ]
            for metric in metrics
        }
        rows.append(
            {
                "mode": mode,
                "effect": "weight_smoothing_interaction",
                "seeds": ",".join(str(seed) for seed in seeds),
                **{
                    f"{metric}_{statistic}": float(function(values))
                    for metric in metrics
                    for statistic, function in (
                        ("mean", np.mean),
                        ("std", np.std),
                    )
                    for values in (interaction_deltas[metric],)
                },
            }
        )
    return rows


def write_html(path, summary):
    aggregate_rows = []
    for key, group in summary["groups"].items():
        metrics = group["aggregate"]
        aggregate_rows.append(
            "<tr>"
            f"<td>{escape(group['loss'])}</td><td>{escape(group['mode'])}</td>"
            f"<td>{metrics['average_mAP']['mean']:.4f} ± "
            f"{metrics['average_mAP']['std']:.4f}</td>"
            f"<td>{metrics['final_mAP']['mean']:.4f} ± "
            f"{metrics['final_mAP']['std']:.4f}</td>"
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
            f"<td>{escape(row['loss'])}</td><td>{escape(row['mode'])}</td>"
            f"<td>{row['task']}</td>"
            f"<td>{row['seen_mAP_mean']:.4f} ± {row['seen_mAP_std']:.4f}</td>"
            f"<td>{row['current_mAP_mean']:.4f} ± "
            f"{row['current_mAP_std']:.4f}</td>"
            f"<td>{row['ddp_mAP']:.4f}</td><td>{row['gain_mean']:+.4f}</td>"
            "</tr>"
        )
    component_rows = []
    for row in summary["bal_component_effects"]:
        component_rows.append(
            "<tr>"
            f"<td>{escape(row['mode'])}</td>"
            f"<td>{escape(row['effect'])}</td>"
            f"<td>{row['average_mAP_mean']:+.4f} ± "
            f"{row['average_mAP_std']:.4f}</td>"
            f"<td>{row['final_mAP_mean']:+.4f} ± "
            f"{row['final_mAP_std']:.4f}</td>"
            f"<td>{row['final_cF1_mean']:+.4f} ± "
            f"{row['final_cF1_std']:.4f}</td>"
            f"<td>{row['final_oF1_mean']:+.4f} ± "
            f"{row['final_oF1_std']:.4f}</td>"
            f"<td>{row['forgetting_mean']:+.4f} ± "
            f"{row['forgetting_std']:.4f}</td>"
            "</tr>"
        )
    component_html = ""
    if component_rows:
        component_html = (
            "<h2>BAL component contrasts</h2><table><tr>"
            "<th>Mode</th><th>Effect</th><th>Δ Average mAP</th>"
            "<th>Δ Final mAP</th><th>Δ cF1</th><th>Δ oF1</th>"
            "<th>Δ Forgetting</th></tr>"
            + "".join(component_rows)
            + "</table>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Task Adapter Bank Loss Comparison</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-bottom:28px}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>Task-routed Adapter Bank: BCE / ASL / BAL</h1>"
        "<p>Frozen DDP · prompted CLS · Feature Difference · fixed α=0.03 · "
        "fixed threshold=0.5 · last-epoch checkpoints</p>"
        "<table><tr><th>Loss</th><th>Mode</th><th>Average mAP</th>"
        "<th>Final mAP</th><th>vs DDP</th><th>cF1</th><th>oF1</th>"
        "<th>Forgetting</th></tr>"
        + "".join(aggregate_rows)
        + "</table><h2>Per-task</h2><table><tr><th>Loss</th>"
        "<th>Mode</th><th>Task</th><th>Seen mAP</th><th>Current mAP</th>"
        "<th>DDP mAP</th><th>Gain</th></tr>"
        + "".join(task_rows)
        + "</table>"
        + component_html,
        encoding="utf-8",
    )


def main():
    args = parse_args()
    groups = {}
    run_rows = []
    per_task = []
    per_class = []
    frequency_group_rows = []
    group_definitions = {}
    for mode in args.training_modes:
        for loss_name in args.losses:
            _, rows, aggregate, task_rows, class_rows = collect_group(
                args.output_root, loss_name, mode, args.seeds
            )
            key = f"{loss_name}:{mode}"
            groups[key] = {
                "loss": loss_name,
                "mode": mode,
                "aggregate": aggregate,
            }
            run_rows.extend(rows)
            per_task.extend(task_rows)
            per_class.extend(class_rows)
            definitions = frequency_groups(
                args.output_root, loss_name, mode, args.seeds[0]
            )
            group_definitions[f"{loss_name}:{mode}"] = definitions
            class_ap = {
                row["class_name"]: row["final_ap_mean"] for row in class_rows
            }
            for group_name, definition in definitions.items():
                values = [class_ap[name] for name in definition["classes"]]
                frequency_group_rows.append(
                    {
                        "loss": loss_name,
                        "mode": mode,
                        "frequency_group": group_name,
                        "class_count": len(values),
                        "final_mAP": float(np.mean(values)),
                    }
                )
    component_effect_rows = bal_component_effects(
        run_rows, args.training_modes
    )
    summary = {
        "protocol": {
            "losses": list(args.losses),
            "training_modes": list(args.training_modes),
            "seeds": list(args.seeds),
            "routing": "class_introduction_task",
            "feature_source": "prompted_cls",
            "correction_mode": "feature_difference",
            "inference_alpha": 0.03,
            "checkpoint_rule": "last_epoch",
            "decision_threshold": 0.5,
            "validation_used_for_checkpoint_selection": False,
            "validation_used_for_threshold_selection": False,
            "test_used_for_selection": False,
        },
        "groups": groups,
        "runs": run_rows,
        "per_task": per_task,
        "per_class": per_class,
        "frequency_group_definitions": group_definitions,
        "frequency_groups": frequency_group_rows,
        "bal_component_effects": component_effect_rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "comparison_summary.csv", run_rows)
    write_csv(output_dir / "per_task_results.csv", per_task)
    write_csv(output_dir / "per_class_results.csv", per_class)
    write_csv(output_dir / "frequency_group_results.csv", frequency_group_rows)
    write_csv(
        output_dir / "bal_component_effects.csv", component_effect_rows
    )
    write_html(output_dir / "comparison_summary.html", summary)
    for key, group in groups.items():
        metrics = group["aggregate"]
        print(
            f"{key:24s} final={metrics['final_mAP']['mean']:.4f}±"
            f"{metrics['final_mAP']['std']:.4f}, "
            f"gain={metrics['final_mAP_gain']['mean']:+.4f}"
        )


if __name__ == "__main__":
    main()
