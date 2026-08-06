"""Summarize three-seed Transformer-block Adapter Bank experiments."""

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
        "--training_modes", nargs="+", choices=("full", "16shot"),
        default=("full", "16shot"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_transformer_adapter_bank_comparison",
    )
    return parser.parse_args()


def evaluation_path(root, mode, seed):
    return (
        Path(root)
        / f"emotic_ddp_transformer_adapter_bank_asl_{mode}_seed{seed}_evaluation"
        / "evaluation_summary.json"
    )


def stats(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path, summary):
    rows = []
    for row in summary["groups"]:
        rows.append(
            "<tr>"
            f"<td>{escape(row['training_mode'])}</td>"
            f"<td>{row['average_mAP']['mean']:.4f} ± {row['average_mAP']['std']:.4f}</td>"
            f"<td>{row['final_mAP']['mean']:.4f} ± {row['final_mAP']['std']:.4f}</td>"
            f"<td>{row['final_mAP_gain']['mean']:+.4f}</td>"
            f"<td>{row['final_cF1']['mean']:.4f}</td>"
            f"<td>{row['final_oF1']['mean']:.4f}</td>"
            f"<td>{row['forgetting']['mean']:.4f}</td>"
            "</tr>"
        )
    reference = summary.get("reference_cls_feature_difference_asl")
    reference_html = ""
    if reference is not None:
        reference_html = (
            "<h2>Frozen reference</h2>"
            f"<p>Previous CLS Feature-Difference Task Bank (ASL, full): "
            f"final mAP {reference['final_mAP']['mean']:.4f} ± "
            f"{reference['final_mAP']['std']:.4f}; DDP baseline "
            f"{reference['baseline_final_mAP']:.4f}.</p>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Transformer Adapter Bank Comparison</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>EMOTIC Transformer-block Task Adapter Bank</h1>"
        f"<pre>{escape(json.dumps(summary['locked_protocol'], indent=2, ensure_ascii=False))}</pre>"
        + reference_html
        + "<table><tr><th>Training data</th><th>Average mAP</th>"
        "<th>Final mAP</th><th>vs DDP</th><th>Final cF1</th>"
        "<th>Final oF1</th><th>Forgetting</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    runs = []
    task_rows = []
    for mode in args.training_modes:
        summaries = []
        for seed in args.seeds:
            path = evaluation_path(args.output_root, mode, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            protocol = data["protocol"]
            if protocol["adapter_loss"] != "asl":
                raise ValueError(f"Unexpected Adapter loss in {path}")
            if protocol["checkpoint_rule"] != "fixed_last_epoch":
                raise ValueError(f"Checkpoint selection leaked into {path}")
            if protocol["decision_threshold"] != 0.5:
                raise ValueError(f"Threshold is not locked in {path}")
            summaries.append(data)
            aggregate = data["aggregate"]
            runs.append(
                {
                    "training_mode": mode,
                    "seed": seed,
                    "average_mAP": aggregate["average_mAP"],
                    "final_mAP": aggregate["final_mAP"],
                    "final_mAP_gain": aggregate["final_mAP_gain"],
                    "final_cF1": aggregate["final_cF1"],
                    "final_oF1": aggregate["final_oF1"],
                    "forgetting": aggregate["forgetting"][
                        "average_forgetting_old_classes"
                    ],
                }
            )
        mode_runs = [row for row in runs if row["training_mode"] == mode]
        metrics = (
            "average_mAP",
            "final_mAP",
            "final_mAP_gain",
            "final_cF1",
            "final_oF1",
            "forgetting",
        )
        groups.append(
            {
                "training_mode": mode,
                "seeds": list(args.seeds),
                **{
                    metric: stats([row[metric] for row in mode_runs])
                    for metric in metrics
                },
                "baseline_final_mAP": summaries[0]["aggregate"][
                    "baseline_final_mAP"
                ],
            }
        )
        for task_id in range(8):
            rows = [summary["tasks"][task_id] for summary in summaries]
            task_rows.append(
                {
                    "training_mode": mode,
                    "task": task_id,
                    "seen_classes": rows[0]["seen_classes"],
                    "test_mAP_mean": stats([row["test"]["mAP"] for row in rows])[
                        "mean"
                    ],
                    "test_mAP_std": stats([row["test"]["mAP"] for row in rows])[
                        "std"
                    ],
                    "gain_mean": float(np.mean([row["test_mAP_gain"] for row in rows])),
                    "current_mAP_mean": float(
                        np.mean([row["current_test"]["mAP"] for row in rows])
                    ),
                }
            )

    reference_rows = []
    for seed in args.seeds:
        path = (
            Path(args.output_root)
            / (
                "emotic_ddp_task_adapter_bank_loss_asl_full_"
                f"feature_difference_seed{seed}"
            )
            / "evaluation_summary.json"
        )
        if not path.is_file():
            reference_rows = []
            break
        reference_rows.append(
            json.loads(path.read_text(encoding="utf-8"))["aggregate"]
        )
    reference = None
    if reference_rows:
        reference = {
            "method": "CLS Feature-Difference Task Bank",
            "training_mode": "full",
            "adapter_loss": "asl",
            "checkpoint_rule": "fixed_last_epoch",
            "seeds": list(args.seeds),
            "average_mAP": stats([row["average_mAP"] for row in reference_rows]),
            "final_mAP": stats([row["final_mAP"] for row in reference_rows]),
            "final_cF1": stats([row["final_cF1"] for row in reference_rows]),
            "final_oF1": stats([row["final_oF1"] for row in reference_rows]),
            "forgetting": stats(
                [
                    row["forgetting"]["average_forgetting_old_classes"]
                    for row in reference_rows
                ]
            ),
            "baseline_final_mAP": reference_rows[0]["baseline_final_mAP"],
        }

    summary = {
        "locked_protocol": {
            "ddp_main_loss": "two_way_bce_frozen_checkpoint",
            "adapter_loss": "asl(gamma_neg=9.8,gamma_pos=0,clip=0.05)",
            "adapter": "parallel ViT MLP, layers 4-12, 768-128-768, ReLU",
            "routing": "class introduction task",
            "optimizer": "Adam(lr=4e-4), cosine, 20 epochs",
            "checkpoint": "fixed last epoch",
            "threshold": 0.5,
            "selection": "no test/validation model selection",
        },
        "groups": groups,
        "reference_cls_feature_difference_asl": reference,
        "runs": runs,
        "per_task": task_rows,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "run_metrics.csv", runs)
    write_csv(output_dir / "per_task_metrics.csv", task_rows)
    write_html(output_dir / "comparison_summary.html", summary)
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
