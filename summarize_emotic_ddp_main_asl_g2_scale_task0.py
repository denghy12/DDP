"""Summarize the three-seed Task-0 ASL-2 loss-scale experiment."""

import argparse
import csv
import json
import statistics
from html import escape
from pathlib import Path

from summarize_emotic_ddp_main_asl_task0_ablation import (
    eligibility,
    mean_std,
    resolve_run_root,
)


METHODS = {
    "two_way_bce": {
        "objective": "two_way_bce",
        "gamma_neg": None,
        "loss_w": 0.03,
    },
    "asl_g2": {
        "objective": "asl",
        "gamma_neg": 2.0,
        "loss_w": 0.03,
    },
    "asl_g2_lw009": {
        "objective": "asl",
        "gamma_neg": 2.0,
        "loss_w": 0.09,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_main_asl_g2_scale_task0_summary",
    )
    return parser.parse_args()


def validate_protocol(protocol, method, seed, source):
    specification = METHODS[method]
    checks = {
        "objective": protocol.get("ddp_main_classification_loss")
        == specification["objective"],
        "seed": int(protocol.get("seed", -1)) == seed,
        "checkpoint": protocol.get("checkpoint_rule") == "fixed_last_epoch",
        "validation": protocol.get("training_time_eval_splits") == ["val"],
        "no_test": protocol.get("test_used_during_training") is False,
        "epochs": int(protocol.get("epochs_per_task", -1)) == 30,
        "loss_w": float(protocol.get("loss_w", -1))
        == specification["loss_w"],
        "reduction": protocol.get("reduction") == "sum",
        "labels": protocol.get("supervision") == "current_task_classes_only",
    }
    if specification["gamma_neg"] is not None:
        asl = protocol.get("asl", {})
        checks.update(
            {
                "gamma_neg": float(asl.get("gamma_neg", -1))
                == specification["gamma_neg"],
                "gamma_pos": float(asl.get("gamma_pos", -1)) == 0.0,
                "clip": float(asl.get("clip", -1)) == 0.05,
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Protocol mismatch {failed} in {source}")


def load_run(output_root, method, seed):
    root = resolve_run_root(output_root, method, seed)
    diagnostics_path = root / "training_diagnostics.json"
    detail_path = root / "detail" / f"{root.name}_per_class_task_table.json"
    checkpoint_path = root / "checkpoints" / "task0.pth"
    for path in (diagnostics_path, detail_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    validate_protocol(diagnostics["protocol"], method, seed, diagnostics_path)
    history = [
        row for row in diagnostics["epoch_history"] if int(row["task"]) == 0
    ]
    if len(history) != 30:
        raise RuntimeError(f"Expected 30 Task-0 epochs in {diagnostics_path}")
    overall = detail["overall_rows"][-1]
    first = diagnostics["first_batch_health"][0]
    classes = []
    for row in detail["table_rows"][:5]:
        classes.append(
            {
                "class_name": row["class_name"],
                "ap": 100.0 * float(row["task0_ap"]),
                "precision": 100.0 * float(row["task0_precision"]),
                "recall": 100.0 * float(row["task0_recall"]),
                "f1": 100.0 * float(row["task0_f1"]),
            }
        )
    return {
        "method": method,
        "seed": seed,
        "val_mAP": float(overall["mAP"]),
        "val_cF1": float(overall["cF1"]),
        "val_oF1": float(overall["oF1"]),
        "val_loss": float(overall["loss"]),
        "initial_raw_loss_per_label": float(first["raw_loss_per_label"]),
        "initial_prompt_gradient_norm": float(first["prompt_gradient_norm"]),
        "epoch29_raw_loss_per_label": float(history[-1]["raw_loss_per_label"]),
        "classes": classes,
        "root": str(root),
        "checkpoint": str(checkpoint_path),
    }


def aggregate(runs):
    metrics = (
        "val_mAP",
        "val_cF1",
        "val_oF1",
        "val_loss",
        "initial_raw_loss_per_label",
        "initial_prompt_gradient_norm",
        "epoch29_raw_loss_per_label",
    )
    return {metric: mean_std([run[metric] for run in runs]) for metric in metrics}


def paired_delta(candidate_runs, reference_runs):
    result = {}
    for candidate, reference in zip(candidate_runs, reference_runs):
        if candidate["seed"] != reference["seed"]:
            raise RuntimeError("Paired comparison seed mismatch")
    for metric in ("val_mAP", "val_cF1", "val_oF1"):
        values = [
            candidate[metric] - reference[metric]
            for candidate, reference in zip(candidate_runs, reference_runs)
        ]
        result[metric] = {
            **mean_std(values),
            "per_seed": {
                str(candidate["seed"]): float(value)
                for candidate, value in zip(candidate_runs, values)
            },
        }
    return result


def aggregate_classes(runs):
    rows = []
    for class_index, reference in enumerate(runs[0]["classes"]):
        row = {"class_name": reference["class_name"]}
        for metric in ("ap", "precision", "recall", "f1"):
            row[metric] = mean_std(
                [run["classes"][class_index][metric] for run in runs]
            )
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        method: [load_run(args.output_root, method, seed) for seed in args.seeds]
        for method in METHODS
    }
    aggregates = {method: aggregate(rows) for method, rows in runs.items()}
    deltas = {
        "scaled_minus_bce": paired_delta(
            runs["asl_g2_lw009"], runs["two_way_bce"]
        ),
        "scaled_minus_unscaled_asl_g2": paired_delta(
            runs["asl_g2_lw009"], runs["asl_g2"]
        ),
    }
    gate = eligibility(deltas["scaled_minus_bce"])
    class_results = {
        method: aggregate_classes(rows) for method, rows in runs.items()
    }
    summary = {
        "protocol": {
            "task": 0,
            "seeds": list(args.seeds),
            "epochs": 30,
            "checkpoint_rule": "fixed_last_epoch",
            "training_time_eval_splits": ["val"],
            "test_used": False,
            "threshold": 0.5,
            "candidate": {
                "loss": "asl",
                "gamma_neg": 2.0,
                "gamma_pos": 0.0,
                "clip": 0.05,
                "loss_w": 0.09,
                "scale_origin": "rounded training-gradient ratio; no val/test metric",
            },
            "full_run_gate": {
                "mean_val_mAP_delta_min": -0.1,
                "mean_val_cF1_delta_min": -2.0,
                "mean_val_oF1_delta_min": -2.0,
            },
        },
        "methods": aggregates,
        "paired_deltas": deltas,
        "eligibility_vs_bce": gate,
        "recommended_for_full_tasks": gate["passes"],
        "per_class": class_results,
        "runs": runs,
    }
    (output_dir / "scale_matched_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_rows = []
    for method, specification in METHODS.items():
        stats = aggregates[method]
        csv_rows.append(
            {
                "method": method,
                "gamma_neg": specification["gamma_neg"],
                "loss_w": specification["loss_w"],
                "val_mAP_mean": stats["val_mAP"]["mean"],
                "val_mAP_std": stats["val_mAP"]["std"],
                "val_cF1_mean": stats["val_cF1"]["mean"],
                "val_cF1_std": stats["val_cF1"]["std"],
                "val_oF1_mean": stats["val_oF1"]["mean"],
                "val_oF1_std": stats["val_oF1"]["std"],
                "initial_gradient_mean": stats[
                    "initial_prompt_gradient_norm"
                ]["mean"],
            }
        )
    write_csv(output_dir / "scale_matched_summary.csv", csv_rows)

    html_rows = []
    for row in csv_rows:
        html_rows.append(
            "<tr>"
            f"<td>{escape(row['method'])}</td>"
            f"<td>{escape(str(row['gamma_neg']))}</td>"
            f"<td>{row['loss_w']:.2f}</td>"
            f"<td>{row['val_mAP_mean']:.4f} ± {row['val_mAP_std']:.4f}</td>"
            f"<td>{row['val_cF1_mean']:.4f} ± {row['val_cF1_std']:.4f}</td>"
            f"<td>{row['val_oF1_mean']:.4f} ± {row['val_oF1_std']:.4f}</td>"
            f"<td>{row['initial_gradient_mean']:.4f}</td>"
            "</tr>"
        )
    decision = "PASS" if gate["passes"] else "STOP"
    html = (
        "<!doctype html><meta charset='utf-8'><title>ASL-2 scale match</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:9px;text-align:right}"
        "th{background:#416fbd;color:#fff}</style>"
        "<h1>DDP main ASL-2: Task-0 loss-scale alignment</h1>"
        "<p>Three seeds · fixed-last · val only · threshold 0.50 · no test. "
        f"Pre-registered full-run decision: <b>{decision}</b>.</p>"
        "<table><tr><th>Method</th><th>gamma_neg</th><th>loss_w</th>"
        "<th>Val mAP</th><th>cF1</th><th>oF1</th>"
        "<th>Initial gradient norm</th></tr>"
        + "".join(html_rows)
        + "</table>"
    )
    (output_dir / "scale_matched_summary.html").write_text(
        html, encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"Saved {output_dir}")


if __name__ == "__main__":
    main()
