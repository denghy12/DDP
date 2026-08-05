"""Summarize the locked three-seed Task-0 mild-ASL ablation."""

import argparse
import csv
import json
import statistics
from html import escape
from pathlib import Path


METHODS = {
    "two_way_bce": {"objective": "two_way_bce", "gamma_neg": None},
    "asl_g9p8": {"objective": "asl", "gamma_neg": 9.8},
    "asl_g4": {"objective": "asl", "gamma_neg": 4.0},
    "asl_g2": {"objective": "asl", "gamma_neg": 2.0},
}
MILD_METHODS = ("asl_g4", "asl_g2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_main_asl_task0_ablation_summary",
    )
    return parser.parse_args()


def mean_std(values):
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }


def run_name(method, seed):
    return f"emotic_ddp_main_task0_{method}_seed{seed}_v1"


def legacy_seed0_name(method, seed):
    if seed != 0:
        return None
    suffix = {
        "two_way_bce": "two_way_bce",
        "asl_g9p8": "asl",
    }.get(method)
    if suffix is None:
        return None
    return f"emotic_ddp_main_healthcheck_{suffix}_task0_seed0_v2"


def resolve_run_root(output_root, method, seed):
    output_root = Path(output_root)
    primary = output_root / run_name(method, seed)
    if primary.is_dir():
        return primary
    legacy = legacy_seed0_name(method, seed)
    if legacy is not None and (output_root / legacy).is_dir():
        return output_root / legacy
    return primary


def validate_protocol(protocol, method, seed, source):
    specification = METHODS[method]
    actual = protocol.get("ddp_main_classification_loss")
    if actual != specification["objective"]:
        raise RuntimeError(
            f"Objective mismatch in {source}: {actual} != "
            f"{specification['objective']}"
        )
    if int(protocol.get("seed", -1)) != seed:
        raise RuntimeError(f"Seed mismatch in {source}")
    if protocol.get("checkpoint_rule") != "fixed_last_epoch":
        raise RuntimeError(f"Checkpoint rule mismatch in {source}")
    if protocol.get("training_time_eval_splits") != ["val"]:
        raise RuntimeError(f"Training split mismatch in {source}")
    if protocol.get("test_used_during_training") is not False:
        raise RuntimeError(f"Test leakage audit failed in {source}")
    if int(protocol.get("epochs_per_task", -1)) != 30:
        raise RuntimeError(f"Epoch protocol mismatch in {source}")
    if float(protocol.get("loss_w", -1)) != 0.03:
        raise RuntimeError(f"Loss scale mismatch in {source}")
    if protocol.get("reduction") != "sum":
        raise RuntimeError(f"Reduction mismatch in {source}")
    if protocol.get("supervision") != "current_task_classes_only":
        raise RuntimeError(f"Visible-label protocol mismatch in {source}")
    if specification["gamma_neg"] is not None:
        asl = protocol.get("asl", {})
        if float(asl.get("gamma_neg", -1)) != specification["gamma_neg"]:
            raise RuntimeError(f"gamma_neg mismatch in {source}")
        if float(asl.get("gamma_pos", -1)) != 0.0:
            raise RuntimeError(f"gamma_pos mismatch in {source}")
        if float(asl.get("clip", -1)) != 0.05:
            raise RuntimeError(f"ASL clip mismatch in {source}")


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
    protocol = diagnostics["protocol"]
    validate_protocol(protocol, method, seed, diagnostics_path)
    history = [
        row for row in diagnostics["epoch_history"] if int(row["task"]) == 0
    ]
    if len(history) != 30:
        raise RuntimeError(f"Expected 30 Task-0 epochs in {diagnostics_path}")
    overall = detail["overall_rows"][-1]
    first = diagnostics["first_batch_health"][0]
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


def paired_delta(candidate_runs, baseline_runs):
    metrics = ("val_mAP", "val_cF1", "val_oF1")
    result = {}
    for metric in metrics:
        values = [
            candidate[metric] - baseline[metric]
            for candidate, baseline in zip(candidate_runs, baseline_runs)
        ]
        result[metric] = {
            **mean_std(values),
            "per_seed": {
                str(candidate["seed"]): float(value)
                for candidate, value in zip(candidate_runs, values)
            },
        }
    return result


def eligibility(delta):
    checks = {
        "mAP_not_below_bce_by_more_than_0.1": delta["val_mAP"]["mean"] >= -0.1,
        "cF1_not_below_bce_by_more_than_2": delta["val_cF1"]["mean"] >= -2.0,
        "oF1_not_below_bce_by_more_than_2": delta["val_oF1"]["mean"] >= -2.0,
    }
    return {"passes": all(checks.values()), "checks": checks}


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
        method: paired_delta(runs[method], runs["two_way_bce"])
        for method in METHODS
        if method != "two_way_bce"
    }
    gates = {method: eligibility(deltas[method]) for method in MILD_METHODS}
    eligible = [method for method in MILD_METHODS if gates[method]["passes"]]
    summary = {
        "protocol": {
            "task": 0,
            "seeds": list(args.seeds),
            "epochs": 30,
            "checkpoint_rule": "fixed_last_epoch",
            "validation_role": "reporting_and_pre_registered_go_no_go",
            "training_time_eval_splits": ["val"],
            "test_used": False,
            "threshold": 0.5,
            "loss_w": 0.03,
            "asl_common": {"gamma_pos": 0.0, "clip": 0.05, "eps": 1e-8},
            "pre_registered_priority": list(MILD_METHODS),
            "full_run_gate": {
                "mean_val_mAP_delta_min": -0.1,
                "mean_val_cF1_delta_min": -2.0,
                "mean_val_oF1_delta_min": -2.0,
            },
        },
        "methods": aggregates,
        "paired_minus_bce": deltas,
        "eligibility": gates,
        "eligible_in_pre_registered_order": eligible,
        "recommended_full_candidate": eligible[0] if eligible else None,
        "runs": runs,
    }
    (output_dir / "task0_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_rows = []
    for method in METHODS:
        stats = aggregates[method]
        delta = deltas.get(method)
        csv_rows.append(
            {
                "method": method,
                "gamma_neg": METHODS[method]["gamma_neg"],
                "val_mAP_mean": stats["val_mAP"]["mean"],
                "val_mAP_std": stats["val_mAP"]["std"],
                "mAP_delta_vs_bce": 0.0 if delta is None else delta["val_mAP"]["mean"],
                "val_cF1_mean": stats["val_cF1"]["mean"],
                "cF1_delta_vs_bce": 0.0 if delta is None else delta["val_cF1"]["mean"],
                "val_oF1_mean": stats["val_oF1"]["mean"],
                "oF1_delta_vs_bce": 0.0 if delta is None else delta["val_oF1"]["mean"],
                "passes_full_gate": gates.get(method, {}).get("passes"),
            }
        )
    write_csv(output_dir / "task0_ablation_summary.csv", csv_rows)

    html_rows = []
    for row in csv_rows:
        gate = "reference" if row["passes_full_gate"] is None else (
            "PASS" if row["passes_full_gate"] else "STOP"
        )
        html_rows.append(
            "<tr>"
            f"<td>{escape(row['method'])}</td>"
            f"<td>{escape(str(row['gamma_neg']))}</td>"
            f"<td>{row['val_mAP_mean']:.4f} ± {row['val_mAP_std']:.4f}</td>"
            f"<td>{row['mAP_delta_vs_bce']:+.4f}</td>"
            f"<td>{row['val_cF1_mean']:.4f}</td>"
            f"<td>{row['cF1_delta_vs_bce']:+.4f}</td>"
            f"<td>{row['val_oF1_mean']:.4f}</td>"
            f"<td>{row['oF1_delta_vs_bce']:+.4f}</td>"
            f"<td>{gate}</td>"
            "</tr>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>Task-0 mild ASL</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:9px;text-align:right}"
        "th{background:#416fbd;color:#fff}</style>"
        "<h1>DDP main loss: pre-registered Task-0 mild-ASL ablation</h1>"
        "<p>Three seeds · 30 fixed epochs · val only · threshold 0.50 · "
        "no test access. Deltas are paired by seed against two-way BCE.</p>"
        "<table><tr><th>Method</th><th>gamma_neg</th><th>Val mAP</th>"
        "<th>ΔmAP</th><th>cF1</th><th>ΔcF1</th><th>oF1</th>"
        "<th>ΔoF1</th><th>Full gate</th></tr>"
        + "".join(html_rows)
        + "</table>"
    )
    (output_dir / "task0_ablation_summary.html").write_text(
        html, encoding="utf-8"
    )
    print(json.dumps(summary["eligibility"], ensure_ascii=False, indent=2))
    print(f"Saved {output_dir}")


if __name__ == "__main__":
    main()
