"""Summarize the pre-registered Task-0 C300 three-seed decision gate."""

from __future__ import annotations

import argparse
import json
import statistics
from html import escape
from pathlib import Path


SEEDS = (0, 1, 2)
RUN_TEMPLATE = "emotic_ddp_final_token_c300_multiseed_seed{seed}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="./output")
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_final_token_c300_multiseed_summary",
    )
    return parser.parse_args()


def _audit(data, seed):
    args = data.get("args", {})
    protocol = data.get("protocol", {})
    health = data.get("training_health_check", {})
    checks = health.get("checks", {})
    conditions = {
        "healthy": health.get("status") == "passed"
        and bool(checks)
        and all(checks.values()),
        "task0_full": int(args.get("task_id", -1)) == 0
        and args.get("training_mode") == "full",
        "seed_matches": int(args.get("seed", -1)) == seed,
        "lr_locked": abs(float(args.get("lr", -1)) - 3e-5) < 1e-12,
        "steps_locked": int(data.get("optimizer_steps", -1)) == 300,
        "alpha_locked": float(args.get("residual_scale", -1)) == 0.03,
        "identity_weight_locked": float(args.get("identity_weight", -1)) == 0.1,
        "test_absent": protocol.get("test_dataset_constructed") is False
        and protocol.get("test_labels_used") is False,
        "validation_reporting_only": protocol.get("validation_role")
        == "post-training reporting only",
        "drift_diagnostics_present": isinstance(
            data.get("final_validation_diagnostics"), dict
        ),
    }
    if not all(conditions.values()):
        failed = [name for name, passed in conditions.items() if not passed]
        raise ValueError(f"Seed {seed} failed locked C300 audit: {failed}")
    return conditions


def _mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def build_summary(output_root: Path):
    runs = []
    initial_maps = []
    class_changes = {}
    for seed in SEEDS:
        path = output_root / RUN_TEMPLATE.format(seed=seed) / "training_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        audit = _audit(data, seed)
        initial_map = float(data["initial_val_mAP"])
        final_map = float(data["final_val_mAP"])
        initial_maps.append(initial_map)
        per_class_change = {
            name: float(data["final_per_class_ap"][name] - initial)
            for name, initial in data["initial_per_class_ap"].items()
        }
        for name, change in per_class_change.items():
            class_changes.setdefault(name, []).append(change)
        history = data["history"][-1]
        runs.append(
            {
                "seed": seed,
                "initial_val_mAP": initial_map,
                "final_val_mAP": final_map,
                "val_mAP_gain": final_map - initial_map,
                "per_class_ap_change": per_class_change,
                "training_loss": float(history["loss"]),
                "identity_loss": float(history["identity_loss"]),
                "up_weight_delta_norm": float(history["up_weight_delta_norm"]),
                "down_weight_delta_norm": float(history["down_weight_delta_norm"]),
                "initial_validation_diagnostics": data[
                    "initial_validation_diagnostics"
                ],
                "final_validation_diagnostics": data[
                    "final_validation_diagnostics"
                ],
                "protocol_audit": audit,
                "source": str(path.resolve()),
            }
        )
    if max(initial_maps) - min(initial_maps) > 1e-6:
        raise ValueError("C300 seeds do not share the same initial DDP baseline")

    final_maps = [run["final_val_mAP"] for run in runs]
    gains = [run["val_mAP_gain"] for run in runs]
    mean_gain = statistics.mean(gains)
    passes = mean_gain > 0.0
    diagnostic_fields = {
        "attention_kl_mean": [
            run["final_validation_diagnostics"]["attention_kl"]["mean"]
            for run in runs
        ],
        "pooled_cosine_drift_mean": [
            run["final_validation_diagnostics"]["pooled_feature_cosine_drift"][
                "mean"
            ]
            for run in runs
        ],
        "path_logit_absolute_drift_mean": [
            run["final_validation_diagnostics"]["path_logit_absolute_drift"][
                "mean"
            ]
            for run in runs
        ],
        "cls_token_delta_l2_mean": [
            run["final_validation_diagnostics"]["cls_token_delta_l2"]["mean"]
            for run in runs
        ],
        "patch_token_delta_l2_mean": [
            run["final_validation_diagnostics"]["patch_token_delta_l2"]["mean"]
            for run in runs
        ],
        "all_token_delta_l2_p95": [
            run["final_validation_diagnostics"]["all_token_delta_l2"]["p95"]
            for run in runs
        ],
        "all_token_delta_l2_max": [
            run["final_validation_diagnostics"]["all_token_delta_l2"]["max"]
            for run in runs
        ],
    }
    return {
        "experiment": "EMOTIC Task-0 Final-token Adapter C300 three-seed gate",
        "locked_configuration": {
            "learning_rate": 3e-5,
            "optimizer_steps": 300,
            "training_mode": "full",
            "task_id": 0,
            "seeds": list(SEEDS),
            "residual_scale": 0.03,
            "identity_weight": 0.1,
            "test_used": False,
            "training_loss_changed": False,
            "diagnostics_enter_training_loss": False,
        },
        "decision_rule": (
            "Proceed to Task0-7 Bank only if the three-seed mean Task0 "
            "validation mAP gain is strictly greater than zero"
        ),
        "shared_initial_val_mAP": initial_maps[0],
        "aggregate": {
            "final_val_mAP": _mean_std(final_maps),
            "val_mAP_gain": _mean_std(gains),
            "per_class_ap_change": {
                name: _mean_std(values) for name, values in class_changes.items()
            },
            "diagnostics": {
                name: _mean_std(values)
                for name, values in diagnostic_fields.items()
            },
            "passes_task0_mean_gain_gate": passes,
            "decision": (
                "eligible_for_explicitly_authorized_full_bank_run"
                if passes
                else "stop_final_token_bank_and_design_pooling_aware_regularization"
            ),
        },
        "runs": runs,
        "automatic_full_bank_launch": False,
    }


def write_html(path: Path, summary):
    columns = (
        "seed",
        "initial_val_mAP",
        "final_val_mAP",
        "val_mAP_gain",
        "training_loss",
        "identity_loss",
        "up_weight_delta_norm",
        "down_weight_delta_norm",
    )
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(run[column]))}</td>" for column in columns)
        + "</tr>"
        for run in summary["runs"]
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Final-token C300 Three-seed Gate</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:7px;text-align:right}"
        "th{background:#416fbd;color:white}pre{white-space:pre-wrap}</style>"
        "<h1>Final-token C300 Task-0 Three-seed Gate</h1>"
        f"<p><b>Decision:</b> {escape(summary['aggregate']['decision'])}</p>"
        f"<p><b>Pass:</b> "
        f"{escape(str(summary['aggregate']['passes_task0_mean_gain_gate']))}</p>"
        f"<table><tr>{header}</tr>{rows}</table>"
        "<h2>Aggregate and drift diagnostics</h2>"
        f"<pre>{escape(json.dumps(summary['aggregate'], indent=2))}</pre>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(Path(args.output_root))
    json_path = output_dir / "c300_multiseed_summary.json"
    html_path = output_dir / "c300_multiseed_summary.html"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(html_path, summary)
    for run in summary["runs"]:
        print(
            f"Seed {run['seed']}: val mAP {run['initial_val_mAP']:.4f} -> "
            f"{run['final_val_mAP']:.4f} ({run['val_mAP_gain']:+.4f})"
        )
    aggregate = summary["aggregate"]
    print(
        f"Mean gain: {aggregate['val_mAP_gain']['mean']:+.4f} ± "
        f"{aggregate['val_mAP_gain']['sample_std']:.4f}"
    )
    print(f"Gate passed: {aggregate['passes_task0_mean_gain_gate']}")
    print(f"Decision: {aggregate['decision']}")
    print(f"Saved {json_path}")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()

