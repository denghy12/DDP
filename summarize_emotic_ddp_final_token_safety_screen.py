"""Summarize the locked Task-0 Final-token training-safety screen.

The report deliberately performs no automatic winner selection.  Validation
is a global-configuration diagnostic only; test data and per-class routing are
not part of this screen.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


CONFIGS = (
    {
        "config": "A",
        "run_name": "emotic_ddp_final_token_safety_A_lr1e4_steps100_seed0",
        "learning_rate": 1e-4,
        "max_optimizer_steps": 100,
    },
    {
        "config": "B",
        "run_name": "emotic_ddp_final_token_safety_B_lr1e4_steps300_seed0",
        "learning_rate": 1e-4,
        "max_optimizer_steps": 300,
    },
    {
        "config": "C",
        "run_name": "emotic_ddp_final_token_safety_C_lr3e5_steps300_seed0",
        "learning_rate": 3e-5,
        "max_optimizer_steps": 300,
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="./output")
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_final_token_safety_screen_summary",
    )
    return parser.parse_args()


def _validate_run(data, expected):
    args = data.get("args", {})
    protocol = data.get("protocol", {})
    health = data.get("training_health_check", {})
    checks = health.get("checks", {})
    conditions = {
        "health_passed": health.get("status") == "passed",
        "all_health_subchecks_passed": bool(checks) and all(checks.values()),
        "task0": int(args.get("task_id", -1)) == 0,
        "full_training_data": args.get("training_mode") == "full",
        "seed0": int(args.get("seed", -1)) == 0,
        "learning_rate_matches": abs(
            float(args.get("lr", -1)) - expected["learning_rate"]
        )
        < 1e-12,
        "optimizer_steps_match": int(data.get("optimizer_steps", -1))
        == expected["max_optimizer_steps"],
        "alpha_locked": float(args.get("residual_scale", -1)) == 0.03,
        "identity_weight_locked": float(args.get("identity_weight", -1)) == 0.1,
        "test_not_constructed": protocol.get("test_dataset_constructed") is False,
        "test_not_used": protocol.get("test_labels_used") is False,
        "validation_reporting_only": protocol.get("validation_role")
        == "post-training reporting only",
    }
    if not all(conditions.values()):
        failed = [name for name, passed in conditions.items() if not passed]
        raise ValueError(
            f"Safety config {expected['config']} failed protocol audit: {failed}"
        )
    return conditions


def build_summary(output_root: Path):
    rows = []
    initial_maps = []
    for expected in CONFIGS:
        path = output_root / expected["run_name"] / "training_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        audit = _validate_run(data, expected)
        history = data.get("history", [])
        if not history:
            raise ValueError(f"Safety config {expected['config']} has no history")
        final_history = history[-1]
        initial_map = float(data["initial_val_mAP"])
        final_map = float(data["final_val_mAP"])
        initial_maps.append(initial_map)
        initial_per_class = data["initial_per_class_ap"]
        final_per_class = data["final_per_class_ap"]
        rows.append(
            {
                **expected,
                "status": "completed_and_healthy",
                "initial_val_mAP": initial_map,
                "final_val_mAP": final_map,
                "reporting_val_gain": final_map - initial_map,
                "training_loss": float(final_history["loss"]),
                "classification_loss": float(
                    final_history["classification_loss"]
                ),
                "identity_loss": float(final_history["identity_loss"]),
                "up_weight_delta_norm": float(
                    final_history["up_weight_delta_norm"]
                ),
                "down_weight_delta_norm": float(
                    final_history["down_weight_delta_norm"]
                ),
                "last_raw_residual_rms": float(
                    final_history["last_raw_residual_rms"]
                ),
                "last_adapted_feature_delta_rms": float(
                    final_history["last_adapted_feature_delta_rms"]
                ),
                "per_class_ap_change": {
                    name: float(final_per_class[name] - value)
                    for name, value in initial_per_class.items()
                },
                "protocol_audit": audit,
                "source": str(path.resolve()),
            }
        )
    if max(initial_maps) - min(initial_maps) > 1e-6:
        raise ValueError("Safety configs do not share the same initial baseline")
    return {
        "experiment": "EMOTIC Task-0 Final-token Adapter safety screen",
        "purpose": (
            "Measure safe global training strength after repairing the "
            "zero-initialization dead-gradient bug"
        ),
        "protocol": {
            "task_id": 0,
            "training_mode": "full",
            "seed": 0,
            "residual_scale": 0.03,
            "identity_weight": 0.1,
            "test_used": False,
            "class_specific_selection": False,
            "automatic_winner_selection": False,
            "validation_role": "global configuration diagnostic",
        },
        "shared_initial_val_mAP": initial_maps[0],
        "runs": rows,
        "selection": None,
    }


def write_html(path: Path, summary):
    columns = (
        "config",
        "learning_rate",
        "max_optimizer_steps",
        "status",
        "initial_val_mAP",
        "final_val_mAP",
        "reporting_val_gain",
        "training_loss",
        "identity_loss",
        "up_weight_delta_norm",
        "down_weight_delta_norm",
        "last_raw_residual_rms",
        "last_adapted_feature_delta_rms",
    )
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns)
        + "</tr>"
        for row in summary["runs"]
    )
    class_rows = "".join(
        f"<tr><td>{escape(row['config'])}</td><td><pre>"
        f"{escape(json.dumps(row['per_class_ap_change'], indent=2))}"
        f"</pre></td></tr>"
        for row in summary["runs"]
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Final-token Adapter Safety Screen</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #d9deea;padding:7px;text-align:right}"
        "th{background:#416fbd;color:white}pre{text-align:left}</style>"
        "<h1>EMOTIC Final-token Adapter Safety Screen</h1>"
        "<p>No automatic winner selection; no test data; no per-class routing.</p>"
        f"<pre>{escape(json.dumps(summary['protocol'], indent=2))}</pre>"
        f"<table><tr>{header}</tr>{body}</table>"
        f"<h2>Per-class validation AP changes</h2><table>{class_rows}</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(Path(args.output_root))
    json_path = output_dir / "safety_screen_summary.json"
    html_path = output_dir / "safety_screen_summary.html"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(html_path, summary)
    for row in summary["runs"]:
        print(
            f"Config {row['config']}: lr={row['learning_rate']}, "
            f"steps={row['max_optimizer_steps']}, "
            f"val mAP {row['initial_val_mAP']:.4f} -> "
            f"{row['final_val_mAP']:.4f} "
            f"({row['reporting_val_gain']:+.4f})"
        )
    print(f"Saved {json_path}")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()

