"""Compare the locked pooling-aware Task-0 pilot with C300 seed 0."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default=(
            "./output/emotic_ddp_final_token_c300_multiseed_seed0/"
            "training_summary.json"
        ),
    )
    parser.add_argument(
        "--candidate",
        default=(
            "./output/emotic_ddp_final_token_pooling_aware_task0_seed0/"
            "training_summary.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_final_token_pooling_aware_task0_seed0",
    )
    return parser.parse_args()


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8")), path


def build_summary(baseline, candidate):
    baseline_args = baseline["args"]
    candidate_args = candidate["args"]
    locked_equal = {
        key: baseline_args.get(key) == candidate_args.get(key)
        for key in (
            "task_id",
            "training_mode",
            "seed",
            "lr",
            "residual_scale",
            "identity_weight",
            "max_optimizer_steps",
        )
    }
    candidate_regularization = candidate.get("regularization", {})
    audit = {
        "locked_training_configuration_matches": all(locked_equal.values()),
        "candidate_health_passed": candidate.get(
            "training_health_check", {}
        ).get("status")
        == "passed",
        "pooling_weight_100": candidate_regularization.get("pooling_weight")
        == 100.0,
        "attention_weight_100": candidate_regularization.get(
            "attention_weight"
        )
        == 100.0,
        "margin_weight_1": candidate_regularization.get("margin_weight") == 1.0,
        "margin_beta_1": candidate_regularization.get("margin_beta") == 1.0,
        "test_absent": candidate.get("protocol", {}).get(
            "test_dataset_constructed"
        )
        is False
        and candidate.get("protocol", {}).get("test_labels_used") is False,
    }
    if not all(audit.values()):
        failed = [name for name, passed in audit.items() if not passed]
        raise ValueError(f"Pooling-aware pilot audit failed: {failed}")
    baseline_gain = float(baseline["reporting_val_gain"])
    candidate_gain = float(candidate["reporting_val_gain"])
    return {
        "experiment": "Task-0 Final-token pooling-aware regularization pilot",
        "training_change": {
            "classification_loss": "unchanged masked class-balanced BCE",
            "token_identity_weight": 0.1,
            "pooling_weight": 100.0,
            "attention_weight": 100.0,
            "margin_weight": 1.0,
            "margin_beta": 1.0,
        },
        "protocol_audit": audit,
        "baseline_c300_seed0": {
            "initial_val_mAP": baseline["initial_val_mAP"],
            "final_val_mAP": baseline["final_val_mAP"],
            "val_mAP_gain": baseline_gain,
            "final_validation_diagnostics": baseline[
                "final_validation_diagnostics"
            ],
        },
        "pooling_aware_candidate": {
            "initial_val_mAP": candidate["initial_val_mAP"],
            "final_val_mAP": candidate["final_val_mAP"],
            "val_mAP_gain": candidate_gain,
            "gain_relative_to_unregularized_c300": candidate_gain - baseline_gain,
            "final_validation_diagnostics": candidate[
                "final_validation_diagnostics"
            ],
            "history": candidate["history"],
        },
        "pilot_passes_positive_gain": candidate_gain > 0.0,
        "automatic_full_bank_launch": False,
    }


def write_html(path, summary):
    base = summary["baseline_c300_seed0"]
    candidate = summary["pooling_aware_candidate"]
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Pooling-aware Final-token Pilot</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse}th,td{border:1px solid #d9deea;"
        "padding:8px;text-align:right}th{background:#416fbd;color:white}"
        "pre{white-space:pre-wrap}</style>"
        "<h1>Task-0 Pooling-aware Final-token Pilot</h1>"
        "<table><tr><th>Method</th><th>Initial mAP</th><th>Final mAP</th>"
        "<th>Gain</th></tr>"
        f"<tr><td>C300</td><td>{base['initial_val_mAP']}</td>"
        f"<td>{base['final_val_mAP']}</td><td>{base['val_mAP_gain']}</td></tr>"
        f"<tr><td>Pooling-aware</td><td>{candidate['initial_val_mAP']}</td>"
        f"<td>{candidate['final_val_mAP']}</td>"
        f"<td>{candidate['val_mAP_gain']}</td></tr></table>"
        f"<h2>Full comparison</h2><pre>"
        f"{escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    baseline, baseline_path = _read(args.baseline)
    candidate, candidate_path = _read(args.candidate)
    summary = build_summary(baseline, candidate)
    summary["sources"] = {
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pooling_aware_pilot_summary.json"
    html_path = output_dir / "pooling_aware_pilot_summary.html"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(html_path, summary)
    candidate_row = summary["pooling_aware_candidate"]
    print(
        "Pooling-aware Task0: "
        f"{candidate_row['initial_val_mAP']:.4f} -> "
        f"{candidate_row['final_val_mAP']:.4f} "
        f"({candidate_row['val_mAP_gain']:+.4f})"
    )
    print(
        "Relative to unregularized C300: "
        f"{candidate_row['gain_relative_to_unregularized_c300']:+.4f}"
    )
    print(f"Positive-gain pilot passed: {summary['pilot_passes_positive_gain']}")
    print(f"Saved {json_path}")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()

