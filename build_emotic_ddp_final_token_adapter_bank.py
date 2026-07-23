"""Validate eight Final-token Adapter checkpoints and write a bank manifest."""

import argparse
import json
from html import escape
from pathlib import Path

import torch

from emotic_final_token_adapter import build_final_token_bank_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an EMOTIC Final-token Adapter Bank manifest"
    )
    parser.add_argument("--bank_dir", required=True)
    parser.add_argument(
        "--training_mode", choices=("full", "16shot"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--residual_scale", type=float, default=0.03)
    parser.add_argument(
        "--selection",
        choices=(
            "fixed_last_epoch_no_validation_selection",
            "fixed_optimizer_steps_no_validation_selection",
        ),
        default="fixed_last_epoch_no_validation_selection",
    )
    parser.add_argument("--optimizer_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--identity_weight", type=float, default=0.1)
    parser.add_argument("--pooling_weight", type=float, default=0.0)
    parser.add_argument("--attention_weight", type=float, default=0.0)
    parser.add_argument(
        "--attention_kl_implementation",
        choices=("legacy_probability_clamp", "log_softmax_stable"),
        default="legacy_probability_clamp",
    )
    parser.add_argument("--margin_weight", type=float, default=0.0)
    parser.add_argument("--margin_beta", type=float, default=1.0)
    parser.add_argument(
        "--training_precision",
        choices=("cuda_amp", "float32"),
        default="cuda_amp",
    )
    return parser.parse_args()


def write_html(path, manifest):
    rows = []
    for task_id, entry in manifest["adapters"].items():
        rows.append(
            "<tr>"
            f"<td>{task_id}</td>"
            f"<td>{escape(str(entry['class_range']))}</td>"
            f"<td>{escape(entry['checkpoint'])}</td>"
            f"<td>{entry['final_epoch']}</td>"
            f"<td>{entry['reporting_val_mAP']}</td>"
            f"<td><code>{entry['sha256'][:12]}</code></td>"
            "</tr>"
        )
    if manifest["selection"] == "fixed_optimizer_steps_no_validation_selection":
        selection_text = (
            f"fixed {manifest['training_protocol']['optimizer_steps']} optimizer "
            "steps"
        )
    else:
        selection_text = "fixed last epoch"
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Final-token Adapter Bank</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:left}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>{escape(manifest['name'])}</h1>"
        f"<p>Mode: {manifest['training_mode']}; seed: {manifest['seed']}; "
        f"fixed α: {manifest['residual_scale']}; tokens: 197</p>"
        f"<p>Checkpoint selection: {escape(selection_text)}; validation is "
        "reporting-only.</p>"
        "<table><tr><th>Task</th><th>Classes</th><th>Checkpoint</th>"
        "<th>Final epoch</th><th>Reporting val mAP</th><th>SHA256</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    bank_dir = Path(args.bank_dir)
    task0 = torch.load(
        bank_dir / "task0" / "final_adapter.pth", map_location="cpu"
    )
    manifest = build_final_token_bank_manifest(
        bank_dir,
        args.training_mode,
        args.seed,
        task0["classnames"],
        residual_scale=args.residual_scale,
        selection=args.selection,
        training_protocol={
            "optimizer_steps": args.optimizer_steps,
            "learning_rate": args.learning_rate,
            "identity_weight": args.identity_weight,
            "pooling_weight": args.pooling_weight,
            "attention_weight": args.attention_weight,
            "attention_kl_implementation": args.attention_kl_implementation,
            "margin_weight": args.margin_weight,
            "margin_beta": args.margin_beta,
            "training_precision": args.training_precision,
        },
    )
    json_path = bank_dir / "final_token_adapter_bank_manifest.json"
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(bank_dir / "final_token_adapter_bank_manifest.html", manifest)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
