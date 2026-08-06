"""Validate eight Transformer Adapter checkpoints and build one bank manifest."""

import argparse
import json
from html import escape
from pathlib import Path

import torch

from emotic_transformer_adapter_bank import build_transformer_bank_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_dir", required=True)
    parser.add_argument(
        "--training_mode", choices=("full", "16shot"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def write_html(path, manifest):
    rows = []
    for task_id, entry in manifest["adapters"].items():
        rows.append(
            "<tr>"
            f"<td>{task_id}</td>"
            f"<td>{escape(str(entry['class_range']))}</td>"
            f"<td>{escape(entry['checkpoint'])}</td>"
            f"<td>{entry['epoch']}</td>"
            f"<td>{entry['reporting_val_mAP']:.4f}</td>"
            f"<td><code>{entry['sha256'][:12]}</code></td>"
            "</tr>"
        )
    architecture = manifest["architecture"]
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Transformer Adapter Bank</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:left}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>{escape(manifest['name'])}</h1>"
        f"<p>Mode: {manifest['training_mode']}; seed: {manifest['seed']}; "
        f"loss: {manifest['classification_loss']}; checkpoint: fixed last; "
        f"layers: {architecture['layer_numbers']}; bottleneck: "
        f"{architecture['hidden_dim']}→{architecture['bottleneck_dim']}→"
        f"{architecture['hidden_dim']}.</p>"
        "<table><tr><th>Task</th><th>Classes</th><th>Checkpoint</th>"
        "<th>Epoch</th><th>Reporting val mAP</th><th>SHA256</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    bank_dir = Path(args.bank_dir)
    task0_path = bank_dir / "task0" / "last_transformer_adapter.pth"
    if not task0_path.is_file():
        raise FileNotFoundError(task0_path)
    task0 = torch.load(task0_path, map_location="cpu")
    manifest = build_transformer_bank_manifest(
        bank_dir=bank_dir,
        training_mode=args.training_mode,
        seed=args.seed,
        classnames=task0["classnames"],
    )
    json_path = bank_dir / "transformer_adapter_bank_manifest.json"
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(bank_dir / "transformer_adapter_bank_manifest.html", manifest)
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
