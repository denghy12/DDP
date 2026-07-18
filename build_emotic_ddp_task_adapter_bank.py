"""Validate eight task Adapter checkpoints and write a portable bank manifest."""

import argparse
import json
from html import escape
from pathlib import Path

import torch

from emotic_task_adapter_bank import build_bank_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build an EMOTIC Adapter Bank manifest")
    parser.add_argument("--bank_dir", required=True)
    parser.add_argument("--training_mode", choices=("full", "16shot"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--inference_alpha", type=float, default=0.03)
    return parser.parse_args()


def write_html(path, manifest):
    rows = []
    for task_id, entry in manifest["adapters"].items():
        rows.append(
            "<tr>"
            f"<td>{task_id}</td>"
            f"<td>{escape(str(entry['class_range']))}</td>"
            f"<td>{escape(entry['checkpoint'])}</td>"
            f"<td>{entry['best_epoch']}</td>"
            f"<td>{entry['selection_score']}</td>"
            f"<td><code>{entry['sha256'][:12]}</code></td>"
            "</tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Task Adapter Bank</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:left}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>{escape(manifest['name'])}</h1>"
        f"<p>Mode: {manifest['training_mode']}; seed: {manifest['seed']}; "
        f"fixed α: {manifest['inference_alpha']}</p>"
        "<table><tr><th>Task</th><th>Classes</th><th>Checkpoint</th>"
        "<th>Epoch</th><th>Val mAP</th><th>SHA256</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    bank_dir = Path(args.bank_dir)
    task0 = torch.load(bank_dir / "task0" / "best_adapter.pth", map_location="cpu")
    manifest = build_bank_manifest(
        bank_dir,
        args.training_mode,
        args.seed,
        task0["classnames"],
        inference_alpha=args.inference_alpha,
    )
    json_path = bank_dir / "adapter_bank_manifest.json"
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(bank_dir / "adapter_bank_manifest.html", manifest)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

