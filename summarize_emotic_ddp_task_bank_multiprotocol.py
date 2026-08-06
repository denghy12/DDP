#!/usr/bin/env python3
"""Aggregate B10-C4/B4-C2 Task Bank runs against frozen DDP baselines."""

from __future__ import annotations

import argparse
import json
import statistics
from html import escape
from pathlib import Path
from typing import Dict, Mapping, Sequence


METRICS = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")
PROTOCOLS = {
    "b10c4": ("emotic_b10c4_v0.1", 5),
    "b4c2": ("emotic_b4c2_v0.1", 12),
}


def _load(path: Path) -> Mapping:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _aggregate(values: Sequence[float]) -> Dict[str, float]:
    if len(values) != 3:
        raise ValueError("Formal aggregation requires seeds 0, 1, and 2")
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
    }


def _summary_metrics(payload: Mapping) -> Dict[str, float]:
    main = payload.get("main_table")
    if not isinstance(main, Mapping):
        raise ValueError("Benchmark summary lacks main_table")
    return {metric: float(main[metric]) for metric in METRICS}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--b10c4-baseline-root", type=Path, required=True)
    parser.add_argument("--b4c2-baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _format(metric: Mapping[str, float]) -> str:
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def main() -> None:
    args = _parse_args()
    rows = []
    for short_name, (protocol_id, task_count) in PROTOCOLS.items():
        baseline_root = getattr(args, f"{short_name}_baseline_root")
        per_seed = []
        for seed in range(3):
            adapted_path = (
                args.run_root
                / "benchmarks"
                / protocol_id
                / "A"
                / "DDP-Task-Adapter-Bank-Full"
                / f"seed{seed}"
                / "metrics"
                / "summary.json"
            )
            baseline_path = (
                baseline_root
                / "benchmarks"
                / protocol_id
                / "A"
                / "Original-DDP-Tau2"
                / f"seed{seed}"
                / "metrics"
                / "summary.json"
            )
            adapted_payload = _load(adapted_path)
            baseline_payload = _load(baseline_path)
            adapted = _summary_metrics(adapted_payload)
            baseline = _summary_metrics(baseline_payload)
            adapted_tasks = adapted_payload.get("summary", {}).get("task_metrics", [])
            if len(adapted_tasks) != task_count:
                raise ValueError(
                    f"{short_name} seed{seed} expected {task_count} task rows"
                )
            per_seed.append(
                {
                    "seed": seed,
                    "ddp": baseline,
                    "task_bank": adapted,
                    "gain": {
                        metric: adapted[metric] - baseline[metric]
                        for metric in METRICS
                    },
                }
            )
        rows.append(
            {
                "protocol": short_name.upper().replace("C", "-C"),
                "protocol_id": protocol_id,
                "task_count": task_count,
                "per_seed": per_seed,
                "ddp": {
                    metric: _aggregate([row["ddp"][metric] for row in per_seed])
                    for metric in METRICS
                },
                "task_bank": {
                    metric: _aggregate(
                        [row["task_bank"][metric] for row in per_seed]
                    )
                    for metric in METRICS
                },
                "gain": {
                    metric: _aggregate([row["gain"][metric] for row in per_seed])
                    for metric in METRICS
                },
            }
        )
    payload = {
        "schema_version": 1,
        "method": "DDP-Task-Adapter-Bank-Full",
        "reporting_split": "test",
        "threshold": 0.5,
        "seeds": [0, 1, 2],
        "test_used_for_selection": False,
        "protocols": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "multiprotocol_summary.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_rows = []
    for row in rows:
        for method_key, method_name in (
            ("ddp", "Original-DDP-Tau2"),
            ("task_bank", "DDP Task Bank Full"),
        ):
            values = row[method_key]
            gain = "—" if method_key == "ddp" else _format(row["gain"]["final_mAP"])
            html_rows.append(
                "<tr>"
                f"<td>{escape(row['protocol'])}</td>"
                f"<td>{escape(method_name)}</td>"
                f"<td>{_format(values['final_mAP'])}</td>"
                f"<td>{gain}</td>"
                f"<td>{_format(values['average_mAP'])}</td>"
                f"<td>{_format(values['final_cF1'])}</td>"
                f"<td>{_format(values['final_oF1'])}</td>"
                f"<td>{_format(values['forgetting'])}</td>"
                "</tr>"
            )
    (args.output_dir / "multiprotocol_summary.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Task Bank Multi-protocol Summary</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>EMOTIC DDP Task Adapter Bank: B10-C4 / B4-C2</h1>"
        "<p>Held-out test, mean ± sample standard deviation over seeds 0–2; "
        "fixed threshold 0.5; test is never used for selection.</p>"
        "<table><tr><th>Protocol</th><th>Method</th><th>Final mAP</th>"
        "<th>Δ Final mAP</th><th>Average mAP</th><th>Final cF1</th>"
        "<th>Final oF1</th><th>Forgetting</th></tr>"
        + "".join(html_rows)
        + "</table>",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
