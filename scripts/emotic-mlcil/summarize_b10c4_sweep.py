#!/usr/bin/env python3
"""Validate and summarize all 36 locked EMOTIC B10-C4 result bundles."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


METHODS = (
    ("finetune", "Sequential Fine-Tuning", "0"),
    ("lwf", "LwF", "0"),
    ("ewc", "EWC", "0"),
    ("agcn", "AGCN", "0"),
    ("csc", "CSC", "0"),
    ("multi_lane", "MULTI-LANE", "0"),
    ("l3a", "L3A", "0"),
    ("original_ddp", "Original-DDP-Tau2", "0"),
    ("er", "ER", "20/class"),
    ("prs", "PRS", "20/class"),
    ("derpp", "DER++", "20/class"),
    ("krt", "KRT", "20/class"),
)
SEEDS = (0, 1, 2)
METRICS = ("final_mAP", "final_cF1", "final_oF1", "average_mAP", "forgetting")


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _finite(value: object, role: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{role} must be finite")
    return number


def _aggregate(values: Sequence[float]) -> Dict[str, float]:
    if len(values) != 3:
        raise ValueError("Three-seed aggregation requires exactly three values")
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def _format(value: Mapping[str, float]) -> str:
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def _write_markdown(path: Path, methods: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# EMOTIC B10-C4 Track-A 12-Baseline Summary",
        "",
        "All values are held-out-test mean ± sample standard deviation over seeds 0--2.",
        "",
        "| Method | Memory | Last mAP | Last cF1 | Last oF1 | Avg. mAP | Forgetting |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        aggregate = method["aggregate"]
        lines.append(
            "| {name} | {memory} | {final_map} | {cf1} | {of1} | {avg_map} | {forgetting} |".format(
                name=method["display_name"],
                memory=method["memory_display"],
                final_map=_format(aggregate["final_mAP"]),
                cf1=_format(aggregate["final_cF1"]),
                of1=_format(aggregate["final_oF1"]),
                avg_map=_format(aggregate["average_mAP"]),
                forgetting=_format(aggregate["forgetting"]),
            )
        )
    lines.extend(
        [
            "",
            "Memory is replay image-sample capacity. F1 uses the fixed global threshold 0.5.",
            "All method hyperparameters were transferred from the frozen B5-C3 runs without B10-C4 test tuning.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.resolve()
    methods: List[Dict[str, Any]] = []
    class_order_hash: Optional[str] = None
    source_tree_hash: Optional[str] = None
    protocol_hashes: Dict[str, str] = {}
    total_bundles = 0

    for method_key, display_name, memory_display in METHODS:
        per_seed: List[Dict[str, Any]] = []
        task_curves: List[List[float]] = []
        for seed in SEEDS:
            root = (
                run_root
                / "benchmarks"
                / "emotic_b10c4_v0.1"
                / "A"
                / display_name
                / f"seed{seed}"
            )
            required = (
                root / "config_resolved.json",
                root / "run_manifest.json",
                root / "metrics" / "summary.json",
                root / "metrics" / "task_metrics.json",
                root / "results_to_sync" / args.run_id / "sync_manifest.json",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete {display_name} seed{seed}: " + ", ".join(missing)
                )
            config = _read_json(required[0])
            manifest = _read_json(required[1])
            summary = _read_json(required[2])
            task_payload = _read_json(required[3])
            sync_manifest = _read_json(required[4])
            if manifest.get("method") != display_name:
                raise ValueError(f"Method identity mismatch: {root}")
            if manifest.get("protocol_id") != "emotic_b10c4_v0.1":
                raise ValueError(f"Protocol identity mismatch: {root}")
            if manifest.get("track") != "A" or int(manifest.get("seed", -1)) != seed:
                raise ValueError(f"Track/seed mismatch: {root}")
            if manifest.get("git_commit") != args.expected_git_commit:
                raise ValueError(f"Git commit mismatch: {root}")
            if manifest.get("git_dirty") is not False:
                raise ValueError(f"Dirty formal result: {root}")
            if manifest.get("reporting_split") != "test":
                raise ValueError(f"Non-test formal result: {root}")
            if manifest.get("configuration_locked") is not True:
                raise ValueError(f"Unlocked formal result: {root}")
            if manifest.get("test_labels_used_for_selection") is not False:
                raise ValueError(f"Test labels influenced selection: {root}")
            if manifest.get("eligible_for_main_table") is not True:
                raise ValueError(f"Ineligible formal result: {root}")
            if manifest.get("prediction_reused_task_ids") not in ([], None):
                raise ValueError(f"Predictions were reused across protocols: {root}")
            protocol = config.get("protocol")
            if not isinstance(protocol, Mapping):
                raise ValueError(f"Missing resolved protocol: {root}")
            tasks = protocol.get("tasks")
            if not isinstance(tasks, list) or [len(task) for task in tasks] != [10, 4, 4, 4, 4]:
                raise ValueError(f"Not an alphabetic B10-C4 task split: {root}")
            class_order = protocol.get("class_order")
            if not isinstance(class_order, list) or class_order != sorted(
                class_order, key=str.casefold
            ):
                raise ValueError(f"Class order is not alphabetical: {root}")
            current_class_hash = str(manifest.get("class_order_hash", ""))
            current_tree_hash = str(manifest.get("source_tree_hash", ""))
            current_protocol_hash = str(manifest.get("protocol_hash", ""))
            if class_order_hash is None:
                class_order_hash = current_class_hash
                source_tree_hash = current_tree_hash
            if current_class_hash != class_order_hash or current_tree_hash != source_tree_hash:
                raise ValueError("Class order or source tree differs across jobs")
            seed_key = str(seed)
            if seed_key not in protocol_hashes:
                protocol_hashes[seed_key] = current_protocol_hash
            if current_protocol_hash != protocol_hashes[seed_key]:
                raise ValueError("Protocol hash differs within one seed")

            task_rows = task_payload.get("tasks")
            if not isinstance(task_rows, list) or len(task_rows) != 5:
                raise ValueError(f"Expected five task metrics: {root}")
            if [int(row.get("task_id", -1)) for row in task_rows] != list(range(5)):
                raise ValueError(f"Task metric IDs differ: {root}")
            scores = sorted((root / "scores").glob("task*_scores.pt"))
            if [path.name for path in scores] != [
                f"task{task}_scores.pt" for task in range(5)
            ]:
                raise ValueError(f"Canonical score files differ: {root}")
            if sync_manifest.get("contains_pth") is not False:
                raise ValueError(f"Sync bundle may contain checkpoints: {root}")
            if list((root / "results_to_sync" / args.run_id).rglob("*.pth")):
                raise RuntimeError(f"Sync bundle contains .pth: {root}")

            main_table = summary.get("main_table")
            details = summary.get("summary")
            if not isinstance(main_table, Mapping) or not isinstance(details, Mapping):
                raise ValueError(f"Incomplete summary: {root}")
            row = {
                "seed": seed,
                **{
                    metric: _finite(
                        main_table.get(metric), f"{display_name} seed{seed} {metric}"
                    )
                    for metric in METRICS
                },
                "replay_memory_samples": int(
                    main_table.get("replay_memory", {}).get("samples", -1)
                ),
                "replay_memory_bytes": int(
                    main_table.get("replay_memory", {}).get("bytes", -1)
                ),
                "parameter_growth": int(main_table.get("parameter_growth", -1)),
            }
            task_curve = [
                _finite(task.get("mAP"), f"{display_name} seed{seed} task mAP")
                for task in details.get("task_metrics", [])
            ]
            if len(task_curve) != 5:
                raise ValueError(f"Summary task curve differs: {root}")
            per_seed.append(row)
            task_curves.append(task_curve)
            total_bundles += 1

        if method_key in {"er", "prs", "derpp"}:
            if [row["replay_memory_samples"] for row in per_seed] != [520, 520, 520]:
                raise ValueError(f"{display_name} final replay capacity differs")
        if memory_display == "0" and any(
            row["replay_memory_samples"] != 0 for row in per_seed
        ):
            raise ValueError(f"{display_name} unexpectedly reports replay memory")
        aggregate = {
            metric: _aggregate([row[metric] for row in per_seed])
            for metric in METRICS
        }
        aggregate.update(
            {
                "replay_memory_samples": _aggregate(
                    [float(row["replay_memory_samples"]) for row in per_seed]
                ),
                "replay_memory_bytes": _aggregate(
                    [float(row["replay_memory_bytes"]) for row in per_seed]
                ),
                "parameter_growth": _aggregate(
                    [float(row["parameter_growth"]) for row in per_seed]
                ),
            }
        )
        per_task = [
            {
                "task": task,
                **_aggregate([curve[task] for curve in task_curves]),
            }
            for task in range(5)
        ]
        methods.append(
            {
                "method_key": method_key,
                "display_name": display_name,
                "memory_display": memory_display,
                "per_seed": per_seed,
                "aggregate": aggregate,
                "per_task_mAP": per_task,
            }
        )

    if total_bundles != 36:
        raise RuntimeError(f"Expected 36 formal bundles, found {total_bundles}")
    ranking = {
        metric: [
            method["display_name"]
            for method in sorted(
                methods,
                key=lambda item: item["aggregate"][metric]["mean"],
                reverse=(metric != "forgetting"),
            )
        ]
        for metric in METRICS
    }
    payload = {
        "summary_schema_version": 1,
        "run_id": args.run_id,
        "protocol_id": "emotic_b10c4_v0.1",
        "track": "A",
        "task_sizes": [10, 4, 4, 4, 4],
        "seen_class_counts": [10, 14, 18, 22, 26],
        "class_order": "alphabetical_same_26_class_order_as_b5c3",
        "class_order_hash": class_order_hash,
        "protocol_hash_by_seed": protocol_hashes,
        "git_commit": args.expected_git_commit,
        "source_tree_hash": source_tree_hash,
        "configuration_policy": "reuse_frozen_b5c3_hyperparameters_without_b10c4_test_tuning",
        "reporting_split": "test",
        "configuration_locked": True,
        "bundle_count": total_bundles,
        "aggregation": "mean_and_sample_standard_deviation",
        "methods": methods,
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.markdown_output, methods)
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(args.markdown_output),
        "bundle_count": total_bundles,
        "best_final_mAP": ranking["final_mAP"][0],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
