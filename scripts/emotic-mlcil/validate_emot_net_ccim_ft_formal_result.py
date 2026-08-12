#!/usr/bin/env python3
"""Validate one locked EMOT-Net+CCIM-FT Track-B held-out test run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


def _object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _equal(actual: Any, expected: Any, role: str) -> None:
    if actual != expected:
        raise ValueError(f"{role}={actual!r}; expected {expected!r}")


def _finite(value: Any, role: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{role} must be finite")
    return float(value)


def validate(run_root: Path, run_id: str, expected_git_commit: str) -> dict:
    root = run_root.resolve()
    matches = sorted(root.glob("benchmarks/*/B/EMOT-Net+CCIM-FT/seed0"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one EMOT-Net+CCIM-FT seed0 artifact, found {len(matches)}"
        )
    seed_root = matches[0]
    manifest = _object(seed_root / "run_manifest.json")
    config = _object(seed_root / "config_resolved.json")
    summary = _object(seed_root / "metrics" / "summary.json")
    task_payload = _object(seed_root / "metrics" / "task_metrics.json")

    expected_manifest = {
        "method": "EMOT-Net+CCIM-FT",
        "method_family": "Static EMOTIC model / Sequential Fine-Tuning",
        "protocol_id": "emotic_b5c3_track_b_v0.1",
        "track": "B",
        "seed": 0,
        "git_commit": expected_git_commit,
        "git_dirty": False,
        "core_runtime_version": "0.12.1",
        "test_labels_used_for_selection": False,
        "checkpoint_selection_split": "val",
        "reporting_split": "test",
        "configuration_locked": True,
        "eligible_for_main_table": True,
        "replay_memory_samples": 0,
        "replay_memory_bytes": 0,
    }
    for key, expected in expected_manifest.items():
        _equal(manifest.get(key), expected, f"manifest.{key}")

    runner = config.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("runner configuration must be an object")
    for key, expected in {
        "train_batch_size": 52,
        "eval_batch_size": 16,
        "num_workers": 8,
        "reporting_split": "test",
        "configuration_locked": True,
    }.items():
        _equal(runner.get(key), expected, f"runner.{key}")

    method = manifest.get("method_configuration")
    if not isinstance(method, Mapping):
        raise ValueError("method_configuration must be an object")
    for key, expected in {
        "strategy": "sequential_finetuning",
        "current_label_only": True,
        "old_label_truth_used": False,
        "future_label_truth_used": False,
        "replay_enabled": False,
        "distillation_enabled": False,
        "benchmark_added_adapter": False,
        "ccim_dictionary_frozen_after_task0": True,
        "ccim_future_task_images_used": False,
        "epochs": 21,
        "early_stopping_patience": 21,
        "learning_rate": 0.01,
        "lr_drop_epoch": 7,
        "lr_drop_gamma": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "amp": False,
        "tf32": False,
        "execution_tower_model_parallel": True,
        "execution_tower_model_parallel_devices": ["cuda:0", "cuda:1", "cuda:2"],
        "execution_cuda_visible_devices": "2,3,4",
    }.items():
        _equal(method.get(key), expected, f"method.{key}")

    tasks = task_payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 8:
        raise ValueError("Formal result must contain eight task metrics")
    _equal([row.get("task_id") for row in tasks], list(range(8)), "task IDs")
    for task_id, row in enumerate(tasks):
        for metric in ("mAP", "cF1", "oF1"):
            _finite(row.get(metric), f"task{task_id}.{metric}")
        score = seed_root / "scores" / f"task{task_id}_scores.pt"
        if not score.is_file() or score.stat().st_size == 0:
            raise FileNotFoundError(score)

    main_table = summary.get("main_table")
    if not isinstance(main_table, Mapping):
        raise ValueError("main_table must be an object")
    metrics = {
        key: _finite(main_table.get(key), f"main_table.{key}")
        for key in (
            "final_mAP",
            "final_cF1",
            "final_oF1",
            "average_mAP",
            "forgetting",
            "parameter_growth",
        )
    }
    training_log = (seed_root / "train.log").read_text(encoding="utf-8")
    lowered = training_log.lower()
    if "out of memory" in lowered or "traceback (most recent call last)" in lowered:
        raise ValueError("Formal training log contains a failure marker")
    if re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered):
        raise ValueError("Formal training log contains NaN")
    records = [line for line in training_log.splitlines() if re.match(r"^task=\d+ training=", line)]
    if len(records) != 8 * 21:
        raise ValueError(f"Expected 168 epoch records, found {len(records)}")

    return {
        "result_schema_version": 1,
        "run_id": run_id,
        "protocol_id": "emotic_b5c3_track_b_v0.1",
        "track": "B",
        "method": "EMOT-Net+CCIM-FT",
        "formal_seeds": [0],
        "aggregate_statistics": "not_applicable_single_seed",
        "git_commit": expected_git_commit,
        "main_table": metrics,
        "per_task_mAP": [float(row["mAP"]) for row in tasks],
        "execution": {
            "physical_gpus": [2, 3, 4],
            "local_devices": ["cuda:0", "cuda:1", "cuda:2"],
            "parallelism": "full_batch_native_tower_model_parallel",
            "train_batch_size": 52,
            "workers": 8,
        },
        "eligibility": {
            "reporting_split": "test",
            "configuration_locked": True,
            "test_labels_used_for_selection": False,
            "eligible_for_main_table": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.run_root, args.run_id, args.expected_git_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
