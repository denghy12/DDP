#!/usr/bin/env python3
"""Validate and aggregate locked MULTI-LANE Track-A formal seeds."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


EXPECTED_SEEDS = (0, 1, 2)
LOCK_CONFIRMATION = "MULTI_LANE_TRACK_A_V0_1"
AGGREGATE_METRICS = (
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "multi_lane",
    "upstream_commit": "5ee982c9298d4cfd6af471d9bb2ef3c0aad05373",
    "upstream_archive_sha256": (
        "dfe84ea31f6d7e51877c2661791c716aed7d888b8d783d16ce067cb8ce022d49"
    ),
    "visual_encoder_trainable": False,
    "clip_text_encoder_used": False,
    "benchmark_added_adapter": False,
    "replay_enabled": False,
    "task_lane_capacity_preallocated": True,
    "task_lane_parameters_total": 675840,
    "task_lane_parameters_per_task": 84480,
    "shared_classifier_parameters": 13338,
    "physical_parameter_growth_after_initialization": 0,
    "epochs": 30,
    "effective_learning_rate": 0.0125,
    "registered_train_batch_size": 64,
    "weight_decay": 0.0,
    "num_selectors": 10,
    "num_prompts": 10,
    "num_prompt_layers": 5,
    "normalize": "pre-head",
    "temperature": 1.0,
    "inference_lane_scope": "all_seen_lanes_concat",
    "task_oracle_at_inference": False,
    "training_label_scope": "current_classes_only",
    "old_future_ground_truth_used_for_training": False,
    "amp": True,
    "tf32": True,
}
LOCKED_RUNNER_CONFIGURATION = {
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "num_workers": 2,
    "reporting_split": "test",
    "configuration_locked": True,
    "execution_mode": "sequential",
    "core_runtime_version": "0.5.0",
}
EXPECTED_PARAMETER_STATISTICS = {
    "total_parameters": 86881818,
    "trainable_parameters": 689178,
    "incremental_parameters": 0,
    "per_task_incremental_parameters": {str(task): 0 for task in range(8)},
}
TRAINING_PATTERN = re.compile(r"^task=(\d+) training=(\{.*\})$")


def _read_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _single_path(root: Path, pattern: str, role: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {role}, found {len(matches)}")
    return matches[0]


def _require_equal(actual: Any, expected: Any, role: str) -> None:
    if actual != expected:
        raise ValueError(f"{role}={actual!r}; expected {expected!r}")


def _validate_locked_subset(
    actual: Mapping[str, Any], expected: Mapping[str, Any], role: str
) -> None:
    for key, expected_value in expected.items():
        _require_equal(actual.get(key), expected_value, f"{role}.{key}")


def _training_stability(path: Path) -> Dict[str, Any]:
    records: Dict[int, List[Mapping[str, Any]]] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = TRAINING_PATTERN.match(line)
        if match is None:
            continue
        task_id = int(match.group(1))
        payload = json.loads(match.group(2))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Training record must be an object: {path}")
        records.setdefault(task_id, []).append(payload)
    _require_equal(tuple(sorted(records)), tuple(range(8)), "training task IDs")

    successful_updates = 0
    skipped_updates = 0
    for task_id in range(8):
        task_records = records[task_id]
        _require_equal(len(task_records), 30, f"task{task_id} epoch count")
        _require_equal(
            [int(record.get("epoch", -1)) for record in task_records],
            list(range(30)),
            f"task{task_id} epoch sequence",
        )
        if "validation_current_mAP" not in task_records[-1]:
            raise ValueError(f"task{task_id} final epoch lacks validation mAP")
        for record in task_records:
            for key in ("current_loss", "learning_rate", "next_learning_rate"):
                value = record.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(
                    float(value)
                ):
                    raise ValueError(f"task{task_id} {key} must be finite")
            successful_updates += int(record.get("optimizer_steps", 0))
            skipped_updates += int(record.get("skipped_optimizer_steps", 0))
    lowered = text.lower()
    return {
        "epochs_completed": sum(len(values) for values in records.values()),
        "attempted_optimizer_updates": successful_updates + skipped_updates,
        "optimizer_updates": successful_updates,
        "amp_overflow_skips": skipped_updates,
        "all_logged_values_finite": True,
        "nan_detected": bool(re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered)),
        "oom_detected": "out of memory" in lowered,
        "traceback_detected": "traceback (most recent call last)" in lowered,
    }


def validate_multi_lane_formal_results(
    run_root: Path,
    run_id: str,
    seeds: Sequence[int],
    expected_git_commit: str,
) -> Dict[str, Any]:
    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing formal run root: {root}")
    seed_values = tuple(int(seed) for seed in seeds)
    _require_equal(seed_values, EXPECTED_SEEDS, "formal seeds")
    if not expected_git_commit or len(expected_git_commit) != 40:
        raise ValueError("expected-git-commit must be a full 40-character SHA")

    per_seed: List[Dict[str, Any]] = []
    per_seed_task_maps: List[Tuple[float, ...]] = []
    stability: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}
    protocol_hash_by_seed: Dict[str, Any] = {}
    for seed in seed_values:
        seed_root = _single_path(
            root,
            f"benchmarks/*/A/MULTI-LANE/seed{seed}",
            f"MULTI-LANE seed {seed} artifact directory",
        )
        manifest = _read_object(seed_root / "run_manifest.json")
        config = _read_object(seed_root / "config_resolved.json")
        summary = _read_object(seed_root / "metrics" / "summary.json")

        expected_manifest = {
            "method": "MULTI-LANE",
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": expected_git_commit,
            "git_dirty": False,
            "core_runtime_version": "0.5.0",
            "test_labels_used_for_selection": False,
            "reporting_split": "test",
            "configuration_locked": True,
            "prediction_reused_task_ids": [],
            "eligible_for_main_table": True,
            "replay_memory_samples": 0,
            "replay_memory_bytes": 0,
            **EXPECTED_PARAMETER_STATISTICS,
        }
        _validate_locked_subset(manifest, expected_manifest, f"seed{seed} manifest")
        method_configuration = manifest.get("method_configuration")
        if not isinstance(method_configuration, Mapping):
            raise ValueError(f"seed{seed} method_configuration must be an object")
        _validate_locked_subset(
            method_configuration,
            LOCKED_METHOD_CONFIGURATION,
            f"seed{seed} method_configuration",
        )

        runner_configuration = config.get("runner")
        if not isinstance(runner_configuration, Mapping):
            raise ValueError(f"seed{seed} runner configuration must be an object")
        _validate_locked_subset(
            runner_configuration,
            LOCKED_RUNNER_CONFIGURATION,
            f"seed{seed} runner",
        )
        protocol_configuration = config.get("protocol")
        if not isinstance(protocol_configuration, Mapping):
            raise ValueError(f"seed{seed} protocol configuration must be an object")
        _require_equal(
            protocol_configuration.get("seed"), seed, f"seed{seed} protocol seed"
        )
        _require_equal(
            protocol_configuration.get("protocol_hash"),
            manifest.get("protocol_hash"),
            f"seed{seed} protocol hash",
        )
        _require_equal(
            protocol_configuration.get("class_order_hash"),
            manifest.get("class_order_hash"),
            f"seed{seed} class-order hash",
        )
        protocol_hash_by_seed[str(seed)] = manifest.get("protocol_hash")

        main_table = summary.get("main_table")
        if not isinstance(main_table, Mapping):
            raise ValueError(f"seed{seed} main_table must be an object")
        values: Dict[str, float] = {}
        for metric in AGGREGATE_METRICS:
            value = main_table.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"seed{seed} {metric} must be finite")
            values[metric] = float(value)
        _require_equal(
            main_table.get("replay_memory"),
            {"samples": 0, "bytes": 0},
            f"seed{seed} replay",
        )
        task_metrics = summary.get("task_metrics")
        if not isinstance(task_metrics, list) or len(task_metrics) != 8:
            raise ValueError(f"seed{seed} must contain eight task metrics")
        _require_equal(
            [int(row.get("task_id", -1)) for row in task_metrics],
            list(range(8)),
            f"seed{seed} task metric IDs",
        )
        per_seed_task_maps.append(tuple(float(row["mAP"]) for row in task_metrics))

        current_provenance = {
            "git_commit": manifest.get("git_commit"),
            "source_tree_hash": manifest.get("source_tree_hash"),
            "class_order_hash": manifest.get("class_order_hash"),
            "data_split_hash": manifest.get("data_split_hash"),
            "core_base_commit": manifest.get("core_base_commit"),
            "core_runtime_version": manifest.get("core_runtime_version"),
        }
        if seed == seed_values[0]:
            provenance = current_provenance
        else:
            _require_equal(current_provenance, provenance, f"seed{seed} provenance")
        per_seed.append({"seed": seed, **values})
        run_stability = _training_stability(seed_root / "train.log")
        for failure_key in ("nan_detected", "oom_detected", "traceback_detected"):
            if run_stability[failure_key]:
                raise ValueError(f"seed{seed} training log reports {failure_key}")
        stability.append({"seed": seed, **run_stability})

    aggregate: Dict[str, Any] = {}
    for metric in AGGREGATE_METRICS:
        metric_values = [row[metric] for row in per_seed]
        aggregate[metric] = {
            "mean": statistics.mean(metric_values),
            "std": statistics.stdev(metric_values),
        }
    aggregate["aggregation"] = "mean_and_sample_standard_deviation"
    per_task = []
    for task_id in range(8):
        values = [rows[task_id] for rows in per_seed_task_maps]
        per_task.append(
            {
                "task": task_id,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
            }
        )
    return {
        "formal_result_schema_version": 1,
        "method": "MULTI-LANE",
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "run_id": run_id,
        "status": "eligible_for_main_table",
        "configuration_lock_confirmation": LOCK_CONFIRMATION,
        "seeds": list(seed_values),
        "source": {**provenance, "protocol_hash_by_seed": protocol_hash_by_seed},
        "per_seed": per_seed,
        "aggregate": aggregate,
        "per_task_mAP": per_task,
        "training_stability": stability,
        "parameter_reporting_note": (
            "parameter_growth=0 means all eight task lanes were allocated at "
            "initialization; report 675840 task-lane parameters and 13338 "
            "shared-classifier parameters alongside physical growth."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_multi_lane_formal_results(
        args.run_root, args.run_id, args.seeds, args.expected_git_commit
    )
    if args.output.exists():
        raise FileExistsError(f"Formal summary already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
