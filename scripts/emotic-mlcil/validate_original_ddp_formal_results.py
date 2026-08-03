#!/usr/bin/env python3
"""Validate and aggregate locked Original-DDP-Tau2 Track-A formal results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


EXPECTED_SEEDS = (0, 1, 2)
LOCK_CONFIRMATION = "ORIGINAL_DDP_TAU2_TRACK_A_V0_1"
AGGREGATE_METRICS = (
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "original_ddp",
    "source_kind": "fixed_local_original_ddp_snapshot_without_git_metadata",
    "source_snapshot_tree_sha256": (
        "e0b9963e1d891ce95fc0c0d2444389f388fe5a1c2bd04f6ece7dc0617306e5a4"
    ),
    "source_model_sha256": (
        "d7dfb6bd463b387db3224a8fb6631e25435bae5c174e65f22f328899fc42b309"
    ),
    "source_git_commit": None,
    "source_license_status": "no_license_file_observed_reference_only",
    "visual_encoder_trainable": False,
    "text_encoder_optimizer_updated": False,
    "clip_text_encoder_used": True,
    "class_specific_positive_negative_text_prompts": True,
    "class_specific_interlayer_visual_prompts": True,
    "prompt_capacity_preallocated": True,
    "prompt_capacity_parameters": 1064960,
    "prompt_capacity_parameters_per_class": 40960,
    "benchmark_added_adapter": False,
    "replay_enabled": False,
    "optimizer_lifecycle": "one Adam instance retained across all tasks",
    "scheduler_lifecycle": "one MultiStepLR instance retained across all tasks",
    "loss_reduction": "released sum over samples and current classes",
    "training_label_scope": "current_classes_only",
    "old_future_ground_truth_used_for_training": False,
    "prompt_initialization": "released random class-specific contexts",
    "source_pcd_temperature": {"minimum": 1.0, "maximum": 7.0, "gamma": 0.2},
    "registered_pcd_temperature": {
        "minimum": 1.0,
        "maximum": 2.0,
        "gamma": 0.7,
    },
    "epochs": 20,
    "optimizer_lr": 0.0059,
    "loss_weight": 0.03,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1.0e-8,
    "scheduler_milestones": [0, 20],
    "scheduler_gamma": 0.1,
    "temperature_minimum": 1.0,
    "temperature_maximum": 2.0,
    "temperature_gamma": 0.7,
    "n_ctx_positive": 16,
    "n_ctx_negative": 16,
    "visual_prompt_length": 16,
    "visual_prompt_layers": [7, 8, 9, 10, 11],
    "registered_train_batch_size": 8,
    "amp": True,
    "tf32": True,
}
LOCKED_RUNNER_CONFIGURATION = {
    "train_batch_size": 8,
    "eval_batch_size": 1,
    "num_workers": 0,
    "reporting_split": "test",
    "configuration_locked": True,
    "execution_mode": "sequential",
    "core_runtime_version": "0.7.0",
}
EXPECTED_PARAMETER_STATISTICS = {
    "total_parameters": 125388800,
    "trainable_parameters": 1064960,
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


def _finite_number(value: Any, role: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{role} must be finite")
    return float(value)


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

    attempted_by_task: List[int] = []
    applied_by_task: List[int] = []
    skipped_by_task: List[int] = []
    first_loss_by_task: List[float] = []
    final_loss_by_task: List[float] = []
    current_validation_map: List[float] = []
    learning_rate_by_task: List[float] = []
    for task_id in range(8):
        task_records = records[task_id]
        _require_equal(len(task_records), 20, f"task{task_id} epoch record count")
        _require_equal(
            [int(row.get("epoch", -1)) for row in task_records],
            list(range(20)),
            f"task{task_id} epoch IDs",
        )
        expected_lr = 0.00059 if task_id == 0 else 0.000059
        learning_rates = [
            _finite_number(row.get("learning_rate"), f"task{task_id} learning rate")
            for row in task_records
        ]
        if any(not math.isclose(value, expected_lr) for value in learning_rates):
            raise ValueError(f"task{task_id} learning-rate schedule differs")
        applied = sum(
            int(_finite_number(row.get("optimizer_steps"), "optimizer steps"))
            for row in task_records
        )
        skipped = sum(
            int(
                _finite_number(
                    row.get("skipped_optimizer_steps"), "skipped optimizer steps"
                )
            )
            for row in task_records
        )
        if applied <= 0 or skipped < 0:
            raise ValueError(f"task{task_id} optimizer accounting is invalid")
        _require_equal(
            int(task_records[-1].get("scheduler_last_epoch", -1)),
            (task_id + 1) * 20,
            f"task{task_id} scheduler last epoch",
        )
        attempted_by_task.append(applied + skipped)
        applied_by_task.append(applied)
        skipped_by_task.append(skipped)
        first_loss_by_task.append(
            _finite_number(task_records[0].get("scaled_source_bce"), "first loss")
        )
        final_loss_by_task.append(
            _finite_number(task_records[-1].get("scaled_source_bce"), "final loss")
        )
        current_validation_map.append(
            _finite_number(
                task_records[-1].get("validation_current_mAP"),
                f"task{task_id} current validation mAP",
            )
        )
        learning_rate_by_task.append(expected_lr)

    lowered = text.lower()
    attempted = sum(attempted_by_task)
    applied = sum(applied_by_task)
    skipped = sum(skipped_by_task)
    return {
        "attempted_optimizer_updates": attempted,
        "optimizer_updates": applied,
        "amp_overflow_skips": skipped,
        "amp_overflow_skip_fraction": skipped / attempted,
        "attempted_updates_by_task": attempted_by_task,
        "optimizer_updates_by_task": applied_by_task,
        "amp_overflow_skips_by_task": skipped_by_task,
        "first_loss_by_task": first_loss_by_task,
        "final_loss_by_task": final_loss_by_task,
        "current_task_validation_mAP": current_validation_map,
        "effective_learning_rate_by_task": learning_rate_by_task,
        "scheduler_last_epoch": 160,
        "all_logged_values_finite": True,
        "nan_detected": bool(re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered)),
        "oom_detected": "out of memory" in lowered,
        "traceback_detected": "traceback (most recent call last)" in lowered,
    }


def validate_original_ddp_formal_results(
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
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise ValueError("expected-git-commit must be a full lowercase SHA")

    per_seed: List[Dict[str, Any]] = []
    per_seed_task_maps: List[Tuple[float, ...]] = []
    stability: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}
    protocol_hash_by_seed: Dict[str, Any] = {}
    for seed in seed_values:
        seed_root = _single_path(
            root,
            f"benchmarks/*/A/Original-DDP-Tau2/seed{seed}",
            f"Original-DDP-Tau2 seed {seed} artifact directory",
        )
        manifest = _read_object(seed_root / "run_manifest.json")
        config = _read_object(seed_root / "config_resolved.json")
        summary = _read_object(seed_root / "metrics" / "summary.json")
        task_payload = _read_object(seed_root / "metrics" / "task_metrics.json")

        expected_manifest = {
            "method": "Original-DDP-Tau2",
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": expected_git_commit,
            "git_dirty": False,
            "core_runtime_version": "0.7.0",
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
            values[metric] = _finite_number(
                main_table.get(metric), f"seed{seed} {metric}"
            )
        _require_equal(
            main_table.get("replay_memory"),
            {"samples": 0, "bytes": 0},
            f"seed{seed} replay",
        )
        task_metrics = task_payload.get("tasks")
        if not isinstance(task_metrics, list) or len(task_metrics) != 8:
            raise ValueError(f"seed{seed} must contain eight task metrics")
        _require_equal(
            [int(row.get("task_id", -1)) for row in task_metrics],
            list(range(8)),
            f"seed{seed} task metric IDs",
        )
        per_seed_task_maps.append(
            tuple(
                _finite_number(row.get("mAP"), f"seed{seed} task mAP")
                for row in task_metrics
            )
        )

        current_provenance = {
            "git_commit": manifest.get("git_commit"),
            "source_tree_hash": manifest.get("source_tree_hash"),
            "source_snapshot_tree_sha256": method_configuration.get(
                "source_snapshot_tree_sha256"
            ),
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

    aggregate: Dict[str, Any] = {
        "aggregation": "mean_and_sample_standard_deviation"
    }
    for metric in AGGREGATE_METRICS:
        metric_values = [row[metric] for row in per_seed]
        aggregate[metric] = {
            "mean": statistics.mean(metric_values),
            "std": statistics.stdev(metric_values),
        }
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
        "method": "Original-DDP-Tau2",
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
        "configuration_was_frozen_before_all_held_out_seeds": True,
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
    payload = validate_original_ddp_formal_results(
        args.run_root, args.run_id, args.seeds, args.expected_git_commit
    )
    if args.output.exists():
        raise FileExistsError(f"Formal validation output exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
