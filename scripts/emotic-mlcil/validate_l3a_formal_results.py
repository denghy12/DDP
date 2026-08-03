#!/usr/bin/env python3
"""Validate locked L3A Track-A formal results and aggregate three seeds."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ALLOWED_SEED_SEQUENCES = ((0,), (0, 1, 2))
LOCK_CONFIRMATION = "L3A_TRACK_A_V0_1"
AGGREGATE_METRICS = (
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "l3a",
    "upstream_commit": "1067bbd6124a7aa96136baa555080ba9183ab2ac",
    "upstream_tree": "9e6ac470600328f2a393f26e30ea764c3b50102b",
    "upstream_archive_sha256": (
        "307d721d76069a7fe95a10346be1323a842cd9624f8d55200f4cf7cd45462409"
    ),
    "visual_encoder_trainability": "Task 0 only",
    "clip_text_encoder_used": False,
    "benchmark_added_adapter": False,
    "replay_enabled": False,
    "old_future_ground_truth_used_for_training": False,
    "analytic_state_dtype": "float64",
    "analytic_state_elements": 16883712,
    "analytic_state_bytes": 135069696,
    "feature_dim": 512,
    "hidden_dim": 4096,
    "base_epochs": 1,
    "base_learning_rate": 4.0e-5,
    "weight_decay": 1.0e-4,
    "one_cycle_pct_start": 0.2,
    "analytic_repeats": 1,
    "ridge": 1.0,
    "pseudo_label": True,
    "pseudo_threshold": 0.7,
    "weighted_analytic": True,
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
    "core_runtime_version": "0.6.0",
}
EXPECTED_PARAMETER_STATISTICS = {
    "total_parameters": 88396288,
    "trainable_parameters": 86195205,
    "incremental_parameters": 86016,
    "per_task_incremental_parameters": {
        "0": 0,
        "1": 12288,
        "2": 12288,
        "3": 12288,
        "4": 12288,
        "5": 12288,
        "6": 12288,
        "7": 12288,
    },
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
    _require_equal(len(records[0]), 2, "task0 training record count")
    for task_id in range(1, 8):
        _require_equal(
            len(records[task_id]), 1, f"task{task_id} training record count"
        )

    base = records[0][0]
    _require_equal(base.get("base_gradient_training"), 1.0, "task0 gradient flag")
    optimizer_updates = int(
        _finite_number(base.get("optimizer_steps"), "task0 optimizer steps")
    )
    amp_skips = int(
        _finite_number(base.get("skipped_optimizer_steps"), "task0 AMP skips")
    )
    if optimizer_updates <= 0 or amp_skips < 0:
        raise ValueError("Task-0 optimizer counts are invalid")
    base_loss = _finite_number(base.get("base_asl_loss"), "task0 ASL loss")
    final_lr = _finite_number(base.get("learning_rate"), "task0 learning rate")

    analytic_samples: List[int] = []
    pseudo_positives: List[int] = []
    current_validation_map: List[float] = []
    for task_id in range(8):
        analytic = records[task_id][-1]
        samples = int(
            _finite_number(
                analytic.get("analytic_samples"), f"task{task_id} analytic samples"
            )
        )
        counted = int(
            _finite_number(
                analytic.get("counted_samples"), f"task{task_id} counted samples"
            )
        )
        positives = int(
            _finite_number(
                analytic.get("pseudo_positive_labels"),
                f"task{task_id} pseudo positives",
            )
        )
        if samples <= 0 or counted != samples or positives < 0:
            raise ValueError(f"task{task_id} analytic accounting is invalid")
        analytic_samples.append(samples)
        pseudo_positives.append(positives)
        current_validation_map.append(
            _finite_number(
                analytic.get("validation_current_mAP"),
                f"task{task_id} validation mAP",
            )
        )

    lowered = text.lower()
    return {
        "attempted_optimizer_updates": optimizer_updates + amp_skips,
        "optimizer_updates": optimizer_updates,
        "amp_overflow_skips": amp_skips,
        "base_asl_loss": base_loss,
        "final_learning_rate": final_lr,
        "analytic_samples_by_task": analytic_samples,
        "pseudo_positive_labels_by_task": pseudo_positives,
        "current_task_validation_mAP": current_validation_map,
        "all_logged_values_finite": True,
        "nan_detected": bool(re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered)),
        "oom_detected": "out of memory" in lowered,
        "traceback_detected": "traceback (most recent call last)" in lowered,
    }


def validate_l3a_formal_results(
    run_root: Path,
    run_id: str,
    seeds: Sequence[int],
    expected_git_commit: str,
) -> Dict[str, Any]:
    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing formal run root: {root}")
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values not in ALLOWED_SEED_SEQUENCES:
        raise ValueError("L3A formal validation accepts seed 0 or seeds 0 1 2")
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
            f"benchmarks/*/A/L3A/seed{seed}",
            f"L3A seed {seed} artifact directory",
        )
        manifest = _read_object(seed_root / "run_manifest.json")
        config = _read_object(seed_root / "config_resolved.json")
        summary = _read_object(seed_root / "metrics" / "summary.json")
        task_payload = _read_object(seed_root / "metrics" / "task_metrics.json")

        expected_manifest = {
            "method": "L3A",
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": expected_git_commit,
            "git_dirty": False,
            "core_runtime_version": "0.6.0",
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

    aggregation = (
        "single_seed_compliance_gate"
        if len(seed_values) == 1
        else "mean_and_sample_standard_deviation"
    )
    aggregate: Dict[str, Any] = {"aggregation": aggregation}
    for metric in AGGREGATE_METRICS:
        metric_values = [row[metric] for row in per_seed]
        aggregate[metric] = {
            "mean": statistics.mean(metric_values),
            "std": statistics.stdev(metric_values) if len(metric_values) > 1 else None,
        }
    per_task = []
    for task_id in range(8):
        values = [rows[task_id] for rows in per_seed_task_maps]
        per_task.append(
            {
                "task": task_id,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
        )
    return {
        "formal_result_schema_version": 1,
        "method": "L3A",
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "run_id": run_id,
        "status": (
            "seed0_compliance_gate_passed"
            if seed_values == (0,)
            else "eligible_for_main_table"
        ),
        "configuration_lock_confirmation": LOCK_CONFIRMATION,
        "seeds": list(seed_values),
        "source": {**provenance, "protocol_hash_by_seed": protocol_hash_by_seed},
        "per_seed": per_seed,
        "aggregate": aggregate,
        "per_task_mAP": per_task,
        "training_stability": stability,
        "seed0_gate_does_not_select_configuration_from_metrics": True,
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
    payload = validate_l3a_formal_results(
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
