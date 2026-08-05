#!/usr/bin/env python3
"""Validate and aggregate locked DER++ Track-A formal results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


EXPECTED_SEEDS = (0, 1, 2)
LOCK_CONFIRMATION = "DERPP_20C_TRACK_A_V0_1"
AGGREGATE_METRICS = (
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)
EXPECTED_CAPACITIES = (100, 160, 220, 280, 340, 400, 460, 520)
EXPECTED_TRAINING_SAMPLES = (5353, 4394, 861, 9931, 4352, 2536, 627, 1526)
EXPECTED_PARAMETERS = {
    "total_parameters": 86205978,
    "trainable_parameters": 86205978,
    "incremental_parameters": 10773,
    "per_task_incremental_parameters": {
        "0": 0,
        "1": 1539,
        "2": 1539,
        "3": 1539,
        "4": 1539,
        "5": 1539,
        "6": 1539,
        "7": 1539,
    },
}
REPLAY_CONTRACT = {
    "contract_id": "emotic_derpp_replay_20c_v0.1",
    "protocol_id": "emotic_b5c3_v0.1",
    "sample_unit": "unique_emotic_person_sample",
    "capacity_kind": "per_seen_class",
    "samples_per_seen_class": 20,
    "task_capacities": list(EXPECTED_CAPACITIES),
    "replay_to_current_ratio": 1.0,
    "update_timing": "online_after_optimizer_attempt",
    "image_payload": "post_transform_float32_tensor",
    "stores_logits": True,
    "stored_logits": "capture_time_visible_columns_only",
    "replay_draws_per_current_batch": 2,
    "objective_weighting": "source_derpp_weighted_sum",
    "deduplicate_by_sample_id": True,
    "stored_targets": "visible_columns_only",
    "future_truth_stored": False,
}
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "derpp",
    "visual_encoder_trainable": True,
    "clip_text_encoder_used": False,
    "benchmark_added_adapter": False,
    "selection_metric": "current_label_validation_mAP",
    "preprocessing": "shared_DDP_EMOTIC_full_image",
    "feature_dim": 512,
    "epochs": 10,
    "early_stopping_patience": 3,
    "backbone_learning_rate": 1.0e-5,
    "head_learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "gradient_clip_norm": 1.0,
    "amp": True,
    "tf32": True,
    "replay_policy": "reservoir",
    "replay_contract": REPLAY_CONTRACT,
    "upstream_repository": "https://github.com/aimagelab/mammoth",
    "upstream_commit": "cb9a36d788d6ad051c9eee0da358b25421d909f5",
    "source_original_capacity": None,
    "source_original_update_timing": "online_after_each_optimizer_step",
    "training_lifecycle": "online_pre_update_logits_then_validation_snapshot_restore",
    "current_and_replay_equal_sample_weight": False,
    "replay_loss": (
        "alpha_masked_capture_logit_mse_plus_beta_visible_masked_sigmoid_bce"
    ),
    "derpp_alpha": 0.5,
    "derpp_beta": 0.5,
    "independent_replay_draws_per_current_batch": 2,
    "same_task_replay_enabled": True,
    "stored_logit_scope": "capture_time_seen_columns",
    "stable_id_logit_merge": "latest_coherent_image_logit_capture",
    "upstream_tag": "neurips2020",
    "upstream_archive_sha256": (
        "d7cdffefdb7d77939a1055984cb586ad83af0220439cd76e4a106ea201c1695b"
    ),
    "upstream_derpp_sha256": (
        "d38736e8d8c888300a8e5ddcac7cab1ac12e2eb1a0aa4c28f2387bdd79fc973a"
    ),
    "source_reported_buffer_sizes": [200, 500, 5120],
    "track_a_primary_budget_differs_from_source": True,
    "multi_label_mapping": "softmax_cross_entropy_to_visible_masked_sigmoid_bce",
}
LOCKED_RUNNER_CONFIGURATION = {
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "num_workers": 2,
    "reporting_split": "test",
    "configuration_locked": True,
    "execution_mode": "sequential",
    "core_runtime_version": "0.10.0",
}
LOCKED_PROTOCOL_CONFIGURATION = {
    "protocol_id": "emotic_b5c3_v0.1",
    "threshold_policy": {
        "kind": "fixed_global",
        "value": 0.5,
        "per_task_selection": False,
        "per_class_selection": False,
    },
    "primary_metric": "final_mAP",
    "checkpoint_selection": {
        "split": "val",
        "metric": "mAP",
        "mode": "max",
        "tie_break": "earliest_epoch",
        "test_allowed": False,
    },
    "track": "A",
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


def _validate_subset(
    actual: Mapping[str, Any], expected: Mapping[str, Any], role: str
) -> None:
    for key, expected_value in expected.items():
        _require_equal(actual.get(key), expected_value, f"{role}.{key}")


def _finite(value: Any, role: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{role} must be finite")
    return float(value)


def _read_execution_marker(path: Path) -> Mapping[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _training_stability(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    records: Dict[int, List[Mapping[str, Any]]] = {}
    for line in text.splitlines():
        match = TRAINING_PATTERN.match(line)
        if match is None:
            continue
        payload = json.loads(match.group(2))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Training record must be an object: {path}")
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                _finite(value, f"task{match.group(1)} training.{key}")
        records.setdefault(int(match.group(1)), []).append(payload)
    _require_equal(tuple(sorted(records)), tuple(range(8)), "training task IDs")

    epochs: List[int] = []
    selected_epochs: List[int] = []
    attempts: List[int] = []
    skips: List[int] = []
    observations: List[int] = []
    replay_examples: List[int] = []
    memory_samples: List[int] = []
    memory_bytes: List[int] = []
    dark_losses: List[float] = []
    label_losses: List[float] = []
    for task_id in range(8):
        rows = records[task_id]
        selected_rows = [row for row in rows if "selected_epoch" in row]
        _require_equal(len(selected_rows), 1, f"task{task_id} selected training rows")
        selected = selected_rows[0]
        epoch_values = [int(_finite(row.get("epoch"), "epoch")) for row in rows]
        completed = max(epoch_values) + 1
        selected_epoch = int(_finite(selected.get("selected_epoch"), "selected epoch"))
        if completed < 1 or completed > 10 or selected_epoch not in epoch_values:
            raise ValueError(f"task{task_id} has invalid epoch selection")
        attempted = int(_finite(selected.get("optimizer_attempts"), "optimizer attempts"))
        skipped = int(_finite(selected.get("amp_overflow_skips"), "AMP skips"))
        observed = int(
            _finite(selected.get("memory_update_observations"), "memory observations")
        )
        replayed = int(_finite(selected.get("replay_examples"), "replay examples"))
        samples = int(
            _finite(selected.get("selected_replay_samples"), "memory samples")
        )
        byte_count = int(
            _finite(selected.get("selected_replay_bytes"), "memory bytes")
        )
        if attempted <= 0 or skipped < 0 or skipped > attempted or byte_count <= 0:
            raise ValueError(f"task{task_id} has invalid optimizer/memory statistics")
        _require_equal(
            observed, EXPECTED_TRAINING_SAMPLES[task_id], f"task{task_id} observations"
        )
        _require_equal(samples, EXPECTED_CAPACITIES[task_id], f"task{task_id} capacity")
        epochs.append(completed)
        selected_epochs.append(selected_epoch)
        attempts.append(attempted)
        skips.append(skipped)
        observations.append(observed)
        replay_examples.append(replayed)
        memory_samples.append(samples)
        memory_bytes.append(byte_count)
        dark_losses.append(_finite(selected.get("dark_logit_loss"), "dark logit loss"))
        label_losses.append(_finite(selected.get("replay_label_loss"), "replay label loss"))

    if memory_bytes != sorted(memory_bytes):
        raise ValueError("DER++ replay bytes must be nondecreasing")
    lowered = text.lower()
    return {
        "epochs_completed_by_task": epochs,
        "selected_epoch_zero_based_by_task": selected_epochs,
        "optimizer_attempts": sum(attempts),
        "optimizer_attempts_by_task": attempts,
        "amp_overflow_skips": sum(skips),
        "amp_overflow_skips_by_task": skips,
        "successful_optimizer_updates": sum(attempts) - sum(skips),
        "memory_update_observations_by_selected_epoch": observations,
        "two_draw_replay_examples_by_selected_epoch": replay_examples,
        "selected_replay_samples_by_task": memory_samples,
        "selected_replay_bytes_by_task": memory_bytes,
        "selected_dark_logit_loss_by_task": dark_losses,
        "selected_replay_label_loss_by_task": label_losses,
        "all_logged_values_finite": True,
        "nan_detected": bool(re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered)),
        "oom_detected": "out of memory" in lowered,
        "traceback_detected": "traceback (most recent call last)" in lowered,
    }


def validate_derpp_formal_results(
    run_root: Path,
    run_id: str,
    seeds: Sequence[int],
    gpus: Sequence[int],
    expected_git_commit: str,
) -> Dict[str, Any]:
    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing formal run root: {root}")
    seed_values = tuple(int(seed) for seed in seeds)
    gpu_values = tuple(int(gpu) for gpu in gpus)
    _require_equal(seed_values, EXPECTED_SEEDS, "formal seeds")
    if len(gpu_values) != 3 or len(set(gpu_values)) != 3:
        raise ValueError("DER++ formal execution requires three distinct GPUs")
    if any(gpu < 0 for gpu in gpu_values):
        raise ValueError("GPU IDs must be non-negative")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise ValueError("expected-git-commit must be a full lowercase SHA")

    per_seed: List[Dict[str, Any]] = []
    task_curves: List[Tuple[float, ...]] = []
    stability: List[Dict[str, Any]] = []
    source: Dict[str, Any] = {}
    protocol_hashes: Dict[str, Any] = {}
    execution_assignments: List[Dict[str, int]] = []
    for seed, gpu in zip(seed_values, gpu_values):
        seed_root = _single_path(
            root,
            f"benchmarks/*/A/DER++/seed{seed}",
            f"DER++ seed {seed} artifact directory",
        )
        manifest = _read_object(seed_root / "run_manifest.json")
        config = _read_object(seed_root / "config_resolved.json")
        summary = _read_object(seed_root / "metrics" / "summary.json")
        tasks_payload = _read_object(seed_root / "metrics" / "task_metrics.json")

        expected_manifest = {
            "method": "DER++",
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": expected_git_commit,
            "git_dirty": False,
            "core_runtime_version": "0.10.0",
            "test_labels_used_for_selection": False,
            "checkpoint_selection_split": "val",
            "reporting_split": "test",
            "configuration_locked": True,
            "prediction_reused_task_ids": [],
            "eligible_for_main_table": True,
            "replay_memory_samples": 520,
            **EXPECTED_PARAMETERS,
        }
        _validate_subset(manifest, expected_manifest, f"DER++ seed{seed} manifest")
        replay_bytes = int(manifest.get("replay_memory_bytes", 0))
        if replay_bytes <= 0:
            raise ValueError(f"DER++ seed{seed} replay bytes must be positive")

        method_config = manifest.get("method_configuration")
        if not isinstance(method_config, Mapping):
            raise ValueError(f"DER++ seed{seed} method configuration missing")
        _validate_subset(
            method_config,
            LOCKED_METHOD_CONFIGURATION,
            f"DER++ seed{seed} method configuration",
        )
        contract_path = str(method_config.get("replay_contract_path", ""))
        if not contract_path.endswith(
            "/configs/emotic_mlcil/replay_derpp_20c_v0.1.yaml"
        ):
            raise ValueError(f"DER++ seed{seed} replay contract path differs")

        runner = config.get("runner")
        protocol = config.get("protocol")
        if not isinstance(runner, Mapping) or not isinstance(protocol, Mapping):
            raise ValueError(f"DER++ seed{seed} resolved configuration missing")
        _validate_subset(runner, LOCKED_RUNNER_CONFIGURATION, f"DER++ seed{seed} runner")
        _validate_subset(
            protocol, LOCKED_PROTOCOL_CONFIGURATION, f"DER++ seed{seed} protocol"
        )
        _require_equal(protocol.get("seed"), seed, f"DER++ seed{seed} protocol seed")
        _require_equal(
            protocol.get("protocol_hash"),
            manifest.get("protocol_hash"),
            f"DER++ seed{seed} protocol hash",
        )
        protocol_hashes[str(seed)] = manifest.get("protocol_hash")

        main = summary.get("main_table")
        if not isinstance(main, Mapping):
            raise ValueError(f"DER++ seed{seed} main table missing")
        metrics = {
            name: _finite(main.get(name), f"DER++ seed{seed} {name}")
            for name in AGGREGATE_METRICS
        }
        _require_equal(
            main.get("replay_memory"),
            {"samples": 520, "bytes": replay_bytes},
            f"DER++ seed{seed} memory",
        )

        task_rows = tasks_payload.get("tasks")
        if not isinstance(task_rows, list) or len(task_rows) != 8:
            raise ValueError(f"DER++ seed{seed} must contain eight task metrics")
        _require_equal(
            [int(row.get("task_id", -1)) for row in task_rows],
            list(range(8)),
            f"DER++ seed{seed} task IDs",
        )
        task_curves.append(
            tuple(
                _finite(row.get("mAP"), f"DER++ seed{seed} task mAP")
                for row in task_rows
            )
        )
        score_tasks = sorted(
            int(match.group(1))
            for score_path in (seed_root / "scores").glob("task*_scores.pt")
            if (match := re.fullmatch(r"task(\d+)_scores\.pt", score_path.name))
        )
        _require_equal(score_tasks, list(range(8)), f"DER++ seed{seed} score tasks")

        current_source = {
            "git_commit": manifest.get("git_commit"),
            "source_tree_hash": manifest.get("source_tree_hash"),
            "upstream_commit": method_config.get("upstream_commit"),
            "upstream_archive_sha256": method_config.get("upstream_archive_sha256"),
            "class_order_hash": manifest.get("class_order_hash"),
            "data_split_hash": manifest.get("data_split_hash"),
            "core_base_commit": manifest.get("core_base_commit"),
            "core_runtime_version": manifest.get("core_runtime_version"),
        }
        if not source:
            source = current_source
        else:
            _require_equal(current_source, source, f"DER++ seed{seed} source")

        run_stability = _training_stability(seed_root / "train.log")
        for failure in ("nan_detected", "oom_detected", "traceback_detected"):
            if run_stability[failure]:
                raise ValueError(f"DER++ seed{seed} training reports {failure}")
        _require_equal(
            run_stability["selected_replay_bytes_by_task"][-1],
            replay_bytes,
            f"DER++ seed{seed} final logged replay bytes",
        )

        marker = _read_execution_marker(root / "runtime_logs" / f"seed{seed}.done")
        _require_equal(marker.get("seed"), str(seed), f"DER++ seed{seed} marker seed")
        _require_equal(
            marker.get("physical_gpu"), str(gpu), f"DER++ seed{seed} marker GPU"
        )
        _require_equal(marker.get("exit_code"), "0", f"DER++ seed{seed} marker exit")
        _require_equal(
            marker.get("phase"),
            "three_seed_three_gpu_parallel",
            f"DER++ seed{seed} marker phase",
        )

        per_seed.append(
            {
                "seed": seed,
                **metrics,
                "replay_memory_samples": 520,
                "replay_memory_bytes": replay_bytes,
            }
        )
        stability.append({"seed": seed, **run_stability})
        execution_assignments.append({"seed": seed, "physical_gpu": gpu})

    aggregate: Dict[str, Any] = {
        "aggregation": "mean_and_sample_standard_deviation"
    }
    for metric in AGGREGATE_METRICS:
        values = [row[metric] for row in per_seed]
        aggregate[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
        }
    for metric in ("replay_memory_samples", "replay_memory_bytes"):
        values = [row[metric] for row in per_seed]
        aggregate[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
        }
    per_task = []
    for task_id in range(8):
        values = [curve[task_id] for curve in task_curves]
        per_task.append(
            {
                "task": task_id,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
            }
        )
    return {
        "formal_result_schema_version": 1,
        "method": "DER++",
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "run_id": run_id,
        "status": "eligible_for_main_table",
        "configuration_lock_confirmation": LOCK_CONFIRMATION,
        "seeds": list(seed_values),
        "execution": {
            "mode": "three_seed_three_gpu_parallel",
            "max_concurrent_training_processes": 3,
            "assignments": execution_assignments,
        },
        "source": {**source, "protocol_hash_by_seed": protocol_hashes},
        "per_seed": per_seed,
        "aggregate": aggregate,
        "per_task_mAP": per_task,
        "training_stability": stability,
        "configuration_was_frozen_before_all_held_out_seeds": True,
        "test_metrics_must_not_change_configuration": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = validate_derpp_formal_results(
        run_root=args.run_root,
        run_id=args.run_id,
        seeds=args.seeds,
        gpus=args.gpus,
        expected_git_commit=args.expected_git_commit,
    )
    if args.output.exists():
        raise FileExistsError(f"Formal validation output exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
