#!/usr/bin/env python3
"""Validate and aggregate locked ER/PRS Track-A formal results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_METHODS = ("ER", "PRS")
LOCK_CONFIRMATION = "REPLAY_20C_TRACK_A_V0_1"
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
    "contract_id": "emotic_replay_20c_v0.1",
    "protocol_id": "emotic_b5c3_v0.1",
    "sample_unit": "unique_emotic_person_sample",
    "capacity_kind": "per_seen_class",
    "samples_per_seen_class": 20,
    "task_capacities": list(EXPECTED_CAPACITIES),
    "replay_to_current_ratio": 1.0,
    "update_timing": "task_end_single_pass",
    "image_payload": "post_transform_float32_tensor",
    "deduplicate_by_sample_id": True,
    "stored_targets": "visible_columns_only",
    "future_truth_stored": False,
}
COMMON_METHOD_CONFIGURATION = {
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
    "replay_contract": REPLAY_CONTRACT,
    "prs_allocation_power": -0.03,
    "training_lifecycle": "task_based_replay_then_task_end_update",
    "current_and_replay_equal_sample_weight": True,
    "replay_loss": "visible_masked_sigmoid_bce",
}
METHOD_CONFIGURATION = {
    "ER": {
        "strategy": "er",
        "replay_policy": "reservoir",
        "upstream_repository": "Repository-native controlled replay baseline",
        "upstream_commit": "N/A",
        "source_original_capacity": None,
        "source_original_update_timing": "N/A",
        "track_a_primary_budget_differs_from_prs_source": False,
    },
    "PRS": {
        "strategy": "prs",
        "replay_policy": "partitioned_reservoir",
        "upstream_repository": "https://github.com/cdjkim/PRS",
        "upstream_commit": "136cee1863af03cc914dc05dfd41bda8b7bc0bf2",
        "source_original_capacity": 2000,
        "source_original_update_timing": "online_after_each_optimizer_step",
        "track_a_primary_budget_differs_from_prs_source": True,
    },
}
LOCKED_RUNNER_CONFIGURATION = {
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "num_workers": 2,
    "reporting_split": "test",
    "configuration_locked": True,
    "execution_mode": "sequential",
    "core_runtime_version": "0.9.0",
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
    attempts: List[int] = []
    skips: List[int] = []
    observations: List[int] = []
    memory_samples: List[int] = []
    memory_bytes: List[int] = []
    for task_id in range(8):
        rows = records[task_id]
        final_rows = [row for row in rows if "memory_update_observations" in row]
        _require_equal(len(final_rows), 1, f"task{task_id} final training rows")
        final = final_rows[0]
        epoch_values = [int(_finite(row.get("epoch"), "epoch")) for row in rows]
        completed = max(epoch_values) + 1
        if completed < 1 or completed > 10:
            raise ValueError(f"task{task_id} completed invalid epoch count {completed}")
        attempted = int(_finite(final.get("optimizer_attempts"), "optimizer attempts"))
        skipped = int(_finite(final.get("amp_overflow_skips"), "AMP skips"))
        observed = int(
            _finite(final.get("memory_update_observations"), "memory observations")
        )
        samples = int(_finite(final.get("replay_samples_after"), "memory samples"))
        byte_count = int(_finite(final.get("replay_bytes_after"), "memory bytes"))
        if attempted <= 0 or skipped < 0 or skipped > attempted or byte_count <= 0:
            raise ValueError(f"task{task_id} has invalid optimizer/memory statistics")
        _require_equal(observed, EXPECTED_TRAINING_SAMPLES[task_id], f"task{task_id} observations")
        _require_equal(samples, EXPECTED_CAPACITIES[task_id], f"task{task_id} capacity")
        epochs.append(completed)
        attempts.append(attempted)
        skips.append(skipped)
        observations.append(observed)
        memory_samples.append(samples)
        memory_bytes.append(byte_count)

    lowered = text.lower()
    return {
        "epochs_completed_by_task": epochs,
        "optimizer_attempts": sum(attempts),
        "optimizer_attempts_by_task": attempts,
        "amp_overflow_skips": sum(skips),
        "amp_overflow_skips_by_task": skips,
        "memory_update_observations_by_task": observations,
        "replay_samples_after_by_task": memory_samples,
        "replay_bytes_after_by_task": memory_bytes,
        "all_logged_values_finite": True,
        "nan_detected": bool(re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered)),
        "oom_detected": "out of memory" in lowered,
        "traceback_detected": "traceback (most recent call last)" in lowered,
    }


def validate_replay_formal_results(
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

    source: Dict[str, Any] = {}
    protocol_hashes: Dict[str, Dict[str, Any]] = {}
    results: Dict[str, Any] = {}
    for method in EXPECTED_METHODS:
        per_seed: List[Dict[str, Any]] = []
        task_curves: List[Tuple[float, ...]] = []
        stability: List[Dict[str, Any]] = []
        protocol_hashes[method] = {}
        for seed in seed_values:
            seed_root = _single_path(
                root,
                f"benchmarks/*/A/{method}/seed{seed}",
                f"{method} seed {seed} artifact directory",
            )
            manifest = _read_object(seed_root / "run_manifest.json")
            config = _read_object(seed_root / "config_resolved.json")
            summary = _read_object(seed_root / "metrics" / "summary.json")
            tasks_payload = _read_object(seed_root / "metrics" / "task_metrics.json")

            expected_manifest = {
                "method": method,
                "protocol_id": "emotic_b5c3_v0.1",
                "track": "A",
                "seed": seed,
                "git_commit": expected_git_commit,
                "git_dirty": False,
                "core_runtime_version": "0.9.0",
                "test_labels_used_for_selection": False,
                "reporting_split": "test",
                "configuration_locked": True,
                "prediction_reused_task_ids": [],
                "eligible_for_main_table": True,
                "replay_memory_samples": 520,
                **EXPECTED_PARAMETERS,
            }
            _validate_subset(manifest, expected_manifest, f"{method} seed{seed} manifest")
            if int(manifest.get("replay_memory_bytes", 0)) <= 0:
                raise ValueError(f"{method} seed{seed} replay bytes must be positive")

            method_config = manifest.get("method_configuration")
            if not isinstance(method_config, Mapping):
                raise ValueError(f"{method} seed{seed} method configuration missing")
            _validate_subset(
                method_config,
                {**COMMON_METHOD_CONFIGURATION, **METHOD_CONFIGURATION[method]},
                f"{method} seed{seed} method configuration",
            )
            contract_path = str(method_config.get("replay_contract_path", ""))
            if not contract_path.endswith("/configs/emotic_mlcil/replay_20c_v0.1.yaml"):
                raise ValueError(f"{method} seed{seed} replay contract path differs")

            runner = config.get("runner")
            if not isinstance(runner, Mapping):
                raise ValueError(f"{method} seed{seed} runner configuration missing")
            _validate_subset(
                runner, LOCKED_RUNNER_CONFIGURATION, f"{method} seed{seed} runner"
            )
            protocol = config.get("protocol")
            if not isinstance(protocol, Mapping):
                raise ValueError(f"{method} seed{seed} protocol configuration missing")
            _validate_subset(
                protocol,
                LOCKED_PROTOCOL_CONFIGURATION,
                f"{method} seed{seed} protocol",
            )
            _require_equal(protocol.get("seed"), seed, f"{method} seed{seed} protocol seed")
            _require_equal(
                protocol.get("protocol_hash"),
                manifest.get("protocol_hash"),
                f"{method} seed{seed} protocol hash",
            )
            protocol_hashes[method][str(seed)] = manifest.get("protocol_hash")

            main = summary.get("main_table")
            if not isinstance(main, Mapping):
                raise ValueError(f"{method} seed{seed} main table missing")
            metrics = {
                name: _finite(main.get(name), f"{method} seed{seed} {name}")
                for name in AGGREGATE_METRICS
            }
            expected_memory = {
                "samples": 520,
                "bytes": int(manifest["replay_memory_bytes"]),
            }
            _require_equal(main.get("replay_memory"), expected_memory, f"{method} seed{seed} memory")

            task_rows = tasks_payload.get("tasks")
            if not isinstance(task_rows, list) or len(task_rows) != 8:
                raise ValueError(f"{method} seed{seed} must contain eight task metrics")
            _require_equal(
                [int(row.get("task_id", -1)) for row in task_rows],
                list(range(8)),
                f"{method} seed{seed} task IDs",
            )
            task_curves.append(
                tuple(
                    _finite(row.get("mAP"), f"{method} seed{seed} task mAP")
                    for row in task_rows
                )
            )
            score_tasks = sorted(
                int(match.group(1))
                for path in (seed_root / "scores").glob("task*_scores.pt")
                if (match := re.fullmatch(r"task(\d+)_scores\.pt", path.name))
            )
            _require_equal(score_tasks, list(range(8)), f"{method} seed{seed} score tasks")

            current_source = {
                "git_commit": manifest.get("git_commit"),
                "source_tree_hash": manifest.get("source_tree_hash"),
                "class_order_hash": manifest.get("class_order_hash"),
                "data_split_hash": manifest.get("data_split_hash"),
                "core_base_commit": manifest.get("core_base_commit"),
                "core_runtime_version": manifest.get("core_runtime_version"),
            }
            if not source:
                source = current_source
            else:
                _require_equal(current_source, source, f"{method} seed{seed} source")

            run_stability = _training_stability(seed_root / "train.log")
            for failure in ("nan_detected", "oom_detected", "traceback_detected"):
                if run_stability[failure]:
                    raise ValueError(f"{method} seed{seed} training reports {failure}")
            per_seed.append(
                {
                    "seed": seed,
                    **metrics,
                    "replay_memory_samples": 520,
                    "replay_memory_bytes": int(manifest["replay_memory_bytes"]),
                }
            )
            stability.append({"seed": seed, **run_stability})

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
        results[method] = {
            "per_seed": per_seed,
            "aggregate": aggregate,
            "per_task_mAP": per_task,
            "training_stability": stability,
        }

    comparisons = {}
    for metric in AGGREGATE_METRICS:
        comparisons[f"PRS_minus_ER_{metric}"] = (
            results["PRS"]["aggregate"][metric]["mean"]
            - results["ER"]["aggregate"][metric]["mean"]
        )
    return {
        "formal_result_schema_version": 1,
        "methods": list(EXPECTED_METHODS),
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "run_id": run_id,
        "status": "eligible_for_main_table",
        "configuration_lock_confirmation": LOCK_CONFIRMATION,
        "seeds": list(seed_values),
        "execution": {
            "physical_gpu": 0,
            "max_concurrent_training_processes": 3,
            "waves": ["ER_seed012", "PRS_seed012"],
            "automatic_wave_handoff": True,
        },
        "source": {**source, "protocol_hash_by_method_and_seed": protocol_hashes},
        "results": results,
        "comparison": comparisons,
        "configuration_was_frozen_before_all_held_out_seeds": True,
        "test_metrics_must_not_change_configuration": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = validate_replay_formal_results(
        run_root=args.run_root,
        run_id=args.run_id,
        seeds=args.seeds,
        expected_git_commit=args.expected_git_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
