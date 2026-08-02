#!/usr/bin/env python3
"""Validate and aggregate locked CSC seeds before result packaging."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


EXPECTED_SEEDS = (0, 1, 2)
LOCK_CONFIRMATION = "CSC_TRACK_A_V0_1"
AGGREGATE_METRICS = (
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "csc",
    "upstream_commit": "0bab38a00d6e0555f2df855ae2fe8db1fea68b12",
    "visual_encoder_trainable": True,
    "clip_text_encoder_used": False,
    "benchmark_added_adapter": False,
    "replay_enabled": False,
    "epochs": 20,
    "learning_rate": 4.0e-5,
    "weight_decay": 1.0e-4,
    "one_cycle_pct_start": 0.2,
    "alpha": 0.5,
    "entropy_strength": 0.04,
    "entropy_scope": "all_seen_classes_as_released",
    "branch_combination": "arithmetic_mean_as_released",
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
    "core_runtime_version": "0.4.0",
}


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


def validate_csc_formal_results(
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
    provenance: Dict[str, Any] = {}
    protocol_hash_by_seed: Dict[str, Any] = {}
    for seed in seed_values:
        seed_root = _single_path(
            root,
            f"benchmarks/*/A/CSC/seed{seed}",
            f"CSC seed {seed} artifact directory",
        )
        manifest = _read_object(seed_root / "run_manifest.json")
        config = _read_object(seed_root / "config_resolved.json")
        summary = _read_object(seed_root / "metrics" / "summary.json")

        expected_manifest = {
            "method": "CSC",
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": expected_git_commit,
            "git_dirty": False,
            "core_runtime_version": "0.4.0",
            "test_labels_used_for_selection": False,
            "reporting_split": "test",
            "configuration_locked": True,
            "prediction_reused_task_ids": [],
            "eligible_for_main_table": True,
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
        # The registered seed is part of the protocol payload, so its hash is
        # intentionally seed-specific even though class order and data split
        # must remain identical across the formal runs.
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
        replay = main_table.get("replay_memory")
        _require_equal(replay, {"samples": 0, "bytes": 0}, f"seed{seed} replay")

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
            _require_equal(
                current_provenance,
                provenance,
                f"seed{seed} formal provenance",
            )
        per_seed.append({"seed": seed, **values})

    aggregate = {}
    for metric in AGGREGATE_METRICS:
        values = [row[metric] for row in per_seed]
        aggregate[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
        }
    return {
        "formal_result_schema_version": 1,
        "method": "CSC",
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "run_id": run_id,
        "status": "eligible_for_main_table",
        "configuration_lock_confirmation": LOCK_CONFIRMATION,
        "seeds": list(seed_values),
        "source": {
            **provenance,
            "protocol_hash_by_seed": protocol_hash_by_seed,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "fixed_threshold_calibration_note": (
            "Primary mAP is threshold-free; cF1/oF1 retain the protocol's fixed "
            "0.5 threshold and must not trigger post-test calibration."
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
    payload = validate_csc_formal_results(
        args.run_root,
        args.run_id,
        args.seeds,
        args.expected_git_commit,
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
