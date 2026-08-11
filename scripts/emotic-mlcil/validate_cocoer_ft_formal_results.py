#!/usr/bin/env python3
"""Validate and aggregate locked CocoER-FT Track-B formal results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


EXPECTED_SEEDS = (0, 1, 2)
LOCK_CONFIRMATION = "COCOER_FT_TRACK_B_V0_1"
AGGREGATE_METRICS = (
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "sequential_finetuning",
    "conversion_interface": "CocoER-FT-v0.1",
    "upstream_commit": "dac8fc139e61b87f1bf0b27c581798df2a5a9d38",
    "input_mode": "cocoer_multilevel",
    "full_26_class_gwt_checkpoint_used": False,
    "full_26_class_vi_mapping_checkpoint_used": False,
    "current_label_only": True,
    "old_label_truth_used": False,
    "future_label_truth_used": False,
    "distillation_enabled": False,
    "replay_enabled": False,
    "ewc_enabled": False,
    "benchmark_added_adapter": False,
    "clip_visual_backbone": "OpenAI CLIP RN50 (native CocoER component)",
    "clip_text_encoder_used": False,
    "selection_metric": "current_label_validation_mAP",
    "f1_threshold": 0.5,
    "epochs": 20,
    "early_stopping_patience": 20,
    "learning_rate": 6.0e-5,
    "lr_step_epochs": 3,
    "lr_gamma": 0.1,
    "beta1": 0.9,
    "beta2": 0.96,
    "weight_decay": 0.01,
    "inside_lr": 0.1,
    "pseudo_threshold": 0.3,
    "grad_distance_weight": 0.1,
    "gradient_clip_norm": 10.0,
    "encoder_blocks": 3,
    "amp": True,
    "tf32": False,
}
LOCKED_RUNNER_CONFIGURATION = {
    "train_batch_size": 64,
    "eval_batch_size": 16,
    "num_workers": 0,
    "reporting_split": "test",
    "configuration_locked": True,
    "execution_mode": "sequential",
    "core_runtime_version": "0.10.1",
}
EXPECTED_ASSET_HASHES = {
    "resnet50_initialization_sha256": (
        "ac8cddaea082ebd9932cdcd604c91aafa01d98b517a185c5926dc35a1e4dd277"
    ),
    "clip_rn50_sha256": (
        "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
    ),
    "head_box_cache_sha256": (
        "f4a0795bd55d101f712d4e5fe2e46dbeda215d939a5cd37b992884bec62d309f"
    ),
}


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _require(actual: Any, expected: Any, role: str) -> None:
    if actual != expected:
        raise ValueError(f"{role}={actual!r}; expected {expected!r}")


def _subset(actual: Mapping[str, Any], expected: Mapping[str, Any], role: str) -> None:
    for key, value in expected.items():
        _require(actual.get(key), value, f"{role}.{key}")


def _finite(value: Any, role: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{role} must be finite")
    return float(value)


def validate(
    run_root: Path,
    run_id: str,
    seeds: Sequence[int],
    gpus: Sequence[int],
    expected_git_commit: str,
) -> Dict[str, Any]:
    root = run_root.resolve()
    seed_values = tuple(int(seed) for seed in seeds)
    gpu_values = tuple(int(gpu) for gpu in gpus)
    _require(seed_values, EXPECTED_SEEDS, "formal seeds")
    if len(gpu_values) != 3 or len(set(gpu_values)) != 3:
        raise ValueError("CocoER-FT formal seeds require three distinct GPUs")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise ValueError("expected-git-commit must be a full lowercase SHA")

    per_seed: List[Dict[str, Any]] = []
    curves: Dict[str, List[float]] = {}
    provenance: Dict[str, Any] = {}
    protocol_hashes: Dict[str, Any] = {}
    for seed, gpu in zip(seed_values, gpu_values):
        matches = sorted(root.glob(f"benchmarks/*/B/CocoER-FT/seed{seed}"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one CocoER-FT seed{seed} artifact, found {len(matches)}")
        seed_root = matches[0]
        manifest = _read(seed_root / "run_manifest.json")
        config = _read(seed_root / "config_resolved.json")
        summary = _read(seed_root / "metrics" / "summary.json")
        tasks = _read(seed_root / "metrics" / "task_metrics.json").get("tasks")
        expected_manifest = {
            "method": "CocoER-FT",
            "protocol_id": "emotic_b5c3_track_b_v0.1",
            "track": "B",
            "seed": seed,
            "git_commit": expected_git_commit,
            "git_dirty": False,
            "core_runtime_version": "0.10.1",
            "test_labels_used_for_selection": False,
            "checkpoint_selection_split": "val",
            "reporting_split": "test",
            "configuration_locked": True,
            "prediction_reused_task_ids": [],
            "eligible_for_main_table": True,
            "replay_memory_samples": 0,
            "replay_memory_bytes": 0,
            "total_parameters": 307509666,
            "trainable_parameters": 269192770,
            "incremental_parameters": 26985,
        }
        _subset(manifest, expected_manifest, f"seed{seed} manifest")
        method = manifest.get("method_configuration")
        runner = config.get("runner")
        protocol = config.get("protocol")
        if not all(isinstance(value, Mapping) for value in (method, runner, protocol)):
            raise ValueError(f"seed{seed} resolved configuration is incomplete")
        _subset(method, LOCKED_METHOD_CONFIGURATION, f"seed{seed} method")
        _subset(method, EXPECTED_ASSET_HASHES, f"seed{seed} assets")
        _subset(runner, LOCKED_RUNNER_CONFIGURATION, f"seed{seed} runner")
        _require(protocol.get("track"), "B", f"seed{seed} protocol track")
        _require(protocol.get("seed"), seed, f"seed{seed} protocol seed")
        _require(protocol.get("threshold_policy", {}).get("value"), 0.5, f"seed{seed} threshold")
        if not isinstance(tasks, list) or len(tasks) != 8:
            raise ValueError(f"seed{seed} must contain eight task rows")
        _require([row.get("task_id") for row in tasks], list(range(8)), f"seed{seed} task IDs")
        curves[str(seed)] = [
            _finite(row.get("mAP"), f"seed{seed} task mAP") for row in tasks
        ]
        main = summary.get("main_table")
        if not isinstance(main, Mapping):
            raise ValueError(f"seed{seed} main table is missing")
        metrics = {
            name: _finite(main.get(name), f"seed{seed} {name}")
            for name in AGGREGATE_METRICS
        }
        _require(main.get("replay_memory"), {"samples": 0, "bytes": 0}, f"seed{seed} replay")
        log_text = (seed_root / "train.log").read_text(encoding="utf-8").lower()
        if "traceback (most recent call last)" in log_text or "out of memory" in log_text:
            raise RuntimeError(f"seed{seed} training log contains a fatal marker")
        current = {
            "git_commit": manifest.get("git_commit"),
            "source_tree_hash": manifest.get("source_tree_hash"),
            "class_order_hash": manifest.get("class_order_hash"),
            "data_split_hash": manifest.get("data_split_hash"),
            "core_base_commit": manifest.get("core_base_commit"),
            "core_runtime_version": manifest.get("core_runtime_version"),
        }
        if seed == 0:
            provenance = current
        else:
            _require(current, provenance, f"seed{seed} provenance")
        protocol_hashes[str(seed)] = manifest.get("protocol_hash")
        per_seed.append({"seed": seed, "physical_gpu": gpu, **metrics})

    aggregate = {
        metric: {
            "mean": statistics.mean([row[metric] for row in per_seed]),
            "std": statistics.stdev([row[metric] for row in per_seed]),
        }
        for metric in AGGREGATE_METRICS
    }
    aggregate["aggregation"] = "mean_and_sample_standard_deviation"
    return {
        "formal_result_schema_version": 1,
        "method": "CocoER-FT",
        "protocol_id": "emotic_b5c3_track_b_v0.1",
        "track": "B",
        "run_id": run_id,
        "status": "eligible_for_main_table",
        "configuration_lock_confirmation": LOCK_CONFIRMATION,
        "configuration_freeze_basis": (
            "source-registered configuration plus successful batch-64 CUDA smoke; "
            "standalone seed-0 validation was still running and no validation or test "
            "metric was read before this lock"
        ),
        "seeds": list(seed_values),
        "gpus": list(gpu_values),
        "source": {**provenance, "protocol_hash_by_seed": protocol_hashes},
        "per_seed": per_seed,
        "task_mAP_curves": curves,
        "aggregate": aggregate,
        "post_test_tuning_forbidden": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = validate(
        args.run_root, args.run_id, args.seeds, args.gpus, args.expected_git_commit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
