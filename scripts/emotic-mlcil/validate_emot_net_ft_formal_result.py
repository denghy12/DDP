#!/usr/bin/env python3
"""Validate one locked EMOT-Net-FT Track-B held-out result."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping


LOCK_CONFIRMATION = "EMOT_NET_FT_TRACK_B_V0_1"
EXPECTED_TASKS = tuple(range(8))
EXPECTED_STEPS_PER_EPOCH = (103, 85, 17, 191, 84, 49, 13, 30)
EXPECTED_PARAMETER_STATISTICS = {
    "total_parameters": 14069914,
    "trainable_parameters": 14069914,
    "incremental_parameters": 5397,
    "per_task_incremental_parameters": {
        "0": 0,
        "1": 771,
        "2": 771,
        "3": 771,
        "4": 771,
        "5": 771,
        "6": 771,
        "7": 771,
    },
}
LOCKED_RUNNER_CONFIGURATION = {
    "train_batch_size": 52,
    "eval_batch_size": 16,
    "num_workers": 0,
    "reporting_split": "test",
    "configuration_locked": True,
    "execution_mode": "sequential",
    "core_runtime_version": "0.10.0",
}
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "sequential_finetuning",
    "conversion_interface": "EMOT-Net-FT-v0.1",
    "upstream_repository": "https://github.com/rkosti/emotic",
    "upstream_commit": "69c3a5106aed08121cd12f6a5b359c745136931e",
    "upstream_license": "MIT",
    "native_context_backbone": "model_myVDavg_640_Places.t7 architecture",
    "native_body_backbone": "alexnet_features.t7 official release architecture",
    "native_release_archive_sha256": (
        "ce6096c1af5a3e91badbc06752e2dbbd04fc63f67f24acc95d76a68e1f7e339b"
    ),
    "native_body_variant": "official_dropbox_alexnet",
    "native_initialization_sha256": (
        "036d0778d7ddb94682b20fbb635fdf9c8f5d43e8f09fd6773ac7ab9245d11027"
    ),
    "native_source_asset_sha256": {
        "model_myVDavg_640_Places.t7": (
            "bbf8a09edb1a17338f8004b78cf2e83f3cccc3a7f0bf7c3705368b482cab1e7c"
        ),
        "alexnet_features.t7": (
            "0abdbce4910f4c242433d614287448d110a81e7d26562ab291364762cf2dae87"
        ),
    },
    "input_mode": "body_context",
    "preprocessing": "EMOTIC mean/std; context 224x224; body 128x128; no augmentation",
    "current_label_only": True,
    "old_label_truth_used": False,
    "future_label_truth_used": False,
    "distillation_enabled": False,
    "replay_enabled": False,
    "ewc_enabled": False,
    "benchmark_added_adapter": False,
    "clip_visual_encoder_used": False,
    "clip_text_encoder_used": False,
    "loss": "source weighted sigmoid MSE over current classes only",
    "class_weight_scope": "current mini-batch visible labels only",
    "selection_metric": "current_label_validation_mAP",
    "sampling": "uniform shuffled current-task samples",
    "source_static_class_sampling_used": False,
    "fusion_dim": 256,
    "epochs": 21,
    "early_stopping_patience": 21,
    "learning_rate": 0.01,
    "lr_drop_epoch": 7,
    "lr_drop_gamma": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "dropout": 0.5,
    "norm_factor": 1.2,
    "discrete_loss_weight": 1.0 / 6.0,
    "gradient_clip_norm": 10.0,
    "amp": False,
    "tf32": False,
}
TRAINING_PATTERN = re.compile(r"^task=(\d+) training=(\{.*\})$")


def _read_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


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
    records: Dict[int, List[Mapping[str, Any]]] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = TRAINING_PATTERN.match(line)
        if match is None:
            continue
        task_id = int(match.group(1))
        payload = json.loads(match.group(2))
        if not isinstance(payload, Mapping):
            raise ValueError("Training record must be an object")
        records.setdefault(task_id, []).append(payload)
    _require_equal(tuple(sorted(records)), EXPECTED_TASKS, "training task IDs")

    applied_by_task: List[int] = []
    skipped_by_task: List[int] = []
    best_epochs: List[int] = []
    best_current_map: List[float] = []
    for task_id in EXPECTED_TASKS:
        rows = records[task_id]
        _require_equal(len(rows), 21, f"task{task_id} epoch count")
        _require_equal(
            tuple(int(row.get("epoch", -1)) for row in rows),
            tuple(range(21)),
            f"task{task_id} epoch IDs",
        )
        applied = [int(_finite(row.get("optimizer_steps"), "optimizer steps")) for row in rows]
        skipped = [int(_finite(row.get("skipped_optimizer_steps"), "skipped steps")) for row in rows]
        _require_equal(
            tuple(applied),
            (EXPECTED_STEPS_PER_EPOCH[task_id],) * 21,
            f"task{task_id} optimizer steps",
        )
        _require_equal(sum(skipped), 0, f"task{task_id} skipped steps")
        for row in rows:
            _finite(row.get("weighted_classification_loss"), "classification loss")
            _finite(row.get("weighted_sigmoid_mse"), "weighted sigmoid MSE")
            _finite(row.get("validation_current_mAP"), "validation mAP")
        best = max(rows, key=lambda row: float(row["validation_current_mAP"]))
        applied_by_task.append(sum(applied))
        skipped_by_task.append(sum(skipped))
        best_epochs.append(int(best["epoch"]))
        best_current_map.append(float(best["validation_current_mAP"]))

    lowered = text.lower()
    return {
        "epochs_by_task": [21] * 8,
        "optimizer_updates": sum(applied_by_task),
        "optimizer_updates_by_task": applied_by_task,
        "skipped_optimizer_updates": sum(skipped_by_task),
        "skipped_optimizer_updates_by_task": skipped_by_task,
        "best_epoch_by_task": best_epochs,
        "best_current_task_validation_mAP": best_current_map,
        "all_logged_values_finite": True,
        "nan_detected": bool(re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered)),
        "oom_detected": "out of memory" in lowered,
        "traceback_detected": "traceback (most recent call last)" in lowered,
    }


def validate_emot_net_ft_formal_result(
    run_root: Path, run_id: str, expected_git_commit: str
) -> Dict[str, Any]:
    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing formal run root: {root}")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise ValueError("expected-git-commit must be a full lowercase SHA")

    matches = sorted(root.glob("benchmarks/*/B/EMOT-Net-FT/seed0"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one EMOT-Net-FT seed0 bundle, found {len(matches)}")
    seed_root = matches[0]
    manifest = _read_object(seed_root / "run_manifest.json")
    config = _read_object(seed_root / "config_resolved.json")
    summary = _read_object(seed_root / "metrics" / "summary.json")
    task_payload = _read_object(seed_root / "metrics" / "task_metrics.json")

    expected_manifest = {
        "method": "EMOT-Net-FT",
        "method_family": "Static EMOTIC model / Sequential Fine-Tuning",
        "protocol_id": "emotic_b5c3_track_b_v0.1",
        "track": "B",
        "seed": 0,
        "git_commit": expected_git_commit,
        "git_dirty": False,
        "core_runtime_version": "0.10.0",
        "test_labels_used_for_selection": False,
        "checkpoint_selection_split": "val",
        "reporting_split": "test",
        "configuration_locked": True,
        "prediction_reused_task_ids": [],
        "eligible_for_main_table": True,
        "replay_memory_samples": 0,
        "replay_memory_bytes": 0,
        **EXPECTED_PARAMETER_STATISTICS,
    }
    _validate_subset(manifest, expected_manifest, "manifest")
    method_configuration = manifest.get("method_configuration")
    if not isinstance(method_configuration, Mapping):
        raise ValueError("method_configuration must be an object")
    _validate_subset(method_configuration, LOCKED_METHOD_CONFIGURATION, "method")

    runner = config.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("runner configuration must be an object")
    _validate_subset(runner, LOCKED_RUNNER_CONFIGURATION, "runner")

    tasks = task_payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task metrics must be a list")
    _require_equal(len(tasks), 8, "task metric count")
    _require_equal(
        tuple(int(row.get("task_id", -1)) for row in tasks),
        EXPECTED_TASKS,
        "task metric IDs",
    )
    for task_id, row in enumerate(tasks):
        _finite(row.get("mAP"), f"task{task_id} mAP")
        _finite(row.get("cF1"), f"task{task_id} cF1")
        _finite(row.get("oF1"), f"task{task_id} oF1")
        score_path = seed_root / "scores" / f"task{task_id}_scores.pt"
        if not score_path.is_file() or score_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing canonical score payload: {score_path}")

    main_table = summary.get("main_table")
    if not isinstance(main_table, Mapping):
        raise ValueError("main_table must be an object")
    _require_equal(main_table.get("method"), "EMOT-Net-FT", "main-table method")
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
    stability = _training_stability(seed_root / "train.log")
    if any(
        stability[key]
        for key in ("nan_detected", "oom_detected", "traceback_detected")
    ):
        raise ValueError("Formal training log contains an instability marker")

    return {
        "result_schema_version": 1,
        "run_id": run_id,
        "protocol_id": "emotic_b5c3_track_b_v0.1",
        "track": "B",
        "method": "EMOT-Net-FT",
        "formal_seeds": [0],
        "aggregate_statistics": "not_applicable_single_seed",
        "git_commit": expected_git_commit,
        "protocol_hash": manifest.get("protocol_hash"),
        "class_order_hash": manifest.get("class_order_hash"),
        "main_table": metrics,
        "per_task_mAP": [float(row["mAP"]) for row in tasks],
        "per_task_cF1": [float(row["cF1"]) for row in tasks],
        "per_task_oF1": [float(row["oF1"]) for row in tasks],
        "training_stability": stability,
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
    payload = validate_emot_net_ft_formal_result(
        args.run_root, args.run_id, args.expected_git_commit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
