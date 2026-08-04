#!/usr/bin/env python3
"""Validate and aggregate locked AGCN Track-A formal results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


EXPECTED_SEEDS = (0, 1, 2)
LOCK_CONFIRMATION = "AGCN_TRACK_A_V0_1"
AGGREGATE_METRICS = (
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)
LOCKED_METHOD_CONFIGURATION = {
    "strategy": "agcn",
    "upstream_repository": "https://github.com/Kaile-Du/AGCN",
    "upstream_commit": "3afe2ecbbef0051c6e841a97c369885011a683f0",
    "upstream_tree": "6c25689b81d0807523108a782ee59630d25a9b40",
    "upstream_archive_sha256": (
        "b5843d3ee0964767b48c49ba4b71fdf93c4bf954460669ed2b1cd05f9504f301"
    ),
    "upstream_license": "Apache-2.0",
    "upstream_license_note": (
        "root LICENSE is Apache-2.0; GCN.py contains a bare BSD header comment"
    ),
    "visual_encoder_trainable": True,
    "clip_text_encoder_used": False,
    "benchmark_added_adapter": False,
    "replay_enabled": False,
    "old_future_ground_truth_used_for_training": False,
    "backbone_substitution": (
        "ImageNet-pretrained ResNet-101 -> OpenAI CLIP ViT-B/16 visual encoder"
    ),
    "retained_agcn_components": [
        "glove_6b_300d_label_nodes",
        "two_layer_300_1024_visualdim_gcn",
        "online_augmented_correlation_matrix",
        "old_model_sigmoid_distillation",
        "old_graph_node_relationship_mse",
        "one_epoch_per_task",
    ],
    "acm_old_soft_label_activation": "softmax_as_released_code",
    "distillation_activation": "sigmoid_as_released_code",
    "loss_weight_resolution": "paper_table_3_fills_undefined_released_a_b_c",
    "adam_epsilon_resolution": "1e-8_released_torch_default_paper_reports_1e-4",
    "selection_policy": "fixed_one_epoch_validation_monitoring_only",
    "visual_dim": 512,
    "embedding_dim": 300,
    "graph_hidden_dim": 1024,
    "epochs": 1,
    "visual_learning_rate": 1.0e-4,
    "graph_learning_rate": 3.0e-5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1.0e-8,
    "classification_weight": 0.07,
    "distillation_weight": 0.93,
    "relationship_weight": 1.0e5,
    "task0_threshold": 0.0,
    "current_threshold": 0.4,
    "cross_threshold": 0.3,
    "task0_graph_scale": 0.28,
    "later_graph_scale": 0.25,
    "reverse_bayes_scale": 0.5,
    "task0_degree_exponent": -0.8,
    "later_degree_exponent": -0.5,
    "amp": False,
    "tf32": False,
}
LOCKED_EMBEDDING_ASSET = {
    "kind": "external_glove_6b_300d_mapping",
    "sha256": "e926b8672cd761169586668c54e4676093b152e2b02a218747f55bf72e1d1bee",
    "source_sha256": "0a7aebbe49097dc6e5ffff7a25e9aa20181a6e862050ca68bd2f36e056739e00",
    "source_md5": "29e9329ac2241937d55b852e8284e89b",
    "source_distribution": (
        "gensim-data glove-wiki-gigaword-300; converted from Stanford GloVe "
        "text to word2vec text and gzip-compressed"
    ),
}
LOCKED_RUNNER_CONFIGURATION = {
    "train_batch_size": 8,
    "eval_batch_size": 32,
    "num_workers": 0,
    "reporting_split": "test",
    "configuration_locked": True,
    "execution_mode": "sequential",
    "core_runtime_version": "0.8.0",
}
EXPECTED_PARAMETER_STATISTICS = {
    "total_parameters": 87024128,
    "trainable_parameters": 87024128,
    "incremental_parameters": 0,
    "per_task_incremental_parameters": {str(task): 0 for task in range(8)},
}
EXPECTED_OPTIMIZER_STEPS = (670, 550, 108, 1242, 544, 317, 79, 191)
EXPECTED_TRAINING_SAMPLES = (5353, 4394, 861, 9931, 4352, 2536, 627, 1526)
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

    optimizer_steps: List[int] = []
    skipped_steps: List[int] = []
    training_samples: List[int] = []
    current_loss: List[float] = []
    distillation_loss: List[float] = []
    relationship_loss: List[float] = []
    total_loss: List[float] = []
    validation_map: List[float] = []
    adjacency_nonzero: List[int] = []
    adjacency_density: List[float] = []
    current_positive_mass: List[float] = []
    old_soft_mass: List[float] = []
    cross_soft_hard_mass: List[float] = []

    for task_id in range(8):
        task_records = records[task_id]
        _require_equal(len(task_records), 1, f"task{task_id} training record count")
        row = task_records[0]
        _require_equal(int(row.get("epoch", -1)), 0, f"task{task_id} epoch")
        applied = int(_finite_number(row.get("optimizer_steps"), "optimizer steps"))
        skipped = int(
            _finite_number(row.get("skipped_optimizer_steps"), "skipped steps")
        )
        samples = int(_finite_number(row.get("acm_samples"), "ACM samples"))
        _require_equal(applied, EXPECTED_OPTIMIZER_STEPS[task_id], f"task{task_id} optimizer steps")
        _require_equal(skipped, 0, f"task{task_id} skipped steps")
        _require_equal(samples, EXPECTED_TRAINING_SAMPLES[task_id], f"task{task_id} ACM samples")

        current = _finite_number(row.get("current_loss"), "current loss")
        distill = _finite_number(row.get("distillation_loss"), "distillation loss")
        relation = _finite_number(row.get("relationship_loss"), "relationship loss")
        total = _finite_number(row.get("total_loss"), "total loss")
        density = _finite_number(row.get("adjacency_density"), "adjacency density")
        nonzero = int(_finite_number(row.get("adjacency_nonzero"), "adjacency nonzero"))
        current_mass = _finite_number(
            row.get("acm_current_positive_mass"), "current positive mass"
        )
        old_mass = _finite_number(row.get("acm_old_soft_mass"), "old soft mass")
        cross_mass = _finite_number(
            row.get("acm_cross_soft_hard_mass"), "cross soft-hard mass"
        )
        if current <= 0 or total <= 0 or current_mass <= 0:
            raise ValueError(f"task{task_id} has invalid positive training values")
        if not (0 < density <= 1) or nonzero <= 0:
            raise ValueError(f"task{task_id} has invalid graph statistics")
        if task_id == 0:
            _require_equal(distill, 0.0, "task0 distillation loss")
            _require_equal(relation, 0.0, "task0 relationship loss")
            _require_equal(old_mass, 0.0, "task0 old soft mass")
            _require_equal(cross_mass, 0.0, "task0 cross mass")
        elif min(distill, relation, old_mass, cross_mass) <= 0:
            raise ValueError(f"task{task_id} teacher/ACM statistics must be positive")

        optimizer_steps.append(applied)
        skipped_steps.append(skipped)
        training_samples.append(samples)
        current_loss.append(current)
        distillation_loss.append(distill)
        relationship_loss.append(relation)
        total_loss.append(total)
        validation_map.append(
            _finite_number(row.get("validation_current_mAP"), "validation mAP")
        )
        adjacency_nonzero.append(nonzero)
        adjacency_density.append(density)
        current_positive_mass.append(current_mass)
        old_soft_mass.append(old_mass)
        cross_soft_hard_mass.append(cross_mass)

    lowered = text.lower()
    return {
        "optimizer_updates": sum(optimizer_steps),
        "optimizer_updates_by_task": optimizer_steps,
        "skipped_optimizer_updates": sum(skipped_steps),
        "skipped_optimizer_updates_by_task": skipped_steps,
        "training_samples_by_task": training_samples,
        "current_loss_by_task": current_loss,
        "distillation_loss_by_task": distillation_loss,
        "relationship_loss_by_task": relationship_loss,
        "total_loss_by_task": total_loss,
        "current_task_validation_mAP": validation_map,
        "adjacency_nonzero_by_task": adjacency_nonzero,
        "adjacency_density_by_task": adjacency_density,
        "current_positive_mass_by_task": current_positive_mass,
        "old_soft_mass_by_task": old_soft_mass,
        "cross_soft_hard_mass_by_task": cross_soft_hard_mass,
        "all_logged_values_finite": True,
        "nan_detected": bool(re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered)),
        "oom_detected": "out of memory" in lowered,
        "traceback_detected": "traceback (most recent call last)" in lowered,
    }


def validate_agcn_formal_results(
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
            f"benchmarks/*/A/AGCN/seed{seed}",
            f"AGCN seed {seed} artifact directory",
        )
        manifest = _read_object(seed_root / "run_manifest.json")
        config = _read_object(seed_root / "config_resolved.json")
        summary = _read_object(seed_root / "metrics" / "summary.json")
        task_payload = _read_object(seed_root / "metrics" / "task_metrics.json")

        expected_manifest = {
            "method": "AGCN",
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": expected_git_commit,
            "git_dirty": False,
            "core_runtime_version": "0.8.0",
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
        embedding_asset = method_configuration.get("class_embedding_asset")
        if not isinstance(embedding_asset, Mapping):
            raise ValueError(f"seed{seed} embedding asset must be an object")
        _validate_locked_subset(
            embedding_asset, LOCKED_EMBEDDING_ASSET, f"seed{seed} embedding asset"
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
        _require_equal(protocol_configuration.get("seed"), seed, f"seed{seed} protocol seed")
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
        score_tasks = sorted(
            int(match.group(1))
            for path in (seed_root / "scores").glob("task*_scores.pt")
            if (match := re.fullmatch(r"task(\d+)_scores\.pt", path.name))
        )
        _require_equal(score_tasks, list(range(8)), f"seed{seed} score tasks")

        current_provenance = {
            "git_commit": manifest.get("git_commit"),
            "source_tree_hash": manifest.get("source_tree_hash"),
            "upstream_commit": method_configuration.get("upstream_commit"),
            "upstream_tree": method_configuration.get("upstream_tree"),
            "upstream_archive_sha256": method_configuration.get(
                "upstream_archive_sha256"
            ),
            "embedding_asset_sha256": embedding_asset.get("sha256"),
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
        "method": "AGCN",
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
    payload = validate_agcn_formal_results(
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
