"""Protocol-driven orchestration without method-specific task constants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .artifacts import ArtifactStore, SHARD_SCHEMA_VERSION
from .data_module import EMOTICMLCILDataModule
from .evaluator import BenchmarkEvaluator
from .method_base import BenchmarkMethod
from .methods.ddp import DDPBenchmarkMethod
from .methods.clip_classifier import (
    ElasticWeightConsolidationMethod,
    LearningWithoutForgettingMethod,
    SequentialFineTuningMethod,
)
from .protocol import BenchmarkProtocol
from .protocol import load_protocol
from .types import (
    BenchmarkSummary,
    MemoryStatistics,
    ParameterStatistics,
    PredictionOutput,
    TaskContext,
    TaskMetrics,
)


BASE_COMMIT = "f9459d0769f4ef3ee93e51db31df6ec509a933ad"
CORE_BASE_COMMIT = "00f399f13bc7552c254c8f6e6c095a8be4f56146"
CORE_RUNTIME_VERSION = "0.3.1"


def _current_git_commit() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else BASE_COMMIT


def _current_source_state() -> Dict[str, Any]:
    """Record the exact Core source tree even before it receives a commit."""

    repository_root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    source_paths = [
        repository_root / "requirements.txt",
        repository_root / "benchmarks" / "emotic_mlcil",
        repository_root / "configs" / "emotic_mlcil",
        repository_root / "docs" / "benchmarks",
        repository_root / "scripts" / "emotic-mlcil",
        repository_root / "tests" / "emotic_mlcil",
    ]
    files: List[Path] = []
    for path in source_paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and "__pycache__" not in candidate.parts
            )
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(repository_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "git_commit": _current_git_commit(),
        # A synchronized server mirror without .git is intentionally treated
        # as non-clean. The content hash below remains the exact provenance.
        "git_dirty": (
            status.returncode != 0 or bool(status.stdout.strip())
        ),
        "source_tree_hash": digest.hexdigest(),
    }


def _validate_reporting_split(
    protocol: BenchmarkProtocol,
    reporting_split: str,
    configuration_locked: bool,
) -> None:
    if reporting_split == protocol.test_split:
        if not configuration_locked:
            raise RuntimeError(
                "Test evaluation requires lock_configuration() after all "
                "configuration and checkpoint-selection decisions are frozen"
            )
    elif reporting_split != protocol.validation_split:
        raise ValueError("reporting_split must be protocol validation or test")


def _method_metadata(method: BenchmarkMethod) -> Dict[str, Any]:
    return {
        "name": method.method_name,
        "family": method.method_family,
        "backbone": method.backbone,
        "upstream_repository": getattr(
            method, "upstream_repository", "TBD"
        ),
        "upstream_commit": getattr(method, "upstream_commit", "TBD"),
        "resolved_method_config": dict(method.resolved_method_config()),
    }


def _finalize_run(
    *,
    protocol: BenchmarkProtocol,
    artifact_store: ArtifactStore,
    task_rows: Sequence[TaskMetrics],
    parameter_stats: ParameterStatistics,
    memory_stats: MemoryStatistics,
    method_metadata: Mapping[str, Any],
    reporting_split: str,
    configuration_locked: bool,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    git_commit: str,
    git_dirty: bool,
    source_tree_hash: str,
    shard_run_id: Optional[str] = None,
    prediction_reused_task_ids: Sequence[int] = (),
) -> BenchmarkSummary:
    evaluator = BenchmarkEvaluator(protocol)
    summary = evaluator.summarize(task_rows)
    data_split_hash = {
        f"task{row.task_id}": row.split_hash for row in task_rows
    }
    manifest: Dict[str, Any] = {
        "method": method_metadata["name"],
        "method_family": method_metadata["family"],
        "protocol_id": protocol.protocol_id,
        "track": protocol.track,
        "seed": protocol.seed,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_tree_hash": source_tree_hash,
        "base_commit": BASE_COMMIT,
        "core_base_commit": CORE_BASE_COMMIT,
        "core_runtime_version": CORE_RUNTIME_VERSION,
        "backbone": method_metadata["backbone"],
        "class_order_hash": protocol.class_order_hash,
        "protocol_hash": protocol.protocol_hash,
        "data_split_hash": data_split_hash,
        "threshold_policy": {
            "kind": "fixed_global",
            "value": protocol.threshold,
            "per_task_selection": False,
            "per_class_selection": False,
        },
        **memory_stats.as_dict(),
        "total_parameters": parameter_stats.total_parameters,
        "trainable_parameters": parameter_stats.trainable_parameters,
        "incremental_parameters": parameter_stats.incremental_parameters,
        "per_task_incremental_parameters": dict(
            parameter_stats.per_task_incremental_parameters
        ),
        "upstream_repository": method_metadata["upstream_repository"],
        "upstream_commit": method_metadata["upstream_commit"],
        "method_configuration": dict(
            method_metadata.get("resolved_method_config", {})
        ),
        "test_labels_used_for_selection": False,
        "checkpoint_selection_split": protocol.validation_split,
        "reporting_split": reporting_split,
        "configuration_locked": configuration_locked,
        "prediction_reused_task_ids": [
            int(task_id) for task_id in prediction_reused_task_ids
        ],
        "eligible_for_main_table": (
            reporting_split == protocol.test_split
            and configuration_locked
            and not git_dirty
        ),
    }
    if shard_run_id is not None:
        manifest["shard_run_id"] = shard_run_id
    resolved_config: Dict[str, Any] = {
        "protocol": protocol.as_dict(),
        "runner": {
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "reporting_split": reporting_split,
            "configuration_locked": configuration_locked,
            "execution_mode": (
                "parallel_task_shards"
                if shard_run_id is not None
                else "sequential"
            ),
            "core_runtime_version": CORE_RUNTIME_VERSION,
        },
        "method": {
            "name": method_metadata["name"],
            "family": method_metadata["family"],
            "backbone": method_metadata["backbone"],
            "options": dict(
                method_metadata.get("resolved_method_config", {})
            ),
        },
    }
    if shard_run_id is not None:
        resolved_config["runner"]["shard_run_id"] = shard_run_id
        resolved_config["runner"]["prediction_reused_task_ids"] = [
            int(task_id) for task_id in prediction_reused_task_ids
        ]
    artifact_store.initialize(resolved_config, manifest)
    artifact_store.write_task_metrics(task_rows)
    artifact_store.write_summary(
        summary,
        method_family=method_metadata["family"],
        backbone=method_metadata["backbone"],
        replay_memory_samples=memory_stats.replay_memory_samples,
        replay_memory_bytes=memory_stats.replay_memory_bytes,
        parameter_growth=parameter_stats.incremental_parameters,
    )
    return summary


class BenchmarkRunner:
    def __init__(
        self,
        protocol: BenchmarkProtocol,
        data_module: EMOTICMLCILDataModule,
        method: BenchmarkMethod,
        artifact_store: ArtifactStore,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        if train_batch_size <= 0 or eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if protocol.track not in method.supported_tracks:
            raise ValueError(
                f"{method.method_name} does not support Track {protocol.track}"
            )
        self.protocol = protocol
        self.data_module = data_module
        self.method = method
        self.artifact_store = artifact_store
        self.evaluator = BenchmarkEvaluator(protocol)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self._configuration_locked = False

    def lock_configuration(self) -> None:
        """Irreversibly authorize one held-out test evaluation for this runner."""

        self._configuration_locked = True

    def task_context(self, task_id: int) -> TaskContext:
        current = self.protocol.current_class_indices(task_id)
        seen = self.protocol.seen_class_indices(task_id)
        future = self.protocol.future_class_indices(task_id)
        return TaskContext(
            task_id=task_id,
            protocol_id=self.protocol.protocol_id,
            class_order=self.protocol.class_order,
            class_order_hash=self.protocol.class_order_hash,
            protocol_hash=self.protocol.protocol_hash,
            current_class_indices=current,
            seen_class_indices=seen,
            future_class_indices=future,
            current_class_names=tuple(
                self.protocol.class_order[index] for index in current
            ),
            seen_class_names=tuple(
                self.protocol.class_order[index] for index in seen
            ),
            future_class_names=tuple(
                self.protocol.class_order[index] for index in future
            ),
            seed=self.protocol.seed,
            track=self.protocol.track,
        )

    def run(self, reporting_split: str = "val") -> BenchmarkSummary:
        _validate_reporting_split(
            self.protocol,
            reporting_split,
            self._configuration_locked,
        )

        evaluator_datasets = {
            task_id: self.data_module.evaluator_dataset(
                task_id,
                reporting_split,
                self.evaluator.access_token,
            )
            for task_id in range(self.protocol.num_tasks)
        }
        data_split_hash = {
            f"task{task_id}": dataset.split_hash
            for task_id, dataset in evaluator_datasets.items()
        }
        task_rows: List[TaskMetrics] = []
        last_parameter_stats = None
        last_memory_stats = None

        for task_id in range(self.protocol.num_tasks):
            context = self.task_context(task_id)
            self.method.begin_task(context)
            try:
                train_loader = self.data_module.method_loader(
                    task_id,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers,
                    split=self.protocol.train_split,
                )
                selection_loader = self.data_module.method_loader(
                    task_id,
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                    split=self.protocol.validation_split,
                    shuffle=False,
                )
                self.method.train_task(train_loader, selection_loader)
                for record in self.method.training_log_records():
                    self.artifact_store.append_log(
                        f"task={task_id} training="
                        + json.dumps(
                            dict(record),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )

                evaluation_loader = self.data_module.evaluator_loader(
                    task_id,
                    reporting_split,
                    self.evaluator.access_token,
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                )
                prediction = self.method.predict_scores(evaluation_loader)
                expected_dataset = evaluator_datasets[task_id]
                row = self.evaluator.evaluate_task(
                    task_id,
                    prediction,
                    expected_sample_ids=expected_dataset.sample_ids,
                    expected_split_hash=expected_dataset.split_hash,
                )
                task_rows.append(row)
                self.artifact_store.save_scores(task_id, prediction)
                self.artifact_store.write_task_metrics(task_rows)
                self.method.save_checkpoint(
                    self.artifact_store.checkpoint_dir / f"task{task_id}.pth"
                )
                last_parameter_stats = self.method.parameter_statistics()
                last_memory_stats = self.method.memory_statistics()
                self.artifact_store.append_log(
                    f"task={task_id} split={reporting_split} "
                    f"samples={row.samples} mAP={row.mAP:.6f} "
                    f"cF1={row.cF1:.6f} oF1={row.oF1:.6f}"
                )
            finally:
                self.method.end_task()

        if last_parameter_stats is None or last_memory_stats is None:
            raise RuntimeError("Benchmark produced no task statistics")
        if data_split_hash != {
            f"task{row.task_id}": row.split_hash for row in task_rows
        }:
            raise RuntimeError("Evaluator split hashes changed during the run")
        source_state = _current_source_state()
        return _finalize_run(
            protocol=self.protocol,
            artifact_store=self.artifact_store,
            task_rows=task_rows,
            parameter_stats=last_parameter_stats,
            memory_stats=last_memory_stats,
            method_metadata=_method_metadata(self.method),
            reporting_split=reporting_split,
            configuration_locked=self._configuration_locked,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            git_commit=source_state["git_commit"],
            git_dirty=source_state["git_dirty"],
            source_tree_hash=source_state["source_tree_hash"],
        )

    def run_task_shard(
        self,
        task_id: int,
        shard_run_id: str,
        reporting_split: str = "val",
    ) -> TaskMetrics:
        """Evaluate exactly one task into a concurrency-safe shard directory."""

        self.protocol.current_class_indices(task_id)
        self.artifact_store.validate_shard_run_id(shard_run_id)
        _validate_reporting_split(
            self.protocol,
            reporting_split,
            self._configuration_locked,
        )
        expected_dataset = self.data_module.evaluator_dataset(
            task_id,
            reporting_split,
            self.evaluator.access_token,
        )
        context = self.task_context(task_id)
        self.method.begin_task(context)
        try:
            shard_score_path = (
                self.artifact_store.task_shard_dir(shard_run_id, task_id)
                / "scores.pt"
            )
            prediction_reused = shard_score_path.is_file()
            if prediction_reused:
                prediction = self.artifact_store.load_shard_scores(
                    shard_run_id,
                    task_id,
                )
            else:
                train_loader = self.data_module.method_loader(
                    task_id,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers,
                    split=self.protocol.train_split,
                )
                selection_loader = self.data_module.method_loader(
                    task_id,
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                    split=self.protocol.validation_split,
                    shuffle=False,
                )
                self.method.train_task(train_loader, selection_loader)
                evaluation_loader = self.data_module.evaluator_loader(
                    task_id,
                    reporting_split,
                    self.evaluator.access_token,
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                )
                prediction = self.method.predict_scores(evaluation_loader)
            row = self.evaluator.evaluate_task(
                task_id,
                prediction,
                expected_sample_ids=expected_dataset.sample_ids,
                expected_split_hash=expected_dataset.split_hash,
            )
            if not prediction_reused:
                self.artifact_store.save_shard_scores(
                    shard_run_id,
                    task_id,
                    prediction,
                )
            checkpoint_path = self.artifact_store.shard_checkpoint_path(
                shard_run_id,
                task_id,
            )
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
                self.method.save_checkpoint(checkpoint_path)
            parameter_stats = self.method.parameter_statistics()
            memory_stats = self.method.memory_statistics()
            log_line = (
                f"task={task_id} split={reporting_split} "
                f"samples={row.samples} mAP={row.mAP:.6f} "
                f"cF1={row.cF1:.6f} oF1={row.oF1:.6f} "
                f"prediction_reused={str(prediction_reused).lower()}"
            )
            self.artifact_store.append_shard_log(
                shard_run_id,
                task_id,
                log_line,
            )
            source_state = _current_source_state()
            metadata = {
                "output_schema_version": self.protocol.output_schema_version,
                "protocol_id": self.protocol.protocol_id,
                "protocol_hash": self.protocol.protocol_hash,
                "class_order_hash": self.protocol.class_order_hash,
                "track": self.protocol.track,
                "seed": self.protocol.seed,
                "reporting_split": reporting_split,
                "configuration_locked": self._configuration_locked,
                **source_state,
                "base_commit": BASE_COMMIT,
                "core_base_commit": CORE_BASE_COMMIT,
                "core_runtime_version": CORE_RUNTIME_VERSION,
                "method": _method_metadata(self.method),
                "runner": {
                    "train_batch_size": self.train_batch_size,
                    "eval_batch_size": self.eval_batch_size,
                    "num_workers": self.num_workers,
                },
                "task_metrics": row.as_dict(),
                "parameter_statistics": parameter_stats.as_dict(),
                "memory_statistics": memory_stats.as_dict(),
                "prediction_reused_from_shard": prediction_reused,
                "score_file": "scores.pt",
                "checkpoint_file": "checkpoint.pth",
            }
            # This atomic metadata write is the shard-completion marker.
            self.artifact_store.write_task_shard_metadata(
                shard_run_id,
                task_id,
                metadata,
            )
            return row
        finally:
            self.method.end_task()


def _validate_shard_score_payload(
    protocol: BenchmarkProtocol,
    task_id: int,
    row: TaskMetrics,
    score_path: Path,
) -> None:
    if not score_path.is_file():
        raise FileNotFoundError(f"Missing task shard scores: {score_path}")
    payload = torch.load(score_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Shard score payload must be a mapping: {score_path}")
    scores = payload.get("scores")
    targets = payload.get("targets")
    sample_ids = payload.get("sample_ids")
    expected_shape = (
        row.samples,
        len(protocol.seen_class_indices(task_id)),
    )
    if not isinstance(scores, torch.Tensor) or not isinstance(
        targets, torch.Tensor
    ):
        raise ValueError(f"Shard scores/targets must be tensors: {score_path}")
    if tuple(scores.shape) != expected_shape or tuple(targets.shape) != expected_shape:
        raise ValueError(
            f"Task {task_id} shard tensors must have shape {expected_shape}"
        )
    if not isinstance(sample_ids, list) or len(sample_ids) != row.samples:
        raise ValueError(f"Task {task_id} shard sample IDs are incomplete")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Task {task_id} shard sample IDs are not unique")
    if int(payload.get("task_id", -1)) != task_id:
        raise ValueError(f"Task {task_id} shard score task_id differs")
    if payload.get("class_order_hash") != protocol.class_order_hash:
        raise ValueError(f"Task {task_id} shard class-order hash differs")
    if payload.get("split_hash") != row.split_hash:
        raise ValueError(f"Task {task_id} shard split hash differs")
    if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
        raise ValueError(f"Task {task_id} shard contains NaN or Inf")
    if not bool(((scores >= 0) & (scores <= 1)).all()):
        raise ValueError(f"Task {task_id} shard scores are outside [0, 1]")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError(f"Task {task_id} shard targets are not binary")
    recomputed = BenchmarkEvaluator(protocol).evaluate_task(
        task_id,
        PredictionOutput(
            scores=scores,
            targets=targets,
            sample_ids=sample_ids,
            class_order_hash=str(payload["class_order_hash"]),
            split_hash=str(payload["split_hash"]),
        ),
        expected_sample_ids=sample_ids,
        expected_split_hash=row.split_hash,
    )
    if recomputed.as_dict() != row.as_dict():
        raise ValueError(
            f"Task {task_id} shard metrics do not match saved scores"
        )


def merge_task_shards(
    *,
    protocol: BenchmarkProtocol,
    artifact_store: ArtifactStore,
    shard_run_id: str,
    reporting_split: str,
    configuration_locked: bool,
) -> BenchmarkSummary:
    """Validate all task shards and publish one canonical benchmark result."""

    artifact_store.validate_shard_run_id(shard_run_id)
    _validate_reporting_split(
        protocol,
        reporting_split,
        configuration_locked,
    )
    task_rows: List[TaskMetrics] = []
    shard_metadata: List[Dict[str, Any]] = []
    expected_method: Optional[Mapping[str, str]] = None
    expected_runner: Optional[Mapping[str, Any]] = None
    expected_git_commit: Optional[str] = None
    expected_git_dirty: Optional[bool] = None
    expected_source_tree_hash: Optional[str] = None

    for task_id in range(protocol.num_tasks):
        metadata = artifact_store.load_task_shard_metadata(
            shard_run_id,
            task_id,
        )
        if metadata.get("shard_schema_version") != SHARD_SCHEMA_VERSION:
            raise ValueError(f"Task {task_id} shard schema version differs")
        expected_fields = {
            "shard_run_id": shard_run_id,
            "task_id": task_id,
            "output_schema_version": protocol.output_schema_version,
            "protocol_id": protocol.protocol_id,
            "protocol_hash": protocol.protocol_hash,
            "class_order_hash": protocol.class_order_hash,
            "track": protocol.track,
            "seed": protocol.seed,
            "reporting_split": reporting_split,
            "configuration_locked": configuration_locked,
            "base_commit": BASE_COMMIT,
            "core_base_commit": CORE_BASE_COMMIT,
            "core_runtime_version": CORE_RUNTIME_VERSION,
            "score_file": "scores.pt",
            "checkpoint_file": "checkpoint.pth",
        }
        for key, expected in expected_fields.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Task {task_id} shard {key} differs: "
                    f"{metadata.get(key)!r} != {expected!r}"
                )
        method = metadata.get("method")
        runner = metadata.get("runner")
        git_commit = metadata.get("git_commit")
        git_dirty = metadata.get("git_dirty")
        source_tree_hash = metadata.get("source_tree_hash")
        if not isinstance(method, Mapping):
            raise ValueError(f"Task {task_id} shard method must be a mapping")
        if not isinstance(runner, Mapping):
            raise ValueError(f"Task {task_id} shard runner must be a mapping")
        if not isinstance(git_commit, str) or not git_commit:
            raise ValueError(f"Task {task_id} shard git_commit is invalid")
        if not isinstance(git_dirty, bool):
            raise ValueError(f"Task {task_id} shard git_dirty is invalid")
        if not isinstance(source_tree_hash, str) or len(source_tree_hash) != 64:
            raise ValueError(
                f"Task {task_id} shard source_tree_hash is invalid"
            )
        if expected_method is None:
            expected_method = dict(method)
            expected_runner = dict(runner)
            expected_git_commit = git_commit
            expected_git_dirty = git_dirty
            expected_source_tree_hash = source_tree_hash
        elif (
            dict(method) != dict(expected_method)
            or dict(runner) != dict(expected_runner)
            or git_commit != expected_git_commit
            or git_dirty != expected_git_dirty
            or source_tree_hash != expected_source_tree_hash
        ):
            raise ValueError(
                f"Task {task_id} shard method, runner, or source state differs"
            )
        row_payload = metadata.get("task_metrics")
        if not isinstance(row_payload, Mapping):
            raise ValueError(
                f"Task {task_id} shard task_metrics must be a mapping"
            )
        row = TaskMetrics.from_dict(row_payload)
        if row.task_id != task_id:
            raise ValueError(f"Task {task_id} shard metric task_id differs")
        if row.class_order_hash != protocol.class_order_hash:
            raise ValueError(f"Task {task_id} shard metric class hash differs")
        shard_directory = artifact_store.task_shard_dir(
            shard_run_id,
            task_id,
        )
        _validate_shard_score_payload(
            protocol,
            task_id,
            row,
            shard_directory / "scores.pt",
        )
        checkpoint_path = shard_directory / "checkpoint.pth"
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Task {task_id} shard checkpoint is missing or empty"
            )
        task_rows.append(row)
        shard_metadata.append(metadata)

    if (
        expected_method is None
        or expected_runner is None
        or expected_git_commit is None
        or expected_git_dirty is None
        or expected_source_tree_hash is None
    ):
        raise RuntimeError("No task shards were loaded")
    BenchmarkEvaluator(protocol).summarize(task_rows)
    last_metadata = shard_metadata[-1]
    parameter_payload = last_metadata.get("parameter_statistics")
    memory_payload = last_metadata.get("memory_statistics")
    if not isinstance(parameter_payload, Mapping) or not isinstance(
        memory_payload, Mapping
    ):
        raise ValueError("Final task shard statistics are missing")
    parameter_stats = ParameterStatistics.from_dict(parameter_payload)
    memory_stats = MemoryStatistics.from_dict(memory_payload)
    try:
        train_batch_size = int(expected_runner["train_batch_size"])
        eval_batch_size = int(expected_runner["eval_batch_size"])
        num_workers = int(expected_runner["num_workers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Shard runner settings are incomplete") from exc
    if train_batch_size <= 0 or eval_batch_size <= 0 or num_workers < 0:
        raise ValueError("Shard runner settings are invalid")

    for task_id in range(protocol.num_tasks):
        artifact_store.publish_task_shard(shard_run_id, task_id)
    artifact_store.merge_shard_logs(
        shard_run_id,
        list(range(protocol.num_tasks)),
    )
    return _finalize_run(
        protocol=protocol,
        artifact_store=artifact_store,
        task_rows=task_rows,
        parameter_stats=parameter_stats,
        memory_stats=memory_stats,
        method_metadata=expected_method,
        reporting_split=reporting_split,
        configuration_locked=configuration_locked,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        git_commit=expected_git_commit,
        git_dirty=expected_git_dirty,
        source_tree_hash=expected_source_tree_hash,
        shard_run_id=shard_run_id,
        prediction_reused_task_ids=[
            int(metadata["task_id"])
            for metadata in shard_metadata
            if metadata.get("prediction_reused_from_shard") is True
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the protocol-driven EMOTIC MLCIL benchmark"
    )
    parser.add_argument(
        "--protocol",
        default="configs/emotic_mlcil/protocol_b5c3.yaml",
    )
    parser.add_argument(
        "--method",
        choices=("ddp", "finetune", "lwf", "ewc"),
        default="ddp",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument(
        "--clip-model-path",
        default="./pretrained/clip/ViT-B-16.pt",
    )
    parser.add_argument("--output-root", default="./output")
    parser.add_argument("--reporting-split", default="val")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override only the registered protocol seed (for multi-seed runs)",
    )
    parser.add_argument(
        "--configuration-locked",
        action="store_true",
        help="Required assertion before held-out test evaluation",
    )
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        help=(
            "Runtime EWC coefficient override. This is method metadata, not "
            "part of the frozen dataset protocol hash."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=("full", "person_crop"),
        default="full",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--task-id",
        type=int,
        help="Evaluate one task into an isolated parallel shard",
    )
    execution.add_argument(
        "--merge-shards",
        action="store_true",
        help="Validate and publish all completed task shards without a GPU",
    )
    execution.add_argument(
        "--export-sync-results",
        action="store_true",
        help="Create one checkpoint-free folder for downloading a completed run",
    )
    parser.add_argument(
        "--shard-run-id",
        help="Shared safe identifier for --task-id/--merge-shards",
    )
    return parser.parse_args()


def _legacy_emotic_transforms():
    import torchvision.transforms as transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.05, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def main() -> None:
    args = _parse_args()
    protocol = load_protocol(args.protocol)
    if args.seed is not None:
        protocol = protocol.with_seed(args.seed)
    if args.reporting_split not in {
        protocol.validation_split,
        protocol.test_split,
    }:
        raise ValueError(
            "--reporting-split must match the loaded protocol's val or test split"
        )
    uses_shards = (
        args.task_id is not None
        or args.merge_shards
        or args.export_sync_results
    )
    if uses_shards and not args.shard_run_id:
        raise ValueError("--shard-run-id is required for shard execution")
    if not uses_shards and args.shard_run_id:
        raise ValueError(
            "--shard-run-id requires --task-id, --merge-shards, or "
            "--export-sync-results"
        )
    if args.method != "ddp" and (
        args.task_id is not None or args.merge_shards
    ):
        raise ValueError(
            "Continual training depends on prior tasks and cannot use "
            "independent task shards"
        )
    if args.ewc_lambda is not None:
        if args.method != "ewc":
            raise ValueError("--ewc-lambda is valid only with --method ewc")
        if not math.isfinite(args.ewc_lambda) or args.ewc_lambda <= 0:
            raise ValueError("--ewc-lambda must be finite and positive")
    method_classes = {
        "ddp": DDPBenchmarkMethod,
        "finetune": SequentialFineTuningMethod,
        "lwf": LearningWithoutForgettingMethod,
        "ewc": ElasticWeightConsolidationMethod,
    }
    method_class = method_classes[args.method]
    artifacts = ArtifactStore(
        args.output_root,
        protocol,
        track=protocol.track,
        method_name=method_class.method_name,
        seed=protocol.seed,
    )
    if args.export_sync_results:
        destination = artifacts.export_sync_results(args.shard_run_id)
        print(
            json.dumps(
                {
                    "shard_run_id": args.shard_run_id,
                    "results_to_sync": str(destination),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.merge_shards:
        summary = merge_task_shards(
            protocol=protocol,
            artifact_store=artifacts,
            shard_run_id=args.shard_run_id,
            reporting_split=args.reporting_split,
            configuration_locked=args.configuration_locked,
        )
        print(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))
        return
    if not args.data_root:
        raise ValueError("--data-root is required for benchmark execution")
    train_transform, eval_transform = _legacy_emotic_transforms()
    data_module = EMOTICMLCILDataModule(
        protocol,
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        input_mode=args.input_mode,
    )
    if args.method == "ddp":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required for DDP evaluation")
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_paths = {
            task_id: checkpoint_dir / f"task{task_id}.pth"
            for task_id in range(protocol.num_tasks)
        }
        method = DDPBenchmarkMethod(
            protocol,
            checkpoint_paths=checkpoint_paths,
            clip_model_path=args.clip_model_path,
            device=args.device,
        )
    else:
        option_overrides = (
            {"ewc_lambda": args.ewc_lambda}
            if args.ewc_lambda is not None
            else None
        )
        method = method_class(
            protocol,
            clip_model_path=args.clip_model_path,
            device=args.device,
            option_overrides=option_overrides,
        )
    runner = BenchmarkRunner(
        protocol,
        data_module,
        method,
        artifacts,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.workers,
    )
    if args.configuration_locked:
        runner.lock_configuration()
    if args.task_id is not None:
        row = runner.run_task_shard(
            task_id=args.task_id,
            shard_run_id=args.shard_run_id,
            reporting_split=args.reporting_split,
        )
        print(
            json.dumps(
                {
                    "shard_run_id": args.shard_run_id,
                    "task_metrics": row.as_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        summary = runner.run(reporting_split=args.reporting_split)
        print(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
