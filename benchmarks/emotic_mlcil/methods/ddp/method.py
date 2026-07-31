"""Thin adapter around the repository's existing DDP implementation."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

import torch

from ...method_base import BenchmarkMethod
from ...protocol import BenchmarkProtocol
from ...registry import register_method
from ...types import (
    EvaluationBatch,
    MemoryStatistics,
    ParameterStatistics,
    PredictionOutput,
    TaskContext,
)


def _unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _legacy_optimizer_parameters(
    model: torch.nn.Module,
) -> Sequence[torch.nn.Parameter]:
    """Return the exact tensors registered by ``DDP.build_optimizer_scheduler``."""

    try:
        named_parameters = (
            ("prompt_learner.ctx_pos", model.prompt_learner.ctx_pos),
            ("prompt_learner.ctx_neg", model.prompt_learner.ctx_neg),
            ("visual_prompts", model.visual_prompts),
        )
    except AttributeError as error:
        raise RuntimeError(
            "Loaded DDP model does not expose the three tensors used by the "
            "legacy optimizer"
        ) from error

    unique_parameters = []
    seen_ids = set()
    for name, parameter in named_parameters:
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"DDP optimizer tensor {name} is not an nn.Parameter")
        if not parameter.requires_grad:
            raise RuntimeError(
                f"DDP optimizer tensor {name} unexpectedly has requires_grad=False"
            )
        if id(parameter) not in seen_ids:
            unique_parameters.append(parameter)
            seen_ids.add(id(parameter))
    return tuple(unique_parameters)


def _temperature(
    protocol: BenchmarkProtocol,
    seen_classes: int,
) -> float:
    options = protocol.method_options("ddp")
    temperature = options.get("temperature", {})
    if not isinstance(temperature, dict):
        raise ValueError("method_options.ddp.temperature must be a mapping")
    minimum = float(temperature.get("minimum", 1.0))
    maximum = float(temperature.get("maximum", 2.0))
    gamma = float(temperature.get("gamma", 0.7))
    if minimum <= 0 or maximum <= 0 or gamma <= 0 or maximum < minimum:
        raise ValueError("Invalid DDP temperature schedule")
    base_classes = len(protocol.current_class_indices(0))
    denominator = protocol.num_classes - base_classes
    progress = (
        (seen_classes - base_classes) / denominator if denominator else 1.0
    )
    progress = max(0.0, min(1.0, progress))
    return minimum + (maximum - minimum) * math.pow(progress, gamma)


def _check_evaluation_batch(
    batch: Any,
    task_context: TaskContext,
) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError(
            "DDP predict_scores requires evaluator-only EvaluationBatch values"
        )
    if batch.class_order_hash != task_context.class_order_hash:
        raise ValueError("Evaluation batch class order hash differs from task context")
    if batch.targets_seen.ndim != 2:
        raise ValueError("Evaluation targets must have shape [N, C_seen]")
    if batch.targets_seen.shape[1] != len(task_context.seen_class_indices):
        raise ValueError("Evaluation target columns do not match seen classes")
    return batch


def legacy_ddp_predict_scores(
    model: torch.nn.Module,
    data_loader: Iterable[EvaluationBatch],
    task_context: TaskContext,
    device: Union[str, torch.device],
    temperature: float,
) -> PredictionOutput:
    """Repository-legacy inference formula on a benchmark dataloader.

    This intentionally mirrors `eval_emotic_all_tasks.py`: DDP logits, one
    global task temperature, softmax over negative/positive paths, and the
    positive probability. It exists for a same-loader two-path equivalence test;
    historical score files cannot provide sample-ID equivalence.
    """

    resolved_device = torch.device(device)
    model.eval()
    scores: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    sample_ids: List[str] = []
    split_hash: Optional[str] = None
    seen_classes = len(task_context.seen_class_indices)
    with torch.no_grad():
        for raw_batch in data_loader:
            batch = _check_evaluation_batch(raw_batch, task_context)
            if split_hash is None:
                split_hash = batch.split_hash
            elif split_hash != batch.split_hash:
                raise ValueError("Evaluation loader contains multiple split hashes")
            inputs = batch.images.to(resolved_device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=resolved_device.type == "cuda"):
                logits = model(
                    inputs,
                    cls_id=(0, seen_classes),
                    inference=True,
                )
            scores.append(
                torch.softmax(logits.float().cpu() / temperature, dim=1)[:, 1, :]
            )
            targets.append(batch.targets_seen.detach().float().cpu())
            sample_ids.extend(batch.sample_ids)
    if not scores or split_hash is None:
        raise ValueError("Evaluation loader produced no samples")
    return PredictionOutput(
        scores=torch.cat(scores),
        targets=torch.cat(targets),
        sample_ids=sample_ids,
        class_order_hash=task_context.class_order_hash,
        split_hash=split_hash,
    )


class DDPBenchmarkMethod(BenchmarkMethod):
    """Owns one existing `models.ddp.DDP`; no model code is copied."""

    method_name = "DDP"
    method_family = "Prompt"
    backbone = "OpenAI CLIP ViT-B/16"
    supported_tracks = ("A",)
    upstream_repository = "TBD"
    upstream_commit = "TBD"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        checkpoint_paths: Optional[
            Union[Sequence[Union[str, Path]], Mapping[int, Union[str, Path]]]
        ] = None,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError(
                f"DDP supports tracks {self.supported_tracks}, got {protocol.track}"
            )
        self.protocol = protocol
        self.clip_model_path = str(clip_model_path)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _unwrapped(model) if model is not None else None
        if self.model is not None:
            self.model.to(self.device)
        self.checkpoint_paths = self._normalize_checkpoint_paths(checkpoint_paths)
        self.task_context: Optional[TaskContext] = None
        self.loaded_checkpoint: Optional[Mapping[str, Any]] = None
        self.loaded_checkpoint_path: Optional[Path] = None

    @staticmethod
    def _normalize_checkpoint_paths(
        paths: Optional[
            Union[Sequence[Union[str, Path]], Mapping[int, Union[str, Path]]]
        ]
    ) -> Mapping[int, Path]:
        if paths is None:
            return {}
        if isinstance(paths, Mapping):
            normalized = {int(key): Path(value) for key, value in paths.items()}
        else:
            normalized = {
                task_id: Path(value) for task_id, value in enumerate(paths)
            }
        if any(task_id < 0 for task_id in normalized):
            raise ValueError("Checkpoint task IDs must be non-negative")
        return normalized

    def begin_task(self, task_context: TaskContext) -> None:
        if task_context.protocol_id != self.protocol.protocol_id:
            raise ValueError("Task context protocol_id differs from DDP protocol")
        if task_context.protocol_hash != self.protocol.protocol_hash:
            raise ValueError("Task context protocol_hash differs from DDP protocol")
        if task_context.class_order_hash != self.protocol.class_order_hash:
            raise ValueError("Task context class_order_hash differs from DDP protocol")
        self.task_context = task_context
        checkpoint_path = self.checkpoint_paths.get(task_context.task_id)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif self.model is None:
            raise FileNotFoundError(
                f"No DDP checkpoint configured for task {task_context.task_id}"
            )
        self._rebuild_text_feature_cache()

    def train_task(self, train_loader: Iterable[Any], val_loader: Iterable[Any]) -> None:
        """Legacy DDP checkpoints are trained by the existing unchanged script."""

        if self.task_context is None or self.model is None:
            raise RuntimeError("begin_task must be called before train_task")
        if self.task_context.task_id not in self.checkpoint_paths:
            # An injected model is allowed for unit/smoke tests only.
            return

    def predict_scores(
        self, data_loader: Iterable[EvaluationBatch]
    ) -> PredictionOutput:
        if self.task_context is None or self.model is None:
            raise RuntimeError("begin_task must be called before predict_scores")
        self.model.eval()
        seen_classes = len(self.task_context.seen_class_indices)
        temperature = _temperature(self.protocol, seen_classes)
        output_batches: List[torch.Tensor] = []
        target_batches: List[torch.Tensor] = []
        sample_ids: List[str] = []
        split_hash: Optional[str] = None
        with torch.no_grad():
            for raw_batch in data_loader:
                batch = _check_evaluation_batch(raw_batch, self.task_context)
                if split_hash is None:
                    split_hash = batch.split_hash
                elif split_hash != batch.split_hash:
                    raise ValueError(
                        "Evaluation loader contains multiple split hashes"
                    )
                inputs = batch.images.to(
                    self.device, non_blocking=True
                ).float()
                with torch.cuda.amp.autocast(
                    enabled=self.device.type == "cuda"
                ):
                    logits = self.model(
                        inputs,
                        cls_id=(0, seen_classes),
                        inference=True,
                    )
                scores = torch.softmax(
                    logits.float().cpu() / temperature,
                    dim=1,
                )[:, 1, :]
                output_batches.append(scores)
                target_batches.append(
                    batch.targets_seen.detach().float().cpu()
                )
                sample_ids.extend(batch.sample_ids)
        if not output_batches or split_hash is None:
            raise ValueError("Evaluation loader produced no samples")
        return PredictionOutput(
            scores=torch.cat(output_batches),
            targets=torch.cat(target_batches),
            sample_ids=sample_ids,
            class_order_hash=self.task_context.class_order_hash,
            split_hash=split_hash,
        )

    @property
    def current_temperature(self) -> float:
        if self.task_context is None:
            raise RuntimeError("begin_task must be called before reading temperature")
        return _temperature(
            self.protocol,
            len(self.task_context.seen_class_indices),
        )

    def end_task(self) -> None:
        self.task_context = None

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.loaded_checkpoint_path is None:
            raise RuntimeError(
                "DDP wrapper only saves an unchanged loaded legacy checkpoint"
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.loaded_checkpoint_path == destination.resolve():
            return
        shutil.copy2(self.loaded_checkpoint_path, destination)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("Legacy DDP checkpoint must contain a model state")
        classnames = tuple(checkpoint.get("classnames", ()))
        if classnames != self.protocol.class_order:
            raise ValueError("DDP checkpoint and protocol class orders differ")
        if self.task_context is not None:
            checkpoint_task = int(
                checkpoint.get("task", self.task_context.task_id)
            )
            if checkpoint_task != self.task_context.task_id:
                raise ValueError(
                    f"Checkpoint task {checkpoint_task} != context task "
                    f"{self.task_context.task_id}"
                )
        if self.model is None:
            self.model = self._build_existing_model(checkpoint)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()
        self.loaded_checkpoint = checkpoint
        self.loaded_checkpoint_path = checkpoint_path.resolve()

    def _build_existing_model(
        self, checkpoint: Mapping[str, Any]
    ) -> torch.nn.Module:
        # These imports are intentionally lazy so protocol/interface tests do
        # not construct CLIP or require Dassl.
        from build_cfg import setup_cfg
        from eval_emotic_threshold_sweep import checkpoint_model_args
        from models import ddp

        cli_args = SimpleNamespace(clip_model_path=self.clip_model_path)
        model_args = checkpoint_model_args(checkpoint, cli_args)
        cfg = setup_cfg(model_args)
        return _unwrapped(ddp(cfg, list(self.protocol.class_order)))

    def _rebuild_text_feature_cache(self) -> None:
        if self.model is None or self.task_context is None:
            raise RuntimeError("DDP model and task context are required")
        if not hasattr(self.model, "prompt_learner"):
            # Test doubles may directly implement forward without text prompts.
            return
        from eval_emotic_threshold_sweep import rebuild_text_feature_cache

        rebuild_text_feature_cache(
            self.model,
            len(self.task_context.seen_class_indices),
        )

    def parameter_statistics(self) -> ParameterStatistics:
        if self.model is None:
            raise RuntimeError("DDP model is not loaded")
        total = sum(parameter.numel() for parameter in self.model.parameters())
        # Match DDP.build_optimizer_scheduler rather than counting every tensor
        # that happens to retain requires_grad=True. Several frozen-path tensors
        # are not registered with the optimizer and therefore are not trainable
        # under the audited legacy recipe.
        trainable = sum(
            parameter.numel()
            for parameter in _legacy_optimizer_parameters(self.model)
        )
        # Legacy DDP preallocates its configured prompt tensors; model structure
        # does not grow after a task boundary.
        per_task = {
            task_id: 0 for task_id in range(self.protocol.num_tasks)
        }
        return ParameterStatistics(
            total_parameters=total,
            trainable_parameters=trainable,
            incremental_parameters=0,
            per_task_incremental_parameters=per_task,
        )

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(
            replay_memory_samples=0,
            replay_memory_bytes=0,
        )


register_method("ddp", DDPBenchmarkMethod)
