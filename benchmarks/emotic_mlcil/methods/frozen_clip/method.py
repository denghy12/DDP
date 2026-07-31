"""Frozen-CLIP Sequential Fine-Tuning, LwF, and EWC baselines."""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.helper_functions.detail_report import average_precision

from ...method_base import BenchmarkMethod
from ...protocol import BenchmarkProtocol
from ...registry import register_method
from ...types import (
    EvaluationBatch,
    MemoryStatistics,
    ParameterStatistics,
    PredictionOutput,
    TaskContext,
    TrainBatch,
)
from .model import GrowingMultiLabelClassifier


@dataclass(frozen=True)
class FrozenCLIPOptions:
    feature_dim: int
    bottleneck_dim: int
    residual_scale: float
    epochs: int
    optimization_batch_size: int
    learning_rate: float
    weight_decay: float
    lwf_temperature: float
    lwf_weight: float
    ewc_lambda: float
    ewc_decay: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenCLIPOptions":
        options = cls(
            feature_dim=int(value.get("feature_dim", 512)),
            bottleneck_dim=int(value.get("bottleneck_dim", 128)),
            residual_scale=float(value.get("residual_scale", 0.1)),
            epochs=int(value.get("epochs", 20)),
            optimization_batch_size=int(
                value.get("optimization_batch_size", 256)
            ),
            learning_rate=float(value.get("learning_rate", 1.0e-3)),
            weight_decay=float(value.get("weight_decay", 1.0e-4)),
            lwf_temperature=float(value.get("lwf_temperature", 2.0)),
            lwf_weight=float(value.get("lwf_weight", 1.0)),
            ewc_lambda=float(value.get("ewc_lambda", 100.0)),
            ewc_decay=float(value.get("ewc_decay", 1.0)),
        )
        if options.feature_dim <= 0 or options.bottleneck_dim <= 0:
            raise ValueError("Frozen-CLIP feature dimensions must be positive")
        if options.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if options.epochs <= 0 or options.optimization_batch_size <= 0:
            raise ValueError("epochs and optimization_batch_size must be positive")
        if options.learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if options.lwf_temperature <= 0 or options.lwf_weight < 0:
            raise ValueError("Invalid LwF settings")
        if options.ewc_lambda < 0 or not 0 <= options.ewc_decay <= 1:
            raise ValueError("Invalid EWC settings")
        return options


def _validate_context(
    protocol: BenchmarkProtocol,
    task_context: TaskContext,
) -> None:
    if task_context.protocol_id != protocol.protocol_id:
        raise ValueError("Task context protocol_id differs from protocol")
    if task_context.protocol_hash != protocol.protocol_hash:
        raise ValueError("Task context protocol_hash differs from protocol")
    if task_context.class_order_hash != protocol.class_order_hash:
        raise ValueError("Task context class_order_hash differs from protocol")
    if task_context.track != "A":
        raise ValueError("Frozen-CLIP baselines support Track A only")


def _validate_train_batch(
    batch: Any,
    context: TaskContext,
) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("Training requires protocol-safe TrainBatch values")
    expected = len(context.current_class_indices)
    if batch.targets_current.ndim != 2 or batch.targets_current.shape[1] != expected:
        raise ValueError("Training target columns do not match current classes")
    if batch.visible_mask.ndim != 2:
        raise ValueError("Training visible_mask must have shape [N, C_total]")
    expected_mask = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected_mask[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected_mask):
        raise ValueError("Training batch exposes labels outside current classes")
    return batch


def _validate_evaluation_batch(
    batch: Any,
    context: TaskContext,
) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("Prediction requires evaluator-only EvaluationBatch values")
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("Evaluation class order differs from task context")
    expected = len(context.seen_class_indices)
    if batch.targets_seen.ndim != 2 or batch.targets_seen.shape[1] != expected:
        raise ValueError("Evaluation target columns do not match seen classes")
    return batch


class FrozenCLIPContinualMethod(BenchmarkMethod):
    """Shared implementation; subclasses select the continual objective."""

    method_name = "Frozen CLIP Continual Classifier"
    method_family = "Classifier"
    backbone = "OpenAI CLIP ViT-B/16 (frozen)"
    supported_tracks = ("A",)
    upstream_repository = "Repository-native baseline"
    upstream_commit = "N/A"
    strategy = "finetune"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        model: Optional[GrowingMultiLabelClassifier] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError(
                f"{self.method_name} supports Track A only, got {protocol.track}"
            )
        self.protocol = protocol
        self.options = FrozenCLIPOptions.from_mapping(
            protocol.method_options("frozen_clip_classifier")
        )
        random.seed(protocol.seed)
        np.random.seed(protocol.seed)
        torch.manual_seed(protocol.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(protocol.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.clip_model_path = str(clip_model_path)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._clip_model: Optional[torch.nn.Module] = None
        self._feature_extractor = feature_extractor
        if self._feature_extractor is not None:
            self._freeze_feature_extractor(self._feature_extractor)
        self.model = model or GrowingMultiLabelClassifier(
            feature_dim=self.options.feature_dim,
            bottleneck_dim=self.options.bottleneck_dim,
            residual_scale=self.options.residual_scale,
        )
        if self.model.feature_dim != self.options.feature_dim:
            raise ValueError("Injected classifier feature dimension differs from config")
        self.model.to(self.device)
        self.task_context: Optional[TaskContext] = None
        self._teacher: Optional[GrowingMultiLabelClassifier] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names: Tuple[str, ...] = ()
        self._ewc_fisher: Dict[str, torch.Tensor] = {}
        self._ewc_means: Dict[str, torch.Tensor] = {}
        self.training_history: List[Dict[str, float]] = []

    @staticmethod
    def _freeze_feature_extractor(module: torch.nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _ensure_feature_extractor(self) -> None:
        if self._feature_extractor is not None:
            self._feature_extractor.to(self.device)
            self._freeze_feature_extractor(self._feature_extractor)
            return
        model_path = os.path.abspath(os.path.expanduser(self.clip_model_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CLIP model not found: {model_path}")
        from clip import clip

        clip_model, _ = clip.load(model_path, device=self.device, jit=False)
        self._freeze_feature_extractor(clip_model)
        output_dim = int(getattr(clip_model.visual, "output_dim", -1))
        if output_dim != self.options.feature_dim:
            raise ValueError(
                f"CLIP output dimension {output_dim} != configured "
                f"{self.options.feature_dim}"
            )
        self._clip_model = clip_model
        self._feature_extractor = clip_model

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        self._ensure_feature_extractor()
        assert self._feature_extractor is not None
        self._feature_extractor.eval()
        inputs = images.to(self.device, non_blocking=True).float()
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    if hasattr(self._feature_extractor, "encode_image"):
                        features = self._feature_extractor.encode_image(inputs)
                    else:
                        features = self._feature_extractor(inputs)
            else:
                if hasattr(self._feature_extractor, "encode_image"):
                    features = self._feature_extractor.encode_image(inputs)
                else:
                    features = self._feature_extractor(inputs)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("Frozen feature extractor must return [N, feature_dim]")
        if features.shape[1] != self.options.feature_dim:
            raise ValueError("Frozen feature dimension differs from configuration")
        return F.normalize(features.float(), dim=-1)

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        expected_task = self._completed_task_id + 1
        if task_context.task_id != expected_task:
            raise RuntimeError(
                "Continual baselines require sequential tasks: expected "
                f"{expected_task}, got {task_context.task_id}"
            )
        expected_old_classes = (
            len(self.protocol.seen_class_indices(task_context.task_id - 1))
            if task_context.task_id > 0
            else 0
        )
        if self.model.num_classes != expected_old_classes:
            raise RuntimeError("Classifier head state does not match task boundary")
        if self.strategy == "lwf" and task_context.task_id > 0:
            self._teacher = copy.deepcopy(self.model).to(self.device).eval()
            self._teacher.requires_grad_(False)
        else:
            self._teacher = None
        self.model.add_head(len(task_context.current_class_indices))
        self.model.to(self.device)
        self.model.adapter.requires_grad_(True)
        self.model.heads[-1].requires_grad_(True)
        self.task_context = task_context
        optimizer_ids = {
            id(parameter)
            for parameter in (
                list(self.model.adapter.parameters())
                + list(self.model.heads[-1].parameters())
            )
        }
        self._optimizer_parameter_names = tuple(
            name
            for name, parameter in self.model.named_parameters()
            if id(parameter) in optimizer_ids
        )
        self.training_history = []

    def _cache_loader(
        self,
        loader: Iterable[TrainBatch],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_context is None:
            raise RuntimeError("begin_task must be called before caching data")
        features: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for raw_batch in loader:
            batch = _validate_train_batch(raw_batch, self.task_context)
            features.append(self._encode_images(batch.images).cpu())
            targets.append(batch.targets_current.detach().float().cpu())
        if not features:
            raise ValueError("Training/validation loader produced no samples")
        return torch.cat(features), torch.cat(targets)

    def _optimizer_parameters(self) -> List[torch.nn.Parameter]:
        name_to_parameter = dict(self.model.named_parameters())
        return [name_to_parameter[name] for name in self._optimizer_parameter_names]

    def _lwf_loss(self, features: torch.Tensor) -> torch.Tensor:
        if self._teacher is None:
            return torch.zeros((), device=features.device)
        temperature = self.options.lwf_temperature
        old_classes = self._teacher.num_classes
        with torch.no_grad():
            teacher_logits = self._teacher(features)
            teacher_targets = torch.sigmoid(teacher_logits / temperature)
        student_logits = self.model(features)[:, :old_classes]
        return (
            F.binary_cross_entropy_with_logits(
                student_logits / temperature,
                teacher_targets,
            )
            * temperature
            * temperature
        )

    def _ewc_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), device=self.device)
        if not self._ewc_fisher:
            return penalty
        for name, parameter in self.model.named_parameters():
            if name not in self._ewc_fisher:
                continue
            fisher = self._ewc_fisher[name].to(self.device)
            mean = self._ewc_means[name].to(self.device)
            penalty = penalty + (fisher * (parameter - mean).pow(2)).sum()
        return penalty

    @staticmethod
    def _selection_map(
        model: GrowingMultiLabelClassifier,
        features: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device,
    ) -> float:
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(
                model.current_logits(features.to(device))
            ).float().cpu()
        values = [
            average_precision(scores[:, index], targets[:, index])
            for index in range(targets.shape[1])
        ]
        return 100.0 * sum(values) / len(values)

    def train_task(
        self,
        train_loader: Iterable[TrainBatch],
        val_loader: Iterable[TrainBatch],
    ) -> None:
        if self.task_context is None:
            raise RuntimeError("begin_task must be called before train_task")
        train_features, train_targets = self._cache_loader(train_loader)
        val_features, val_targets = self._cache_loader(val_loader)
        optimizer = torch.optim.AdamW(
            self._optimizer_parameters(),
            lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )
        best_map = -math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        generator = torch.Generator().manual_seed(
            self.protocol.seed * 1000 + self.task_context.task_id
        )
        batch_size = self.options.optimization_batch_size
        for epoch in range(self.options.epochs):
            self.model.train()
            permutation = torch.randperm(
                train_features.shape[0],
                generator=generator,
            )
            classification_total = 0.0
            distillation_total = 0.0
            ewc_total = 0.0
            batches = 0
            for start in range(0, permutation.numel(), batch_size):
                indices = permutation[start : start + batch_size]
                features = train_features[indices].to(self.device)
                targets = train_targets[indices].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                current_logits = self.model.current_logits(features)
                classification = F.binary_cross_entropy_with_logits(
                    current_logits,
                    targets,
                )
                distillation = (
                    self._lwf_loss(features)
                    if self.strategy == "lwf"
                    else torch.zeros((), device=self.device)
                )
                ewc_penalty = (
                    self._ewc_penalty()
                    if self.strategy == "ewc"
                    else torch.zeros((), device=self.device)
                )
                loss = (
                    classification
                    + self.options.lwf_weight * distillation
                    + 0.5 * self.options.ewc_lambda * ewc_penalty
                )
                loss.backward()
                optimizer.step()
                classification_total += float(classification.detach().cpu())
                distillation_total += float(distillation.detach().cpu())
                ewc_total += float(ewc_penalty.detach().cpu())
                batches += 1
            selection_map = self._selection_map(
                self.model,
                val_features,
                val_targets,
                self.device,
            )
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "classification_loss": classification_total / batches,
                    "distillation_loss": distillation_total / batches,
                    "ewc_penalty": ewc_total / batches,
                    "selection_mAP": selection_map,
                }
            )
            if selection_map > best_map:
                best_map = selection_map
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
        if best_state is None:
            raise RuntimeError("No training epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        if self.strategy == "ewc":
            self._consolidate_fisher(train_features, train_targets)
        self._completed_task_id = self.task_context.task_id

    def _consolidate_fisher(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        named_parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        fisher = {
            name: torch.zeros_like(parameter, device=self.device)
            for name, parameter in named_parameters.items()
        }
        self.model.eval()
        batch_size = self.options.optimization_batch_size
        batches = 0
        for start in range(0, features.shape[0], batch_size):
            batch_features = features[start : start + batch_size].to(self.device)
            batch_targets = targets[start : start + batch_size].to(self.device)
            self.model.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                self.model.current_logits(batch_features),
                batch_targets,
            )
            loss.backward()
            for name, parameter in named_parameters.items():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().pow(2)
            batches += 1
        if batches == 0:
            raise RuntimeError("Cannot estimate Fisher information without data")
        updated_fisher: Dict[str, torch.Tensor] = {}
        updated_means: Dict[str, torch.Tensor] = {}
        for name, parameter in self.model.named_parameters():
            if name not in fisher:
                continue
            current = (fisher[name] / batches).detach().cpu()
            previous = self._ewc_fisher.get(name)
            updated_fisher[name] = (
                current
                if previous is None
                else self.options.ewc_decay * previous + current
            )
            updated_means[name] = parameter.detach().cpu().clone()
        self._ewc_fisher = updated_fisher
        self._ewc_means = updated_means

    def predict_scores(
        self,
        data_loader: Iterable[EvaluationBatch],
    ) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must be called before predict_scores")
        expected_classes = len(self.task_context.seen_class_indices)
        if self.model.num_classes != expected_classes:
            raise RuntimeError("Classifier does not contain all seen classes")
        self.model.eval()
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        sample_ids: List[str] = []
        split_hash: Optional[str] = None
        with torch.no_grad():
            for raw_batch in data_loader:
                batch = _validate_evaluation_batch(
                    raw_batch,
                    self.task_context,
                )
                if split_hash is None:
                    split_hash = batch.split_hash
                elif split_hash != batch.split_hash:
                    raise ValueError(
                        "Evaluation loader contains multiple split hashes"
                    )
                features = self._encode_images(batch.images)
                scores.append(torch.sigmoid(self.model(features)).float().cpu())
                targets.append(batch.targets_seen.detach().float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("Evaluation loader produced no samples")
        return PredictionOutput(
            scores=torch.cat(scores),
            targets=torch.cat(targets),
            sample_ids=sample_ids,
            class_order_hash=self.task_context.class_order_hash,
            split_hash=split_hash,
        )

    def end_task(self) -> None:
        self.task_context = None
        self._teacher = None

    def training_log_records(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": self.strategy,
            "feature_cache": "task_local_cpu",
            "selection_metric": "current_label_validation_mAP",
            **self.options.__dict__,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active task is required for checkpointing")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "method": self.method_name,
                "strategy": self.strategy,
                "protocol_id": self.protocol.protocol_id,
                "protocol_hash": self.protocol.protocol_hash,
                "class_order_hash": self.protocol.class_order_hash,
                "completed_task_id": self._completed_task_id,
                "head_sizes": list(self.model.head_sizes),
                "model": self.model.state_dict(),
                "ewc_fisher": self._ewc_fisher,
                "ewc_means": self._ewc_means,
                "training_history": self.training_history,
                "options": self.options.__dict__,
            },
            destination,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("Baseline checkpoint must be a mapping")
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported baseline checkpoint schema")
        expected = {
            "method": self.method_name,
            "strategy": self.strategy,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Checkpoint {key} differs from current method")
        if dict(payload.get("options", {})) != self.options.__dict__:
            raise ValueError("Checkpoint optimizer/model options differ")
        head_sizes = tuple(int(value) for value in payload["head_sizes"])
        completed_task_id = int(payload["completed_task_id"])
        if not 0 <= completed_task_id < self.protocol.num_tasks:
            raise ValueError("Checkpoint completed_task_id is invalid")
        expected_head_sizes = tuple(
            len(self.protocol.current_class_indices(task_id))
            for task_id in range(completed_task_id + 1)
        )
        if head_sizes != expected_head_sizes:
            raise ValueError("Checkpoint heads do not match completed protocol tasks")
        self.model = GrowingMultiLabelClassifier(
            feature_dim=self.options.feature_dim,
            bottleneck_dim=self.options.bottleneck_dim,
            residual_scale=self.options.residual_scale,
        )
        self.model.restore_heads(head_sizes)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self._completed_task_id = completed_task_id
        self._ewc_fisher = {
            str(name): tensor.detach().cpu()
            for name, tensor in dict(payload.get("ewc_fisher", {})).items()
        }
        self._ewc_means = {
            str(name): tensor.detach().cpu()
            for name, tensor in dict(payload.get("ewc_means", {})).items()
        }
        self.training_history = [
            {str(key): float(value) for key, value in dict(row).items()}
            for row in list(payload.get("training_history", []))
        ]

    def parameter_statistics(self) -> ParameterStatistics:
        if self._feature_extractor is None:
            self._ensure_feature_extractor()
        assert self._feature_extractor is not None
        total = sum(
            parameter.numel()
            for parameter in self._feature_extractor.parameters()
        )
        total += sum(parameter.numel() for parameter in self.model.parameters())
        name_to_parameter = dict(self.model.named_parameters())
        trainable = sum(
            name_to_parameter[name].numel()
            for name in self._optimizer_parameter_names
        )
        per_task = {}
        for task_id in range(self.protocol.num_tasks):
            if task_id == 0:
                per_task[task_id] = 0
            else:
                per_task[task_id] = (
                    len(self.protocol.current_class_indices(task_id))
                    * (self.options.feature_dim + 1)
                )
        completed = max(self._completed_task_id, 0)
        incremental = sum(
            per_task[task_id]
            for task_id in range(1, min(completed + 1, self.protocol.num_tasks))
        )
        return ParameterStatistics(
            total_parameters=total,
            trainable_parameters=trainable,
            incremental_parameters=incremental,
            per_task_incremental_parameters=per_task,
        )

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(
            replay_memory_samples=0,
            replay_memory_bytes=0,
        )


class SequentialFineTuningMethod(FrozenCLIPContinualMethod):
    method_name = "Sequential Fine-Tuning"
    method_family = "Fine-Tuning"
    strategy = "finetune"


class LearningWithoutForgettingMethod(FrozenCLIPContinualMethod):
    method_name = "LwF"
    method_family = "Distillation"
    strategy = "lwf"
    upstream_repository = "Repository-native adaptation of LwF"


class ElasticWeightConsolidationMethod(FrozenCLIPContinualMethod):
    method_name = "EWC"
    method_family = "Regularization"
    strategy = "ewc"
    upstream_repository = "Repository-native adaptation of EWC"


register_method("finetune", SequentialFineTuningMethod)
register_method("sequential_finetuning", SequentialFineTuningMethod)
register_method("lwf", LearningWithoutForgettingMethod)
register_method("ewc", ElasticWeightConsolidationMethod)
