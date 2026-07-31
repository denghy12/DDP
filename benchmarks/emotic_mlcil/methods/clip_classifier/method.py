"""CLIP-visual Sequential Fine-Tuning, LwF, and EWC baselines."""

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
class CLIPClassifierOptions:
    feature_dim: int
    epochs: int
    early_stopping_patience: int
    backbone_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    amp: bool
    tf32: bool
    lwf_temperature: float
    lwf_weight: float
    ewc_lambda: float
    ewc_decay: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CLIPClassifierOptions":
        options = cls(
            feature_dim=int(value.get("feature_dim", 512)),
            epochs=int(value.get("epochs", 10)),
            early_stopping_patience=int(
                value.get("early_stopping_patience", 3)
            ),
            backbone_learning_rate=float(
                value.get("backbone_learning_rate", 1.0e-5)
            ),
            head_learning_rate=float(value.get("head_learning_rate", 1.0e-4)),
            weight_decay=float(value.get("weight_decay", 1.0e-4)),
            gradient_clip_norm=float(value.get("gradient_clip_norm", 1.0)),
            amp=bool(value.get("amp", True)),
            tf32=bool(value.get("tf32", True)),
            lwf_temperature=float(value.get("lwf_temperature", 2.0)),
            lwf_weight=float(value.get("lwf_weight", 1.0)),
            ewc_lambda=float(value.get("ewc_lambda", 100.0)),
            ewc_decay=float(value.get("ewc_decay", 1.0)),
        )
        if options.feature_dim <= 0:
            raise ValueError("CLIP feature_dim must be positive")
        if options.epochs <= 0 or options.early_stopping_patience <= 0:
            raise ValueError("Epoch and early-stopping settings must be positive")
        if (
            options.backbone_learning_rate <= 0
            or options.head_learning_rate <= 0
            or options.weight_decay < 0
            or options.gradient_clip_norm <= 0
        ):
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
        raise ValueError("CLIP-visual baselines support Track A only")


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


class CLIPContinualMethod(BenchmarkMethod):
    """Shared classifier; subclasses select only the continual objective."""

    method_name = "CLIP Continual Classifier"
    method_family = "Classifier"
    backbone = "OpenAI CLIP ViT-B/16 visual encoder (fine-tuned)"
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
        self.options = CLIPClassifierOptions.from_mapping(
            protocol.method_options("clip_classifier")
        )
        self._set_seed(protocol.seed)
        self.clip_model_path = str(clip_model_path)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._amp_enabled = self.options.amp and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.options.tf32
            torch.backends.cudnn.allow_tf32 = self.options.tf32

        if model is not None and feature_extractor is not None:
            raise ValueError("Inject either model or feature_extractor, not both")
        if model is None:
            visual_encoder = (
                feature_extractor
                if feature_extractor is not None
                else self._load_clip_visual_encoder()
            )
            visual_encoder.float()
            visual_encoder.requires_grad_(True)
            model = GrowingMultiLabelClassifier(
                visual_encoder=visual_encoder,
                feature_dim=self.options.feature_dim,
            )
        if model.feature_dim != self.options.feature_dim:
            raise ValueError("Classifier feature dimension differs from config")
        model.float()
        model.requires_grad_(True)
        self.model = model.to(self.device)

        self.task_context: Optional[TaskContext] = None
        self._teacher: Optional[GrowingMultiLabelClassifier] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names: Tuple[str, ...] = ()
        self._ewc_fisher: Dict[str, torch.Tensor] = {}
        self._ewc_means: Dict[str, torch.Tensor] = {}
        self._ewc_fisher_device: Dict[str, torch.Tensor] = {}
        self._ewc_means_device: Dict[str, torch.Tensor] = {}
        self.training_history: List[Dict[str, float]] = []

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _load_clip_visual_encoder(self) -> torch.nn.Module:
        model_path = os.path.abspath(os.path.expanduser(self.clip_model_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CLIP model not found: {model_path}")
        from clip import clip

        # Build on CPU in float32, retain only the visual tower, and discard all
        # text-tower parameters. The visual state is then fine-tuned by the
        # continual method.
        clip_model, _ = clip.load(model_path, device="cpu", jit=False)
        visual_encoder = clip_model.visual.float()
        output_dim = int(getattr(visual_encoder, "output_dim", -1))
        if output_dim != self.options.feature_dim:
            raise ValueError(
                f"CLIP output dimension {output_dim} != configured "
                f"{self.options.feature_dim}"
            )
        return visual_encoder

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        return images.to(self.device, non_blocking=True).float()

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

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
        self.model.requires_grad_(True)
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        self._prepare_ewc_device_state()
        self.training_history = []

    def _prepare_ewc_device_state(self) -> None:
        if self.strategy != "ewc":
            self._ewc_fisher_device = {}
            self._ewc_means_device = {}
            return
        self._ewc_fisher_device = {
            name: value.to(self.device, non_blocking=True)
            for name, value in self._ewc_fisher.items()
        }
        self._ewc_means_device = {
            name: value.to(self.device, non_blocking=True)
            for name, value in self._ewc_means.items()
        }

    def _optimizer(self) -> torch.optim.Optimizer:
        backbone_parameters = list(self.model.visual_encoder.parameters())
        head_parameters = list(self.model.heads.parameters())
        return torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": self.options.backbone_learning_rate,
                },
                {
                    "params": head_parameters,
                    "lr": self.options.head_learning_rate,
                },
            ],
            weight_decay=self.options.weight_decay,
        )

    def _old_student_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self._teacher is None:
            raise RuntimeError("Old logits require an LwF teacher")
        old_head_count = len(self._teacher.heads)
        return torch.cat(
            [head(features) for head in self.model.heads[:old_head_count]],
            dim=1,
        )

    def _lwf_loss(
        self,
        images: torch.Tensor,
        student_features: torch.Tensor,
    ) -> torch.Tensor:
        if self._teacher is None:
            return torch.zeros((), device=self.device)
        temperature = self.options.lwf_temperature
        with torch.no_grad():
            teacher_logits = self._teacher(images)
            teacher_targets = torch.sigmoid(teacher_logits / temperature)
        student_logits = self._old_student_logits(student_features)
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
        if not self._ewc_fisher_device:
            return penalty
        for name, parameter in self.model.named_parameters():
            if name not in self._ewc_fisher_device:
                continue
            penalty = penalty + (
                self._ewc_fisher_device[name]
                * (parameter - self._ewc_means_device[name]).pow(2)
            ).sum()
        return 0.5 * penalty

    def _selection_map(self, loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("begin_task must be called before validation")
        self.model.eval()
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.no_grad():
            for raw_batch in loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._prepare_images(batch.images)
                with self._autocast():
                    logits = self.model.current_logits(images)
                scores.append(torch.sigmoid(logits).float().cpu())
                targets.append(batch.targets_current.detach().float().cpu())
        if not scores:
            raise ValueError("Validation loader produced no samples")
        all_scores = torch.cat(scores)
        all_targets = torch.cat(targets)
        values = [
            average_precision(all_scores[:, index], all_targets[:, index])
            for index in range(all_targets.shape[1])
        ]
        return 100.0 * sum(values) / len(values)

    def train_task(
        self,
        train_loader: Iterable[TrainBatch],
        val_loader: Iterable[TrainBatch],
    ) -> None:
        if self.task_context is None:
            raise RuntimeError("begin_task must be called before train_task")
        optimizer = self._optimizer()
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        best_map = -math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0

        for epoch in range(self.options.epochs):
            self.model.train()
            if self._teacher is not None:
                self._teacher.eval()
            classification_total = 0.0
            distillation_total = 0.0
            ewc_total = 0.0
            batches = 0

            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._prepare_images(batch.images)
                targets = batch.targets_current.to(
                    self.device,
                    non_blocking=True,
                ).float()
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    student_features = self.model.encode_images(images)
                    current_logits = self.model.heads[-1](student_features)
                    classification = F.binary_cross_entropy_with_logits(
                        current_logits,
                        targets,
                    )
                    distillation = self._lwf_loss(images, student_features)
                    ewc_penalty = self._ewc_penalty()
                    loss = (
                        classification
                        + self.options.lwf_weight * distillation
                        + self.options.ewc_lambda * ewc_penalty
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.options.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()

                classification_total += float(classification.detach().cpu())
                distillation_total += float(distillation.detach().cpu())
                ewc_total += float(ewc_penalty.detach().cpu())
                batches += 1

            if batches == 0:
                raise ValueError("Training loader produced no samples")
            selection_map = self._selection_map(val_loader)
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
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.options.early_stopping_patience:
                    break

        if best_state is None:
            raise RuntimeError("No training epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        if self.strategy == "ewc":
            self._consolidate_fisher(train_loader)
        self._completed_task_id = self.task_context.task_id

    def _consolidate_fisher(
        self,
        train_loader: Iterable[TrainBatch],
    ) -> None:
        if self.task_context is None:
            raise RuntimeError("Fisher estimation requires an active task")
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
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        batches = 0
        for raw_batch in train_loader:
            batch = _validate_train_batch(raw_batch, self.task_context)
            images = self._prepare_images(batch.images)
            targets = batch.targets_current.to(
                self.device,
                non_blocking=True,
            ).float()
            self.model.zero_grad(set_to_none=True)
            with self._autocast():
                loss = F.binary_cross_entropy_with_logits(
                    self.model.current_logits(images),
                    targets,
                )
            scaler.scale(loss).backward()
            inverse_scale = 1.0 / float(scaler.get_scale())
            for name, parameter in named_parameters.items():
                if parameter.grad is not None:
                    unscaled_gradient = (
                        parameter.grad.detach().float() * inverse_scale
                    )
                    fisher[name] += unscaled_gradient.pow(2)
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
        self._prepare_ewc_device_state()

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
                images = self._prepare_images(batch.images)
                with self._autocast():
                    logits = self.model(images)
                scores.append(torch.sigmoid(logits).float().cpu())
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
        self._ewc_fisher_device = {}
        self._ewc_means_device = {}

    def training_log_records(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": self.strategy,
            "visual_encoder_trainable": True,
            "clip_text_encoder_used": False,
            "benchmark_added_adapter": False,
            "selection_metric": "current_label_validation_mAP",
            "preprocessing": "shared_DDP_EMOTIC_full_image",
            **self.options.__dict__,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active task is required for checkpointing")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 2,
                "method": self.method_name,
                "strategy": self.strategy,
                "protocol_id": self.protocol.protocol_id,
                "protocol_hash": self.protocol.protocol_hash,
                "class_order_hash": self.protocol.class_order_hash,
                "completed_task_id": self._completed_task_id,
                "head_sizes": list(self.model.head_sizes),
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.model.state_dict().items()
                },
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
        if int(payload.get("schema_version", -1)) != 2:
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
        if self.model.heads:
            raise RuntimeError("Load checkpoint requires an unexpanded classifier")
        self.model.restore_heads(head_sizes)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self.model.requires_grad_(True)
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
        total = sum(parameter.numel() for parameter in self.model.parameters())
        name_to_parameter = dict(self.model.named_parameters())
        trainable = sum(
            name_to_parameter[name].numel()
            for name in self._optimizer_parameter_names
        )
        per_task = {}
        for task_id in range(self.protocol.num_tasks):
            per_task[task_id] = (
                0
                if task_id == 0
                else len(self.protocol.current_class_indices(task_id))
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


class SequentialFineTuningMethod(CLIPContinualMethod):
    method_name = "Sequential Fine-Tuning"
    method_family = "Fine-Tuning"
    strategy = "finetune"


class LearningWithoutForgettingMethod(CLIPContinualMethod):
    method_name = "LwF"
    method_family = "Distillation"
    strategy = "lwf"
    upstream_repository = "Repository-native adaptation of LwF"


class ElasticWeightConsolidationMethod(CLIPContinualMethod):
    method_name = "EWC"
    method_family = "Regularization"
    strategy = "ewc"
    upstream_repository = "Repository-native adaptation of EWC"


register_method("finetune", SequentialFineTuningMethod)
register_method("sequential_finetuning", SequentialFineTuningMethod)
register_method("lwf", LearningWithoutForgettingMethod)
register_method("ewc", ElasticWeightConsolidationMethod)
