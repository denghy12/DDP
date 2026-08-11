"""EmotionCLIP converted to current-label-only sequential fine-tuning."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
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
from .model import EmotionCLIPFTModel, EmotionCLIPVisualTransformer


UPSTREAM_COMMIT = "ca142dd4c9664acb2b59c8b14ef3169049f1180b"
UPSTREAM_TREE = "c313af40432b83d4ef41700fb216de2ddecc9ecb882390d939e0b0c9690f79a7"


@dataclass(frozen=True)
class EmotionCLIPFTOptions:
    feature_dim: int = 512
    epochs: int = 25
    early_stopping_patience: int = 5
    backbone_learning_rate: float = 1.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    adamw_beta1: float = 0.98
    adamw_beta2: float = 0.9
    adamw_eps: float = 1.0e-6
    gradient_clip_norm: float = 1.0
    amp: bool = True
    tf32: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EmotionCLIPFTOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown EmotionCLIP-FT option(s): " + ", ".join(unknown))
        options = cls(**{key: value[key] for key in value})
        if options.feature_dim <= 0 or options.epochs <= 0:
            raise ValueError("EmotionCLIP dimensions and epochs must be positive")
        if options.early_stopping_patience <= 0:
            raise ValueError("EmotionCLIP early-stopping patience must be positive")
        if min(options.backbone_learning_rate, options.head_learning_rate) <= 0:
            raise ValueError("EmotionCLIP learning rates must be positive")
        if options.weight_decay < 0 or options.gradient_clip_norm <= 0:
            raise ValueError("EmotionCLIP optimizer settings are invalid")
        if not 0 <= options.adamw_beta1 < 1 or not 0 <= options.adamw_beta2 < 1:
            raise ValueError("EmotionCLIP AdamW betas must lie in [0, 1)")
        if options.adamw_eps <= 0:
            raise ValueError("EmotionCLIP AdamW epsilon must be positive")
        return options


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id or context.protocol_hash != protocol.protocol_hash:
        raise ValueError("EmotionCLIP task context differs from the protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("EmotionCLIP class order differs from the protocol")
    if context.track != "B":
        raise ValueError("Native-backbone EmotionCLIP-FT supports Track B only")


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1:] != (4, 224, 224):
        raise ValueError("EmotionCLIP-FT requires image_bbox_mask [N, 4, 224, 224]")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("EmotionCLIP training requires protocol-safe TrainBatch values")
    _validate_images(batch.images)
    expected_shape = (batch.images.shape[0], len(context.current_class_indices))
    if batch.targets_current.shape != expected_shape:
        raise ValueError("EmotionCLIP current targets do not match the task")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("EmotionCLIP training batch exposes labels outside current classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("EmotionCLIP training IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("EmotionCLIP prediction requires EvaluationBatch values")
    _validate_images(batch.images)
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("EmotionCLIP evaluation class order differs")
    expected = (batch.images.shape[0], len(context.seen_class_indices))
    if batch.targets_seen.shape != expected:
        raise ValueError("EmotionCLIP evaluation targets do not match seen classes")
    return batch


class EmotionCLIPFTBenchmarkMethod(BenchmarkMethod):
    method_name = "EmotionCLIP-FT"
    method_family = "Static EMOTIC model / Sequential Fine-Tuning"
    backbone = "Native EmotionCLIP subject-aware ViT-B/32 visual encoder"
    supported_tracks = ("B",)
    upstream_repository = "https://github.com/Xeaver/EmotionCLIP"
    upstream_commit = UPSTREAM_COMMIT
    upstream_license = "MIT"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[EmotionCLIPFTModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("Native-backbone EmotionCLIP-FT supports Track B only")
        configured = dict(protocol.method_options("emotionclip_ft"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = EmotionCLIPFTOptions.from_mapping(configured)
        self.protocol = protocol
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_seed(protocol.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.options.tf32
            torch.backends.cudnn.allow_tf32 = self.options.tf32
        self._amp_enabled = self.options.amp and self.device.type == "cuda"
        self.checkpoint_path = (
            str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else None
        )
        self.initialization_sha256: Optional[str] = None
        if model is None:
            if self.checkpoint_path is None:
                raise FileNotFoundError(
                    "EmotionCLIP-FT requires the official EmotionCLIP checkpoint; "
                    "generic CLIP and scratch fallbacks are forbidden"
                )
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = EmotionCLIPFTModel(EmotionCLIPVisualTransformer())
            payload = torch.load(checkpoint, map_location="cpu")
            if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
                raise ValueError("Official EmotionCLIP checkpoint must contain model state")
            source_state = {}
            for raw_name, value in payload["model"].items():
                name = str(raw_name)
                if name.startswith("module."):
                    name = name[len("module."):]
                prefix = "backbone.visual."
                if name.startswith(prefix):
                    source_state[name[len(prefix):]] = value
            model.load_official_visual_state(source_state)
            self.initialization_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if model.feature_dim != self.options.feature_dim:
            raise ValueError("EmotionCLIP model feature dimension differs from options")
        self.model = model.float().to(self.device).requires_grad_(True)
        self.task_context: Optional[TaskContext] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names: Tuple[str, ...] = ()
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

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        if task_context.task_id != self._completed_task_id + 1:
            raise RuntimeError("EmotionCLIP-FT tasks must be trained sequentially")
        old_classes = len(task_context.seen_class_indices) - len(task_context.current_class_indices)
        if self.model.num_classes != old_classes:
            raise RuntimeError("EmotionCLIP heads do not match the task boundary")
        self.model.add_head(len(task_context.current_class_indices))
        self.model.to(self.device).requires_grad_(True)
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )
        self.training_history = []

    def _selection_map(self, loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede validation")
        self.model.eval()
        scores, targets = [], []
        with torch.no_grad():
            for raw_batch in loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                with self._autocast():
                    logits = self.model.current_logits(
                        batch.images.to(self.device, non_blocking=True).float()
                    )
                scores.append(torch.sigmoid(logits).float().cpu())
                targets.append(batch.targets_current.detach().float().cpu())
        if not scores:
            raise ValueError("Validation loader produced no samples")
        all_scores, all_targets = torch.cat(scores), torch.cat(targets)
        aps = [
            average_precision(all_scores[:, index], all_targets[:, index])
            for index in range(all_targets.shape[1])
        ]
        return 100.0 * sum(aps) / len(aps)

    def train_task(
        self,
        train_loader: Iterable[TrainBatch],
        val_loader: Iterable[TrainBatch],
    ) -> None:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede training")
        optimizer = torch.optim.AdamW(
            (
                {"params": self.model.visual_encoder.parameters(), "lr": self.options.backbone_learning_rate},
                {"params": self.model.heads.parameters(), "lr": self.options.head_learning_rate},
            ),
            betas=(self.options.adamw_beta1, self.options.adamw_beta2),
            eps=self.options.adamw_eps,
            weight_decay=self.options.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        best_map, best_state, stale_epochs = -math.inf, None, 0
        for epoch in range(self.options.epochs):
            self.model.train()
            total_loss = 0.0
            batches = optimizer_steps = skipped_steps = 0
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = batch.images.to(self.device, non_blocking=True).float()
                targets = batch.targets_current.to(self.device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss = F.binary_cross_entropy_with_logits(
                        self.model.current_logits(images), targets
                    )
                old_scale = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.options.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                stepped = float(scaler.get_scale()) >= old_scale
                optimizer_steps += int(stepped)
                skipped_steps += int(not stepped)
                total_loss += float(loss.detach().cpu())
                batches += 1
            if batches == 0:
                raise ValueError("Training loader produced no samples")
            validation_map = self._selection_map(val_loader)
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "backbone_learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
                    "classification_loss": total_loss / batches,
                    "validation_current_mAP": validation_map,
                    "optimizer_steps": float(optimizer_steps),
                    "skipped_optimizer_steps": float(skipped_steps),
                }
            )
            if validation_map > best_map:
                best_map = validation_map
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.options.early_stopping_patience:
                break
        if best_state is None:
            raise RuntimeError("No EmotionCLIP epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        self._completed_task_id = self.task_context.task_id

    def predict_scores(self, data_loader: Iterable[EvaluationBatch]) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede prediction")
        if self.model.num_classes != len(self.task_context.seen_class_indices):
            raise RuntimeError("EmotionCLIP model does not contain all seen classes")
        self.model.eval()
        scores, targets, sample_ids = [], [], []
        split_hash: Optional[str] = None
        with torch.no_grad():
            for raw_batch in data_loader:
                batch = _validate_eval_batch(raw_batch, self.task_context)
                if split_hash is None:
                    split_hash = batch.split_hash
                elif split_hash != batch.split_hash:
                    raise ValueError("Evaluation loader contains multiple split hashes")
                with self._autocast():
                    logits = self.model(
                        batch.images.to(self.device, non_blocking=True).float()
                    )
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

    def training_log_records(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": "sequential_finetuning",
            "conversion_interface": "EmotionCLIP-FT-v0.1",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_source_tree_sha256": UPSTREAM_TREE,
            "upstream_license": self.upstream_license,
            "upstream_static_emotic_map": 32.91,
            "upstream_static_protocol": "frozen normalized features + one-vs-rest LogisticRegression(C=2.5)",
            "native_checkpoint_path": self.checkpoint_path,
            "native_checkpoint_sha256": self.initialization_sha256,
            "input_mode": "image_bbox_mask",
            "preprocessing": "official bicubic short-side 224 + center crop + ImageNet normalization + bbox mask",
            "feature_normalization": "L2 before task heads (source linear-eval behavior)",
            "current_label_only": True,
            "old_label_truth_used": False,
            "future_label_truth_used": False,
            "distillation_enabled": False,
            "replay_enabled": False,
            "ewc_enabled": False,
            "benchmark_added_adapter": False,
            "visual_encoder_fine_tuned": True,
            "clip_text_encoder_used": False,
            "loss": "BCEWithLogits over current classes only",
            "selection_metric": "current_label_validation_mAP",
            **asdict(self.options),
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
                "protocol_id": self.protocol.protocol_id,
                "protocol_hash": self.protocol.protocol_hash,
                "class_order_hash": self.protocol.class_order_hash,
                "completed_task_id": self._completed_task_id,
                "head_sizes": list(self.model.head_sizes),
                "model": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
                "options": asdict(self.options),
                "native_checkpoint_sha256": self.initialization_sha256,
                "training_history": self.training_history,
            },
            destination,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported EmotionCLIP-FT checkpoint")
        expected = {
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
            "native_checkpoint_sha256": self.initialization_sha256,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Checkpoint {key} differs from EmotionCLIP-FT")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("Checkpoint EmotionCLIP options differ")
        completed = int(payload["completed_task_id"])
        expected_heads = tuple(
            len(self.protocol.current_class_indices(task_id)) for task_id in range(completed + 1)
        )
        head_sizes = tuple(int(value) for value in payload["head_sizes"])
        if head_sizes != expected_heads:
            raise ValueError("Checkpoint heads differ from the protocol")
        if self.model.heads:
            raise RuntimeError("Checkpoint loading requires an unexpanded model")
        self.model.restore_heads(head_sizes)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).requires_grad_(True)
        self._completed_task_id = completed
        self.training_history = [dict(row) for row in payload.get("training_history", [])]

    def parameter_statistics(self) -> ParameterStatistics:
        parameters = dict(self.model.named_parameters())
        total = sum(value.numel() for value in parameters.values())
        trainable = sum(
            parameters[name].numel()
            for name in self._optimizer_parameter_names
            if name in parameters
        )
        per_task = {
            task_id: (0 if task_id == 0 else len(self.protocol.current_class_indices(task_id)) * (self.options.feature_dim + 1))
            for task_id in range(self.protocol.num_tasks)
        }
        completed = max(self._completed_task_id, 0)
        incremental = sum(
            per_task[task_id]
            for task_id in range(1, min(completed + 1, self.protocol.num_tasks))
        )
        return ParameterStatistics(total, trainable, incremental, per_task)

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("emotionclip_ft", EmotionCLIPFTBenchmarkMethod)
