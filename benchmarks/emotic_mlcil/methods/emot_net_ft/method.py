"""Native-backbone EMOT-Net converted to sequential fine-tuning."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

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
from .model import EMOTNetFTModel


UPSTREAM_COMMIT = "69c3a5106aed08121cd12f6a5b359c745136931e"


@dataclass(frozen=True)
class EMOTNetFTOptions:
    fusion_dim: int = 256
    epochs: int = 21
    early_stopping_patience: int = 21
    learning_rate: float = 0.01
    lr_drop_epoch: int = 7
    lr_drop_gamma: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5.0e-4
    dropout: float = 0.5
    norm_factor: float = 1.2
    discrete_loss_weight: float = 1.0 / 6.0
    gradient_clip_norm: float = 10.0
    amp: bool = False
    tf32: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EMOTNetFTOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown EMOT-Net-FT option(s): " + ", ".join(unknown))
        options = cls(**{key: value[key] for key in value})
        if min(
            options.fusion_dim,
            options.epochs,
            options.early_stopping_patience,
            options.lr_drop_epoch,
        ) <= 0:
            raise ValueError("EMOT-Net dimensions and epoch settings must be positive")
        if options.learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("EMOT-Net optimizer settings are invalid")
        if not 0 <= options.momentum < 1:
            raise ValueError("EMOT-Net momentum must lie in [0, 1)")
        if not 0 < options.lr_drop_gamma <= 1:
            raise ValueError("EMOT-Net lr_drop_gamma must lie in (0, 1]")
        if not 0 <= options.dropout < 1 or options.norm_factor <= 1:
            raise ValueError("EMOT-Net dropout or norm_factor is invalid")
        if options.gradient_clip_norm <= 0:
            raise ValueError("EMOT-Net gradient clipping must be positive")
        if options.discrete_loss_weight <= 0:
            raise ValueError("EMOT-Net discrete loss weight must be positive")
        return options


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id or context.protocol_hash != protocol.protocol_hash:
        raise ValueError("EMOT-Net task context differs from the protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("EMOT-Net class order differs from the protocol")
    if context.track != "B":
        raise ValueError("Native-backbone EMOT-Net-FT supports Track B only")


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 5 or images.shape[1] != 2 or images.shape[2] != 3:
        raise ValueError("EMOT-Net requires body_context [N, 2, 3, H, W] images")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("EMOT-Net training requires protocol-safe TrainBatch values")
    _validate_images(batch.images)
    current = len(context.current_class_indices)
    if batch.targets_current.shape != (batch.images.shape[0], current):
        raise ValueError("EMOT-Net current targets do not match the task")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("EMOT-Net training batch exposes labels outside current classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("EMOT-Net training IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("EMOT-Net prediction requires EvaluationBatch values")
    _validate_images(batch.images)
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("EMOT-Net evaluation class order differs")
    expected = (batch.images.shape[0], len(context.seen_class_indices))
    if batch.targets_seen.shape != expected:
        raise ValueError("EMOT-Net evaluation targets do not match seen classes")
    return batch


def class_weights_from_current_targets(
    targets: torch.Tensor,
    norm_factor: float,
) -> torch.Tensor:
    """Exact source formula, restricted to protocol-visible current labels."""

    if targets.ndim != 2:
        raise ValueError("EMOT-Net class-weight targets must be two-dimensional")
    counts = targets.float().gt(0).sum(dim=0)
    weights = torch.full_like(counts, 1.0e-4, dtype=torch.float32)
    present = counts >= 1
    weights[present] = 1.0 / torch.log(counts[present] + float(norm_factor))
    return weights


def weighted_sigmoid_mse(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape or weights.shape != (logits.shape[1],):
        raise ValueError("EMOT-Net weighted-MSE inputs are not aligned")
    return ((torch.sigmoid(logits) - targets).pow(2) * weights.unsqueeze(0)).mean()


class EMOTNetFTBenchmarkMethod(BenchmarkMethod):
    """Official EMOT-Net model under a no-memory sequential FT lifecycle."""

    method_name = "EMOT-Net-FT"
    method_family = "Static EMOTIC model / Sequential Fine-Tuning"
    backbone = "Native EMOT-Net Places-context + AlexNet-body CNN"
    supported_tracks = ("B",)
    upstream_repository = "https://github.com/rkosti/emotic"
    upstream_commit = UPSTREAM_COMMIT
    upstream_license = "MIT"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        native_initialization_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[EMOTNetFTModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("Native-backbone EMOT-Net-FT supports Track B only")
        configured = dict(protocol.method_options("emot_net_ft"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = EMOTNetFTOptions.from_mapping(configured)
        self.protocol = protocol
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_seed(protocol.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.options.tf32
            torch.backends.cudnn.allow_tf32 = self.options.tf32
        self._amp_enabled = self.options.amp and self.device.type == "cuda"

        self.native_initialization_path = (
            str(Path(native_initialization_path).expanduser().resolve())
            if native_initialization_path is not None
            else None
        )
        self.native_initialization_sha256: Optional[str] = None
        self.native_source_asset_sha256: Dict[str, str] = {}
        if model is None:
            if self.native_initialization_path is None:
                raise FileNotFoundError(
                    "EMOT-Net-FT requires an audited PyTorch conversion of the official "
                    "Dropbox Places/AlexNet initialization; CLIP and silent scratch fallback are forbidden"
                )
            initialization = Path(self.native_initialization_path)
            if not initialization.is_file():
                raise FileNotFoundError(initialization)
            model = EMOTNetFTModel(
                fusion_dim=self.options.fusion_dim,
                dropout=self.options.dropout,
            )
            payload = torch.load(initialization, map_location="cpu")
            if not isinstance(payload, Mapping):
                raise ValueError("EMOT-Net native initialization must be a mapping")
            model.load_native_initialization(payload)
            self.native_initialization_sha256 = hashlib.sha256(initialization.read_bytes()).hexdigest()
            self.native_source_asset_sha256 = {
                str(name): str(value)
                for name, value in dict(payload["source_assets"]).items()
            }
        if model.fusion_dim != self.options.fusion_dim:
            raise ValueError("EMOT-Net model fusion width differs from options")
        self.model = model.float().to(self.device)
        self.model.requires_grad_(True)
        self.task_context: Optional[TaskContext] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names: Tuple[str, ...] = ()
        self.training_history: List[Dict[str, float]] = []
        self.current_class_weights: Optional[torch.Tensor] = None

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
            raise RuntimeError("EMOT-Net-FT tasks must be trained sequentially")
        old_classes = len(task_context.seen_class_indices) - len(task_context.current_class_indices)
        if self.model.num_classes != old_classes:
            raise RuntimeError("EMOT-Net heads do not match the task boundary")
        self.model.add_head(len(task_context.current_class_indices))
        self.model.to(self.device).requires_grad_(True)
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )
        self.training_history = []
        self.current_class_weights = None

    def _selection_map(self, loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede validation")
        self.model.eval()
        scores, targets = [], []
        with torch.no_grad():
            for raw_batch in loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = batch.images.to(self.device, non_blocking=True).float()
                with self._autocast():
                    logits = self.model.current_logits(images)
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
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.options.learning_rate,
            momentum=self.options.momentum,
            weight_decay=self.options.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.options.lr_drop_epoch,
            gamma=self.options.lr_drop_gamma,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        best_map = -math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0

        for epoch in range(self.options.epochs):
            self.model.train()
            classification_total = loss_total = 0.0
            batches = optimizer_steps = skipped_steps = 0
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = batch.images.to(self.device, non_blocking=True).float()
                targets = batch.targets_current.to(self.device, non_blocking=True).float()
                weights_cpu = class_weights_from_current_targets(
                    batch.targets_current.detach().float().cpu(),
                    self.options.norm_factor,
                )
                self.current_class_weights = weights_cpu.clone()
                weights = weights_cpu.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    classification = weighted_sigmoid_mse(
                        self.model.current_logits(images), targets, weights
                    )
                    loss = self.options.discrete_loss_weight * classification
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
                classification_total += float(classification.detach().cpu())
                loss_total += float(loss.detach().cpu())
                batches += 1
            if batches == 0:
                raise ValueError("Training loader produced no samples")
            validation_map = self._selection_map(val_loader)
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "weighted_sigmoid_mse": classification_total / batches,
                    "weighted_classification_loss": loss_total / batches,
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
            scheduler.step()
            if stale_epochs >= self.options.early_stopping_patience:
                break
        if best_state is None:
            raise RuntimeError("No EMOT-Net epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        self._completed_task_id = self.task_context.task_id

    def predict_scores(
        self,
        data_loader: Iterable[EvaluationBatch],
    ) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede prediction")
        if self.model.num_classes != len(self.task_context.seen_class_indices):
            raise RuntimeError("EMOT-Net model does not contain all seen classes")
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
        self.current_class_weights = None

    def training_log_records(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": "sequential_finetuning",
            "conversion_interface": "EMOT-Net-FT-v0.1",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_license": self.upstream_license,
            "native_context_backbone": "model_myVDavg_640_Places.t7 architecture",
            "native_body_backbone": "alexnet_features.t7 official release architecture",
            "native_release_archive_sha256": (
                "ce6096c1af5a3e91badbc06752e2dbbd04fc63f67f24acc95d76a68e1f7e339b"
            ),
            "native_body_variant": "official_dropbox_alexnet",
            "native_initialization_path": self.native_initialization_path,
            "native_initialization_sha256": self.native_initialization_sha256,
            "native_source_asset_sha256": dict(self.native_source_asset_sha256),
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
            "source_static_class_sampling_rejected_reason": "would expose future-label membership",
            **asdict(self.options),
        }

    def checkpoint_extra_metadata(self) -> Mapping[str, Any]:
        """Extension point for source methods hosted by the EMOT-Net lifecycle."""

        return {}

    def validate_checkpoint_extra_metadata(self, value: Mapping[str, Any]) -> None:
        if dict(value):
            raise ValueError("Checkpoint contains unexpected EMOT-Net method metadata")

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
                "model": {
                    name: value.detach().cpu()
                    for name, value in self.model.state_dict().items()
                },
                "options": asdict(self.options),
                "native_initialization_sha256": self.native_initialization_sha256,
                "native_source_asset_sha256": dict(self.native_source_asset_sha256),
                "method_extra": dict(self.checkpoint_extra_metadata()),
                "training_history": self.training_history,
                "current_class_weights": self.current_class_weights,
            },
            destination,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported EMOT-Net checkpoint")
        expected = {
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Checkpoint {key} differs from EMOT-Net-FT")
        checkpoint_options = dict(payload.get("options", {}))
        expected_options = asdict(self.options)
        if checkpoint_options != expected_options:
            raise ValueError("Checkpoint EMOT-Net options differ")
        if payload.get("native_initialization_sha256") != self.native_initialization_sha256:
            raise ValueError("Checkpoint native initialization differs")
        if dict(payload.get("native_source_asset_sha256", {})) != self.native_source_asset_sha256:
            raise ValueError("Checkpoint native source assets differ")
        method_extra = payload.get("method_extra", {})
        if not isinstance(method_extra, Mapping):
            raise ValueError("Checkpoint method metadata must be a mapping")
        self.validate_checkpoint_extra_metadata(method_extra)
        completed = int(payload["completed_task_id"])
        expected_heads = tuple(
            len(self.protocol.current_class_indices(task_id))
            for task_id in range(completed + 1)
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
        weights = payload.get("current_class_weights")
        self.current_class_weights = weights.detach().cpu() if isinstance(weights, torch.Tensor) else None

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        parameters = dict(self.model.named_parameters())
        trainable = sum(
            parameters[name].numel()
            for name in self._optimizer_parameter_names
            if name in parameters
        )
        per_task = {
            task_id: (
                0
                if task_id == 0
                else len(self.protocol.current_class_indices(task_id))
                * (self.options.fusion_dim + 1)
            )
            for task_id in range(self.protocol.num_tasks)
        }
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
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("emot_net_ft", EMOTNetFTBenchmarkMethod)
