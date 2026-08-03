"""Released DDP training/model behavior with the registered Tau2 PCD mapping."""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
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


SOURCE_SNAPSHOT_TREE_SHA256 = (
    "e0b9963e1d891ce95fc0c0d2444389f388fe5a1c2bd04f6ece7dc0617306e5a4"
)
SOURCE_MODEL_SHA256 = (
    "d7dfb6bd463b387db3224a8fb6631e25435bae5c174e65f22f328899fc42b309"
)
SOURCE_DRIVER_SHA256 = (
    "76d0767f9de062cf7939dc718f5ed2a6c2c4c3f98c2dd680a47652cea8501fca"
)
SOURCE_LOSS_SHA256 = (
    "eaedae20e4614874134565468d359a4ec2795726078b08eb96e74043e29d28c0"
)


@dataclass(frozen=True)
class OriginalDDPOptions:
    epochs: int = 20
    optimizer_lr: float = 5.9e-3
    loss_weight: float = 0.03
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    scheduler_milestones: Tuple[int, ...] = (0, 20)
    scheduler_gamma: float = 0.1
    temperature_minimum: float = 1.0
    temperature_maximum: float = 2.0
    temperature_gamma: float = 0.7
    n_ctx_positive: int = 16
    n_ctx_negative: int = 16
    visual_prompt_length: int = 16
    visual_prompt_layers: Tuple[int, ...] = (7, 8, 9, 10, 11)
    registered_train_batch_size: int = 8
    amp: bool = True
    tf32: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OriginalDDPOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown original DDP option(s): " + ", ".join(unknown))
        converted = dict(value)
        for key in ("scheduler_milestones", "visual_prompt_layers"):
            if key in converted:
                converted[key] = tuple(int(item) for item in converted[key])
        options = cls(**converted)
        if options.epochs <= 0 or options.registered_train_batch_size <= 0:
            raise ValueError("Original DDP epochs and batch size must be positive")
        if options.optimizer_lr <= 0 or options.loss_weight <= 0:
            raise ValueError("Original DDP learning rate and loss weight must be positive")
        if not 0 <= options.adam_beta1 < 1 or not 0 <= options.adam_beta2 < 1:
            raise ValueError("Original DDP Adam betas must lie in [0, 1)")
        if options.adam_eps <= 0 or not 0 < options.scheduler_gamma <= 1:
            raise ValueError("Original DDP optimizer/scheduler settings are invalid")
        if not options.scheduler_milestones or any(
            item < 0 for item in options.scheduler_milestones
        ):
            raise ValueError("Original DDP scheduler milestones are invalid")
        if (
            options.temperature_minimum <= 0
            or options.temperature_maximum < options.temperature_minimum
            or options.temperature_gamma <= 0
        ):
            raise ValueError("Original DDP PCD temperature settings are invalid")
        if min(
            options.n_ctx_positive,
            options.n_ctx_negative,
            options.visual_prompt_length,
        ) <= 0:
            raise ValueError("Original DDP prompt lengths must be positive")
        if options.visual_prompt_layers != (7, 8, 9, 10, 11):
            raise ValueError("The released DDP visual prompt layers are fixed")
        return options


class OriginalDDPLoss(torch.nn.Module):
    """The released summed two-path BCE objective."""

    def __init__(self, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[1] != 2:
            raise ValueError("Original DDP logits must have shape [N, 2, C]")
        if tuple(targets.shape) != (logits.shape[0], logits.shape[2]):
            raise ValueError("Original DDP targets do not match logits")
        probabilities = torch.softmax(logits, dim=1)
        positive = probabilities[:, 1, :].reshape(-1)
        negative = probabilities[:, 0, :].reshape(-1)
        labels = targets.reshape(-1)
        terms = labels * torch.log(positive.clamp_min(self.epsilon))
        terms = terms + (1.0 - labels) * torch.log(
            negative.clamp_min(self.epsilon)
        )
        return -terms.sum()


def original_ddp_temperature(
    seen_classes: int,
    total_classes: int,
    base_classes: int,
    minimum: float = 1.0,
    maximum: float = 2.0,
    gamma: float = 0.7,
) -> float:
    denominator = total_classes - base_classes
    progress = (
        (seen_classes - base_classes) / denominator if denominator else 1.0
    )
    progress = max(0.0, min(1.0, progress))
    return minimum + (maximum - minimum) * math.pow(progress, gamma)


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id:
        raise ValueError("Task context protocol_id differs from protocol")
    if context.protocol_hash != protocol.protocol_hash:
        raise ValueError("Task context protocol_hash differs from protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("Task context class order differs from protocol")
    if context.track != "A":
        raise ValueError("Original DDP supports Track A only")
    previous = tuple(
        index
        for task_id in range(context.task_id)
        for index in protocol.current_class_indices(task_id)
    )
    if context.seen_class_indices != previous + context.current_class_indices:
        raise ValueError("Original DDP requires protocol-order class expansion")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("Original DDP training requires TrainBatch values")
    current = len(context.current_class_indices)
    if batch.targets_current.ndim != 2 or batch.targets_current.shape[1] != current:
        raise ValueError("Original DDP current targets do not match the task")
    expected_shape = (batch.images.shape[0], len(context.class_order))
    if tuple(batch.visible_mask.shape) != expected_shape:
        raise ValueError("Original DDP visible mask shape differs")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("Original DDP batch exposes old or future labels")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("Original DDP training IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("Original DDP prediction requires EvaluationBatch values")
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("Original DDP evaluation class order differs")
    if (
        batch.targets_seen.ndim != 2
        or batch.targets_seen.shape[1] != len(context.seen_class_indices)
    ):
        raise ValueError("Original DDP evaluation targets do not match seen classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("Original DDP evaluation IDs and images are not aligned")
    return batch


class OriginalDDPBenchmarkMethod(BenchmarkMethod):
    method_name = "Original-DDP-Tau2"
    method_family = "Native MLCIL / Dual-Decoupled Prompting"
    backbone = "OpenAI CLIP ViT-B/16 (frozen)"
    supported_tracks = ("A",)
    source_kind = "fixed_local_original_ddp_snapshot_without_git_metadata"
    source_snapshot_tree_sha256 = SOURCE_SNAPSHOT_TREE_SHA256

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[torch.nn.Module] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("Original DDP supports Track A only")
        configured = dict(protocol.method_options("original_ddp"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = OriginalDDPOptions.from_mapping(configured)
        self.protocol = protocol
        self.clip_model_path = str(clip_model_path)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_seed(protocol.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.options.tf32
            torch.backends.cudnn.allow_tf32 = self.options.tf32
        self._amp_enabled = self.options.amp and self.device.type == "cuda"
        self.model = (
            model if model is not None else self._build_repository_base_model()
        )
        self.model.to(self.device)
        self._assert_source_model_shape()
        self._assert_adapter_free()
        self._freeze_non_optimizer_parameters()
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(self.options.scheduler_milestones),
            gamma=self.options.scheduler_gamma,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        self.criterion = OriginalDDPLoss()
        self.task_context: Optional[TaskContext] = None
        self._completed_task_id = -1
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

    def _build_repository_base_model(self) -> torch.nn.Module:
        model_path = os.path.abspath(os.path.expanduser(self.clip_model_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CLIP model not found: {model_path}")
        from build_cfg import setup_cfg
        from models import ddp

        args = SimpleNamespace(
            positive_prompt=None,
            negative_prompt=None,
            datadir=None,
            clip_model_path=model_path,
            resume=None,
            input_size=224,
            train_input_size=None,
            test_input_size=None,
            lr=None,
            csc=True,
            n_ctx_pos=self.options.n_ctx_positive,
            n_ctx_neg=self.options.n_ctx_negative,
            logit_scale=100.0,
            train_batch_size=self.options.registered_train_batch_size,
            finetune=False,
            finetune_backbone=False,
            finetune_attn=False,
            finetune_text=False,
            base_lr_mult=None,
            backbone_lr_mult=None,
            text_lr_mult=None,
            attn_lr_mult=None,
            portion=1.0,
            partial_portion=1.0 + 1.0e-6,
            mask_file=None,
            config_file="configs/models/vitb16_ep50.yaml",
            dataset_config_file=None,
        )
        built = ddp(setup_cfg(args), list(self.protocol.class_order))
        return built.module if hasattr(built, "module") else built

    def _assert_source_model_shape(self) -> None:
        prompt = getattr(self.model, "prompt_learner", None)
        visual = getattr(self.model, "visual_prompts", None)
        if prompt is None or visual is None:
            raise RuntimeError("Original DDP model lacks released prompt tensors")
        expected_positive = (
            self.protocol.num_classes,
            self.options.n_ctx_positive,
        )
        expected_negative = (
            self.protocol.num_classes,
            self.options.n_ctx_negative,
        )
        if tuple(prompt.ctx_pos.shape[:2]) != expected_positive:
            raise RuntimeError("Original DDP positive contexts are not class-specific")
        if tuple(prompt.ctx_neg.shape[:2]) != expected_negative:
            raise RuntimeError("Original DDP negative contexts are not class-specific")
        if tuple(visual.shape[:2]) != (
            2 * self.protocol.num_classes,
            self.options.visual_prompt_length,
        ):
            raise RuntimeError("Original DDP visual prompt layout differs")
        n_cls = getattr(self.model, "n_cls", self.protocol.num_classes)
        if int(n_cls) != self.protocol.num_classes:
            raise RuntimeError("Original DDP class count differs from protocol")

    def _assert_adapter_free(self) -> None:
        for name in ("feature_adapter", "feature_adapter_bank"):
            if getattr(self.model, name, None) is not None:
                raise RuntimeError(f"Original DDP forbids benchmark-added {name}")

    def _optimizer_parameters(self) -> Tuple[torch.nn.Parameter, ...]:
        prompt = self.model.prompt_learner
        parameters = (prompt.ctx_pos, prompt.ctx_neg, self.model.visual_prompts)
        if len({id(parameter) for parameter in parameters}) != 3:
            raise RuntimeError("Original DDP optimizer parameters are not unique")
        return parameters

    def _freeze_non_optimizer_parameters(self) -> None:
        optimizer_ids = {id(parameter) for parameter in self._optimizer_parameters()}
        for parameter in self.model.parameters():
            parameter.requires_grad_(id(parameter) in optimizer_ids)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        positive, negative, visual = self._optimizer_parameters()
        return torch.optim.Adam(
            [
                {"params": positive},
                {"params": negative},
                {"params": visual},
            ],
            lr=self.options.optimizer_lr,
            betas=(self.options.adam_beta1, self.options.adam_beta2),
            eps=self.options.adam_eps,
            weight_decay=0.0,
        )

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def _images(self, images: torch.Tensor) -> torch.Tensor:
        return images.to(self.device, non_blocking=True).float()

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        expected = self._completed_task_id + 1
        if task_context.task_id != expected:
            raise RuntimeError(
                f"Original DDP requires sequential tasks: expected {expected}, "
                f"got {task_context.task_id}"
            )
        self._assert_adapter_free()
        self.task_context = task_context
        self.training_history = []

    def _temperature(self, seen_classes: int) -> float:
        return original_ddp_temperature(
            seen_classes=seen_classes,
            total_classes=self.protocol.num_classes,
            base_classes=len(self.protocol.current_class_indices(0)),
            minimum=self.options.temperature_minimum,
            maximum=self.options.temperature_maximum,
            gamma=self.options.temperature_gamma,
        )

    @property
    def current_temperature(self) -> float:
        if self.task_context is None:
            raise RuntimeError("Original DDP task is not active")
        return self._temperature(len(self.task_context.seen_class_indices))

    def _rebuild_text_feature_cache(self) -> None:
        if self._completed_task_id < 0:
            return
        prompt = getattr(self.model, "prompt_learner", None)
        encoder = getattr(self.model, "text_encoder", None)
        if prompt is None or encoder is None:
            return
        self.model.text_feature_cache = {}
        with torch.no_grad():
            for task_id in range(self._completed_task_id + 1):
                indices = self.protocol.current_class_indices(task_id)
                bounds = (indices[0], indices[-1] + 1)
                prompts, tokens = prompt(bounds)
                features = encoder(prompts, tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                classes = len(indices)
                self.model.text_feature_cache[bounds] = {
                    "neg": features[:classes].detach().cpu(),
                    "pos": features[classes:].detach().cpu(),
                }

    def _current_validation_map(self, loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("Original DDP task is not active")
        self.model.eval()
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        current = self.task_context.current_class_indices
        low = current[0]
        high = current[-1] + 1
        seen = len(self.task_context.seen_class_indices)
        with torch.no_grad():
            for raw_batch in loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                with self._autocast():
                    logits = self.model(
                        self._images(batch.images),
                        cls_id=(0, seen),
                        inference=True,
                    )
                probabilities = torch.softmax(
                    logits.float() / self._temperature(seen), dim=1
                )[:, 1, low:high]
                scores.append(probabilities.cpu())
                targets.append(batch.targets_current.float().cpu())
        if not scores:
            raise ValueError("Original DDP validation loader produced no samples")
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
            raise RuntimeError("begin_task must precede Original DDP training")
        loader_batch_size = getattr(train_loader, "batch_size", None)
        if (
            loader_batch_size is not None
            and int(loader_batch_size) != self.options.registered_train_batch_size
        ):
            raise ValueError(
                "Original DDP requires registered train batch size "
                f"{self.options.registered_train_batch_size}, got "
                f"{loader_batch_size}"
            )
        current = self.task_context.current_class_indices
        bounds = (current[0], current[-1] + 1)
        for epoch in range(self.options.epochs):
            self.model.train()
            total_loss = 0.0
            batches = 0
            optimizer_steps = 0
            skipped_steps = 0
            epoch_learning_rate = float(self.optimizer.param_groups[0]["lr"])
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                targets = batch.targets_current.to(
                    self.device, non_blocking=True
                ).float()
                self.optimizer.zero_grad()
                with self._autocast():
                    logits = self.model(
                        self._images(batch.images),
                        cls_id=bounds,
                        inference=False,
                    )
                    loss = self.options.loss_weight * self.criterion(
                        logits, targets
                    )
                scale_before = float(self.scaler.get_scale())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if float(self.scaler.get_scale()) >= scale_before:
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
                total_loss += float(loss.detach().cpu())
                batches += 1
            if batches == 0:
                raise ValueError("Original DDP training loader produced no batches")
            self.scheduler.step()
            self.training_history.append(
                {
                    "task_id": float(self.task_context.task_id),
                    "epoch": float(epoch),
                    "scaled_source_bce": total_loss / batches,
                    "optimizer_steps": float(optimizer_steps),
                    "skipped_optimizer_steps": float(skipped_steps),
                    "learning_rate": epoch_learning_rate,
                    "next_learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    "scheduler_last_epoch": float(self.scheduler.last_epoch),
                }
            )
        self._completed_task_id = self.task_context.task_id
        self._rebuild_text_feature_cache()
        validation_map = self._current_validation_map(val_loader)
        self.training_history[-1]["validation_current_mAP"] = validation_map

    def predict_scores(
        self, data_loader: Iterable[EvaluationBatch]
    ) -> PredictionOutput:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("Original DDP task must be trained before prediction")
        self._rebuild_text_feature_cache()
        self.model.eval()
        seen = len(self.task_context.seen_class_indices)
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        sample_ids: List[str] = []
        split_hash: Optional[str] = None
        with torch.no_grad():
            for raw_batch in data_loader:
                batch = _validate_eval_batch(raw_batch, self.task_context)
                if split_hash is None:
                    split_hash = batch.split_hash
                elif split_hash != batch.split_hash:
                    raise ValueError("Original DDP evaluation mixes split hashes")
                with self._autocast():
                    logits = self.model(
                        self._images(batch.images),
                        cls_id=(0, seen),
                        inference=True,
                    )
                probabilities = torch.softmax(
                    logits.float() / self._temperature(seen), dim=1
                )[:, 1, :]
                scores.append(probabilities.cpu())
                targets.append(batch.targets_seen.detach().float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("Original DDP evaluation loader produced no samples")
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
        class_capacity = sum(
            parameter.numel() for parameter in self._optimizer_parameters()
        )
        return {
            "strategy": "original_ddp",
            "source_kind": self.source_kind,
            "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
            "source_model_sha256": SOURCE_MODEL_SHA256,
            "source_driver_sha256": SOURCE_DRIVER_SHA256,
            "source_loss_sha256": SOURCE_LOSS_SHA256,
            "source_git_commit": None,
            "source_license_status": "no_license_file_observed_reference_only",
            "paper": "DDP: Dual-Decoupled Prompting for Multi-Label Class-Incremental Learning",
            "visual_encoder_trainable": False,
            "text_encoder_optimizer_updated": False,
            "clip_text_encoder_used": True,
            "class_specific_positive_negative_text_prompts": True,
            "class_specific_interlayer_visual_prompts": True,
            "prompt_capacity_preallocated": True,
            "prompt_capacity_parameters": class_capacity,
            "prompt_capacity_parameters_per_class": (
                class_capacity // self.protocol.num_classes
            ),
            "benchmark_added_adapter": False,
            "replay_enabled": False,
            "optimizer_lifecycle": "one Adam instance retained across all tasks",
            "scheduler_lifecycle": "one MultiStepLR instance retained across all tasks",
            "loss_reduction": "released sum over samples and current classes",
            "training_label_scope": "current_classes_only",
            "old_future_ground_truth_used_for_training": False,
            "prompt_initialization": "released random class-specific contexts",
            "source_pcd_temperature": {
                "minimum": 1.0,
                "maximum": 7.0,
                "gamma": 0.2,
            },
            "registered_pcd_temperature": {
                "minimum": self.options.temperature_minimum,
                "maximum": self.options.temperature_maximum,
                "gamma": self.options.temperature_gamma,
            },
            "registered_pcd_mapping": (
                "benchmark-matched T=1->2, gamma=0.7 requested for direct "
                "comparison with the repository-local modified DDP"
            ),
            "emotic_mapping": (
                "dataset/task protocol, benchmark preprocessing, and the "
                "explicitly registered PCD schedule changed"
            ),
            "gradient_elision_correction": (
                "frozen CLIP parameters are requires_grad=False; released optimizer "
                "also excludes them, so this changes no forward value or prompt update"
            ),
            **asdict(self.options),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active Original DDP task is required")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "method": self.method_name,
                "protocol_id": self.protocol.protocol_id,
                "protocol_hash": self.protocol.protocol_hash,
                "class_order_hash": self.protocol.class_order_hash,
                "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
                "completed_task_id": self._completed_task_id,
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.model.state_dict().items()
                },
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "training_history": self.training_history,
                "options": asdict(self.options),
            },
            destination,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("Original DDP checkpoint must be a mapping")
        expected = {
            "schema_version": 1,
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
            "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Original DDP checkpoint {key} differs")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("Original DDP checkpoint options differ")
        completed = int(payload["completed_task_id"])
        if not 0 <= completed < self.protocol.num_tasks:
            raise ValueError("Original DDP checkpoint task is invalid")
        if self._completed_task_id >= 0:
            raise RuntimeError("Load Original DDP checkpoint into a fresh method")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self._freeze_non_optimizer_parameters()
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload.get("scaler", {}))
        self.training_history = [dict(row) for row in payload.get("training_history", [])]
        self._completed_task_id = completed
        self._rebuild_text_feature_cache()

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        optimizer_updated = sum(
            parameter.numel() for parameter in self._optimizer_parameters()
        )
        return ParameterStatistics(
            total_parameters=total,
            trainable_parameters=optimizer_updated,
            incremental_parameters=0,
            per_task_incremental_parameters={
                task_id: 0 for task_id in range(self.protocol.num_tasks)
            },
        )

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("original_ddp", OriginalDDPBenchmarkMethod)
