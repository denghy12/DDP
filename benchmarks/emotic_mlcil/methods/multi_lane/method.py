"""Protocol-safe EMOTIC Track-A adaptation of MULTI-LANE (CoLLAs 2024)."""

from __future__ import annotations

import os
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
from .model import MultiLaneModel


@dataclass(frozen=True)
class MultiLaneOptions:
    epochs: int = 30
    source_base_learning_rate: float = 0.05
    source_reference_batch_size: int = 256
    registered_train_batch_size: int = 64
    weight_decay: float = 0.0
    num_selectors: int = 10
    num_prompts: int = 10
    num_prompt_layers: int = 5
    normalize: str = "pre-head"
    temperature: float = 1.0
    amp: bool = True
    tf32: bool = True

    @property
    def learning_rate(self) -> float:
        return self.source_base_learning_rate * (
            self.registered_train_batch_size / self.source_reference_batch_size
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MultiLaneOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError(
                "Unknown MULTI-LANE option(s): " + ", ".join(unknown)
            )
        options = cls(**{key: value[key] for key in value})
        integer_values = (
            options.epochs,
            options.source_reference_batch_size,
            options.registered_train_batch_size,
            options.num_selectors,
            options.num_prompts,
        )
        if any(int(value) <= 0 for value in integer_values):
            raise ValueError("MULTI-LANE counts, epochs, and batch sizes must be positive")
        if options.num_prompt_layers < 0:
            raise ValueError("MULTI-LANE prompt-layer count cannot be negative")
        if options.source_base_learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("Invalid MULTI-LANE optimizer options")
        if options.normalize not in {"none", "pre-head"}:
            raise ValueError("MULTI-LANE normalize must be none or pre-head")
        if options.temperature <= 0:
            raise ValueError("MULTI-LANE temperature must be positive")
        return options


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id:
        raise ValueError("Task context protocol_id differs from protocol")
    if context.protocol_hash != protocol.protocol_hash:
        raise ValueError("Task context protocol_hash differs from protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("Task context class order differs from protocol")
    if context.track != "A":
        raise ValueError("MULTI-LANE CLIP adaptation supports Track A only")
    previous = tuple(
        index
        for task_id in range(context.task_id)
        for index in protocol.current_class_indices(task_id)
    )
    if context.seen_class_indices != previous + context.current_class_indices:
        raise ValueError("MULTI-LANE requires class expansion in protocol order")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError(
            "MULTI-LANE training requires protocol-safe TrainBatch values"
        )
    current = len(context.current_class_indices)
    if batch.targets_current.ndim != 2 or batch.targets_current.shape[1] != current:
        raise ValueError("MULTI-LANE current targets do not match the task")
    expected_shape = (batch.images.shape[0], len(context.class_order))
    if batch.visible_mask.shape != expected_shape:
        raise ValueError("MULTI-LANE visible mask shape differs from the protocol")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError(
            "MULTI-LANE training batch exposes labels outside current classes"
        )
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("MULTI-LANE batch IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("MULTI-LANE prediction requires EvaluationBatch values")
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("MULTI-LANE evaluation class order differs")
    if (
        batch.targets_seen.ndim != 2
        or batch.targets_seen.shape[1] != len(context.seen_class_indices)
    ):
        raise ValueError("MULTI-LANE evaluation targets do not match seen classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("MULTI-LANE evaluation IDs and images are not aligned")
    return batch


class MultiLaneBenchmarkMethod(BenchmarkMethod):
    method_name = "MULTI-LANE"
    method_family = "Native MLCIL / Patch Selectors / Prompt Lanes"
    backbone = "OpenAI CLIP ViT-B/16 visual encoder (frozen)"
    supported_tracks = ("A",)
    upstream_repository = "https://github.com/tdemin16/multi-lane"
    upstream_commit = "5ee982c9298d4cfd6af471d9bb2ef3c0aad05373"
    upstream_archive_sha256 = (
        "dfe84ea31f6d7e51877c2661791c716aed7d888b8d783d16ce067cb8ce022d49"
    )

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        visual_encoder: Optional[torch.nn.Module] = None,
        model: Optional[MultiLaneModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("MULTI-LANE CLIP adaptation supports Track A only")
        configured = dict(protocol.method_options("multi_lane"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = MultiLaneOptions.from_mapping(configured)
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
        if model is not None and visual_encoder is not None:
            raise ValueError("Inject either MULTI-LANE model or visual encoder, not both")
        if model is None:
            if visual_encoder is None:
                visual_encoder = self._load_clip_visual()
            task_sizes = tuple(
                len(protocol.current_class_indices(task_id))
                for task_id in range(protocol.num_tasks)
            )
            model = MultiLaneModel(
                visual_encoder=visual_encoder,
                task_sizes=task_sizes,
                num_selectors=self.options.num_selectors,
                num_prompts=self.options.num_prompts,
                num_prompt_layers=self.options.num_prompt_layers,
                normalize=self.options.normalize,
            )
        self._validate_model(model)
        self.model = model.float().to(self.device)
        self.model.visual_encoder.requires_grad_(False)
        self.model.assert_visual_frozen()
        self.task_context: Optional[TaskContext] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names = self.model.optimizer_parameter_names()
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

    def _load_clip_visual(self) -> torch.nn.Module:
        model_path = os.path.abspath(os.path.expanduser(self.clip_model_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CLIP model not found: {model_path}")
        from clip import clip

        clip_model, _ = clip.load(model_path, device="cpu", jit=False)
        return clip_model.visual.float()

    def _validate_model(self, model: MultiLaneModel) -> None:
        expected_sizes = tuple(
            len(self.protocol.current_class_indices(task_id))
            for task_id in range(self.protocol.num_tasks)
        )
        if model.task_sizes != expected_sizes:
            raise ValueError("MULTI-LANE model task sizes differ from protocol")
        if model.num_selectors != self.options.num_selectors:
            raise ValueError("MULTI-LANE model selector count differs")
        if model.num_prompts != self.options.num_prompts:
            raise ValueError("MULTI-LANE model prompt count differs")
        if model.num_prompt_layers != self.options.num_prompt_layers:
            raise ValueError("MULTI-LANE model prompt-layer count differs")
        if model.normalize != self.options.normalize:
            raise ValueError("MULTI-LANE model normalization differs")

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def _images(self, images: torch.Tensor) -> torch.Tensor:
        return images.to(self.device, non_blocking=True).float()

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        expected = self._completed_task_id + 1
        if task_context.task_id != expected:
            raise RuntimeError(
                f"MULTI-LANE requires sequential tasks: expected {expected}, "
                f"got {task_context.task_id}"
            )
        self.model.activate_task(task_context.task_id)
        self.task_context = task_context
        self.training_history = []
        self.model.assert_visual_frozen()

    def _optimizer(self) -> torch.optim.Optimizer:
        task_parameters = [self.model.selectors, *list(self.model.prompts)]
        head_parameters = list(self.model.head.parameters())
        if not task_parameters or not head_parameters:
            raise RuntimeError("MULTI-LANE has no optimizer-updated parameters")
        return torch.optim.Adam(
            [
                {
                    "params": task_parameters,
                    "weight_decay": self.options.weight_decay,
                },
                {"params": head_parameters, "weight_decay": 0.0},
            ],
            lr=self.options.learning_rate,
        )

    def _selection_map(self, val_loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("MULTI-LANE validation requires an active task")
        self.model.eval()
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                with self._autocast():
                    logits = self.model.current_logits(self._images(batch.images))
                scores.append(torch.sigmoid(logits.float()).cpu())
                targets.append(batch.targets_current.float().cpu())
        if not scores:
            raise ValueError("MULTI-LANE validation loader produced no samples")
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
            raise RuntimeError("begin_task must be called before MULTI-LANE training")
        try:
            steps_per_epoch = len(train_loader)
        except TypeError as exc:
            raise TypeError(
                "MULTI-LANE training loader must define its length"
            ) from exc
        if steps_per_epoch <= 0:
            raise ValueError("MULTI-LANE training loader produced no data")
        loader_batch_size = getattr(train_loader, "batch_size", None)
        if (
            loader_batch_size is not None
            and int(loader_batch_size) != self.options.registered_train_batch_size
        ):
            raise ValueError(
                "MULTI-LANE requires registered train batch size "
                f"{self.options.registered_train_batch_size}, got "
                f"{loader_batch_size}"
            )
        optimizer = self._optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.options.epochs
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        current_indices = list(self.task_context.current_class_indices)
        hidden_indices = [
            index
            for index in range(self.protocol.num_classes)
            if index not in self.task_context.current_class_indices
        ]
        hidden_index_tensor = torch.tensor(
            hidden_indices, device=self.device, dtype=torch.long
        )
        for epoch in range(self.options.epochs):
            self.model.train()
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            loss_total = 0.0
            batches = 0
            optimizer_steps = 0
            skipped_optimizer_steps = 0
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                targets = batch.targets_current.to(
                    self.device, non_blocking=True
                ).float()
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model.current_all_logits(
                        self._images(batch.images)
                    )
                    full_targets = torch.zeros_like(logits, dtype=torch.float32)
                    full_targets[:, current_indices] = targets
                    masked_logits = logits.index_fill(
                        1,
                        hidden_index_tensor,
                        0.0,
                    )
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits.float() / self.options.temperature,
                        full_targets,
                    )
                scale_before_step = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = float(scaler.get_scale()) >= scale_before_step
                if optimizer_stepped:
                    optimizer_steps += 1
                else:
                    skipped_optimizer_steps += 1
                loss_total += float(loss.detach().cpu())
                batches += 1
            if batches == 0:
                raise ValueError("MULTI-LANE training loader produced no batches")
            if optimizer_steps:
                scheduler.step()
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "current_loss": loss_total / batches,
                    "optimizer_steps": float(optimizer_steps),
                    "skipped_optimizer_steps": float(skipped_optimizer_steps),
                    "learning_rate": epoch_learning_rate,
                    "next_learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )
        validation_map = self._selection_map(val_loader)
        self.training_history[-1]["validation_current_mAP"] = validation_map
        self._completed_task_id = self.task_context.task_id
        self.model.assert_visual_frozen()

    def predict_scores(
        self, data_loader: Iterable[EvaluationBatch]
    ) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede MULTI-LANE prediction")
        self.model.eval()
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
                    raise ValueError(
                        "MULTI-LANE evaluation loader mixes split hashes"
                    )
                with self._autocast():
                    logits = self.model.seen_logits(self._images(batch.images))
                scores.append(
                    torch.sigmoid(logits.float() / self.options.temperature).cpu()
                )
                targets.append(batch.targets_seen.float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("MULTI-LANE evaluation loader produced no samples")
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
        task_lane_parameters = self.model.selectors.numel() + sum(
            parameter.numel() for parameter in self.model.prompts
        )
        shared_classifier_parameters = sum(
            parameter.numel() for parameter in self.model.head.parameters()
        )
        return {
            "strategy": "multi_lane",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_archive_sha256": self.upstream_archive_sha256,
            "upstream_license": "CC-BY-NC-4.0",
            "visual_encoder_trainable": False,
            "clip_text_encoder_used": False,
            "benchmark_added_adapter": False,
            "replay_enabled": False,
            "task_lane_capacity_preallocated": True,
            "task_lane_parameters_total": task_lane_parameters,
            "task_lane_parameters_per_task": (
                task_lane_parameters // self.protocol.num_tasks
            ),
            "shared_classifier_parameters": shared_classifier_parameters,
            "physical_parameter_growth_after_initialization": 0,
            "retained_multi_lane_components": [
                "task_specific_patch_selectors",
                "first_layers_task_specific_key_value_prompts",
                "drop_and_replace",
                "previous_task_slice_initialization",
                "shared_linear_classifier",
                "concat_inference_without_task_oracle",
            ],
            "backbone_substitution": (
                "frozen ImageNet ViT-B/16 -> frozen OpenAI CLIP ViT-B/16"
            ),
            "image_pathway_gradient": "disabled_as_released",
            "training_label_scope": "current_classes_only",
            "training_loss_reduction": (
                "released full-class BCE after zero-filling non-current logits "
                "and targets"
            ),
            "inference_lane_scope": "all_seen_lanes_concat",
            "task_oracle_at_inference": False,
            "optimizer_lifecycle": "Adam reset at every task boundary",
            "scheduler": "CosineAnnealingLR reset at every task boundary",
            "selection_policy": "fixed_30_epochs_validation_monitoring_only",
            "source_learning_rate_scaling": (
                "0.05 * registered_train_batch_size / 256"
            ),
            "effective_learning_rate": self.options.learning_rate,
            "old_future_ground_truth_used_for_training": False,
            **asdict(self.options),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active MULTI-LANE task is required")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "method": self.method_name,
                "protocol_id": self.protocol.protocol_id,
                "protocol_hash": self.protocol.protocol_hash,
                "class_order_hash": self.protocol.class_order_hash,
                "upstream_commit": self.upstream_commit,
                "completed_task_id": self._completed_task_id,
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.model.state_dict().items()
                },
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
            raise ValueError("MULTI-LANE checkpoint must be a mapping")
        expected = {
            "schema_version": 1,
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
            "upstream_commit": self.upstream_commit,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"MULTI-LANE checkpoint {key} differs")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("MULTI-LANE checkpoint options differ")
        completed = int(payload["completed_task_id"])
        if not 0 <= completed < self.protocol.num_tasks:
            raise ValueError("MULTI-LANE checkpoint completed task is invalid")
        if self._completed_task_id >= 0 or self.model.current_task_id >= 0:
            raise RuntimeError("Load MULTI-LANE checkpoint into a fresh method")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.restore_task(completed)
        self.model.to(self.device)
        self.model.visual_encoder.requires_grad_(False)
        self.model.assert_visual_frozen()
        self.training_history = [
            {str(key): float(value) for key, value in dict(row).items()}
            for row in payload.get("training_history", [])
        ]
        self._completed_task_id = completed

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        named = dict(self.model.named_parameters())
        trainable = sum(
            named[name].numel()
            for name in self._optimizer_parameter_names
            if name in named
        )
        # The released implementation preallocates all task slices and the full
        # classifier at initialization, so no parameters are added at boundaries.
        per_task = {task_id: 0 for task_id in range(self.protocol.num_tasks)}
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


register_method("multi_lane", MultiLaneBenchmarkMethod)
