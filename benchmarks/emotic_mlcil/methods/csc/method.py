"""Protocol-safe EMOTIC Track-A adaptation of CSC (ECCV 2024)."""

from __future__ import annotations

import copy
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
from .model import CSCModel, CLIPVisualPatchEncoder


@dataclass(frozen=True)
class CSCOptions:
    token_dim: int = 512
    graph_dim: int = 1024
    epochs: int = 20
    learning_rate: float = 4.0e-5
    weight_decay: float = 1.0e-4
    one_cycle_pct_start: float = 0.2
    alpha: float = 0.5
    entropy_strength: float = 0.04
    amp: bool = True
    tf32: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CSCOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown CSC option(s): " + ", ".join(unknown))
        options = cls(**{key: value[key] for key in value})
        if options.token_dim <= 0 or options.graph_dim <= 0 or options.epochs <= 0:
            raise ValueError("CSC dimensions and epochs must be positive")
        if options.learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("Invalid CSC optimizer options")
        if not 0 < options.one_cycle_pct_start < 1:
            raise ValueError("CSC one_cycle_pct_start must lie in (0, 1)")
        if not 0 <= options.alpha <= 1:
            raise ValueError("CSC alpha must lie in [0, 1]")
        if options.entropy_strength < 0:
            raise ValueError("CSC entropy strength cannot be negative")
        return options


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id:
        raise ValueError("Task context protocol_id differs from protocol")
    if context.protocol_hash != protocol.protocol_hash:
        raise ValueError("Task context protocol_hash differs from protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("Task context class order differs from protocol")
    if context.track != "A":
        raise ValueError("CSC CLIP adaptation supports Track A only")
    previous = tuple(
        index
        for task_id in range(context.task_id)
        for index in protocol.current_class_indices(task_id)
    )
    if context.seen_class_indices != previous + context.current_class_indices:
        raise ValueError("CSC requires class expansion in protocol order")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("CSC training requires protocol-safe TrainBatch values")
    current = len(context.current_class_indices)
    if batch.targets_current.ndim != 2 or batch.targets_current.shape[1] != current:
        raise ValueError("CSC current targets do not match the task")
    if batch.visible_mask.shape != (
        batch.images.shape[0],
        len(context.class_order),
    ):
        raise ValueError("CSC visible mask shape differs from the protocol")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("CSC training batch exposes labels outside current classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("CSC batch IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("CSC prediction requires EvaluationBatch values")
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("CSC evaluation class order differs")
    if (
        batch.targets_seen.ndim != 2
        or batch.targets_seen.shape[1] != len(context.seen_class_indices)
    ):
        raise ValueError("CSC evaluation targets do not match seen classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("CSC evaluation IDs and images are not aligned")
    return batch


class CSCBenchmarkMethod(BenchmarkMethod):
    method_name = "CSC"
    method_family = "Native MLCIL / CI-GCN / Confidence Calibration"
    backbone = "OpenAI CLIP ViT-B/16 visual encoder (fine-tuned)"
    supported_tracks = ("A",)
    upstream_repository = "https://github.com/Kaile-Du/CSC"
    upstream_commit = "0bab38a00d6e0555f2df855ae2fe8db1fea68b12"
    upstream_archive_sha256 = (
        "588a098a1c7f3d813dee7df777c283fd27768a08a125d4e60b11b2d6ebdb7faa"
    )

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        token_encoder: Optional[torch.nn.Module] = None,
        model: Optional[CSCModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("CSC CLIP adaptation supports Track A only")
        configured = dict(protocol.method_options("csc"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = CSCOptions.from_mapping(configured)
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
        if model is not None and token_encoder is not None:
            raise ValueError("Inject either CSC model or token encoder, not both")
        if model is None:
            if token_encoder is None:
                token_encoder = self._load_clip_patch_encoder()
            model = CSCModel(
                token_encoder=token_encoder,
                token_dim=self.options.token_dim,
                graph_dim=self.options.graph_dim,
            )
        if model.token_dim != self.options.token_dim:
            raise ValueError("CSC model token_dim differs from method options")
        if model.graph_dim != self.options.graph_dim:
            raise ValueError("CSC model graph_dim differs from method options")
        self.model = model.float().to(self.device)
        self.task_context: Optional[TaskContext] = None
        self._teacher: Optional[CSCModel] = None
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

    def _load_clip_patch_encoder(self) -> CLIPVisualPatchEncoder:
        model_path = os.path.abspath(os.path.expanduser(self.clip_model_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CLIP model not found: {model_path}")
        from clip import clip

        clip_model, _ = clip.load(model_path, device="cpu", jit=False)
        visual = clip_model.visual.float()
        encoder = CLIPVisualPatchEncoder(visual)
        if encoder.output_dim != self.options.token_dim:
            raise ValueError(
                f"CLIP token width {encoder.output_dim} != configured "
                f"{self.options.token_dim}"
            )
        return encoder

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def _images(self, images: torch.Tensor) -> torch.Tensor:
        return images.to(self.device, non_blocking=True).float()

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        expected = self._completed_task_id + 1
        if task_context.task_id != expected:
            raise RuntimeError(
                f"CSC requires sequential tasks: expected {expected}, "
                f"got {task_context.task_id}"
            )
        old_classes = len(task_context.seen_class_indices) - len(
            task_context.current_class_indices
        )
        if self.model.num_classes != old_classes:
            raise RuntimeError("CSC model state does not match the task boundary")
        self._teacher = None
        if task_context.task_id > 0:
            self._teacher = copy.deepcopy(self.model).to(self.device).eval()
            self._teacher.requires_grad_(False)
        self.model.add_task(len(task_context.current_class_indices))
        self.model.to(self.device)
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        self.training_history = []

    @staticmethod
    def _entropy_regularizer(
        logits: torch.Tensor, strength: float
    ) -> torch.Tensor:
        if strength == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        probabilities = torch.sigmoid(logits.float())
        # This is -strength * H for the one-sided Shannon term used by CSC.
        return strength * torch.mean(
            torch.sum(
                probabilities * torch.log(probabilities.clamp_min(1.0e-5)),
                dim=1,
            )
        )

    def _optimizer(self) -> torch.optim.Optimizer:
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("CSC has no optimizer-updated parameters")
        return torch.optim.Adam(
            parameters,
            lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )

    def _selection_map(self, val_loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("CSC validation requires an active task")
        current_classes = len(self.task_context.current_class_indices)
        self.model.eval()
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                with self._autocast():
                    output = self.model(self._images(batch.images))
                scores.append(
                    torch.sigmoid(output["logits"][:, -current_classes:].float()).cpu()
                )
                targets.append(batch.targets_current.float().cpu())
        if not scores:
            raise ValueError("CSC validation loader produced no samples")
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
            raise RuntimeError("begin_task must be called before CSC training")
        try:
            steps_per_epoch = len(train_loader)
        except TypeError as exc:
            raise TypeError("CSC training loader must define its length") from exc
        if steps_per_epoch <= 0:
            raise ValueError("CSC training loader produced no data")
        optimizer = self._optimizer()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.options.learning_rate,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.options.one_cycle_pct_start,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        old_classes = len(self.task_context.seen_class_indices) - len(
            self.task_context.current_class_indices
        )
        for epoch in range(self.options.epochs):
            self.model.train()
            if self._teacher is not None:
                self._teacher.eval()
            current_total = 0.0
            distillation_total = 0.0
            entropy_total = 0.0
            loss_total = 0.0
            batches = 0
            optimizer_steps = 0
            skipped_optimizer_steps = 0
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._images(batch.images)
                current_targets = batch.targets_current.to(
                    self.device, non_blocking=True
                ).float()
                teacher_targets: Optional[torch.Tensor] = None
                if self._teacher is not None:
                    with torch.no_grad(), self._autocast():
                        teacher_targets = torch.sigmoid(
                            self._teacher(images)["logits"].float()
                        )
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model(images)["logits"].float()
                    current_loss = F.binary_cross_entropy_with_logits(
                        logits[:, old_classes:], current_targets
                    )
                    distillation_loss = torch.zeros(
                        (), device=self.device, dtype=logits.dtype
                    )
                    entropy_loss = torch.zeros(
                        (), device=self.device, dtype=logits.dtype
                    )
                    if teacher_targets is None:
                        loss = current_loss
                    else:
                        distillation_loss = F.binary_cross_entropy_with_logits(
                            logits[:, :old_classes], teacher_targets
                        )
                        entropy_loss = self._entropy_regularizer(
                            logits, self.options.entropy_strength
                        )
                        loss = (
                            self.options.alpha * current_loss
                            + (1.0 - self.options.alpha) * distillation_loss
                            + entropy_loss
                        )
                scale_before_step = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = float(scaler.get_scale()) >= scale_before_step
                if optimizer_stepped:
                    scheduler.step()
                    optimizer_steps += 1
                else:
                    skipped_optimizer_steps += 1
                current_total += float(current_loss.detach().cpu())
                distillation_total += float(distillation_loss.detach().cpu())
                entropy_total += float(entropy_loss.detach().cpu())
                loss_total += float(loss.detach().cpu())
                batches += 1
            if batches == 0:
                raise ValueError("CSC training loader produced no batches")
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "current_loss": current_total / batches,
                    "distillation_loss": distillation_total / batches,
                    "entropy_loss": entropy_total / batches,
                    "total_loss": loss_total / batches,
                    "optimizer_steps": float(optimizer_steps),
                    "skipped_optimizer_steps": float(skipped_optimizer_steps),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
        validation_map = self._selection_map(val_loader)
        self.training_history[-1]["validation_current_mAP"] = validation_map
        self._completed_task_id = self.task_context.task_id

    def predict_scores(
        self, data_loader: Iterable[EvaluationBatch]
    ) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede CSC prediction")
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
                    raise ValueError("CSC evaluation loader mixes split hashes")
                with self._autocast():
                    logits = self.model(self._images(batch.images))["logits"]
                scores.append(torch.sigmoid(logits.float()).cpu())
                targets.append(batch.targets_seen.float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("CSC evaluation loader produced no samples")
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
            "strategy": "csc",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_archive_sha256": self.upstream_archive_sha256,
            "upstream_license_status": "no_license_file_observed_reference_only",
            "visual_encoder_trainable": True,
            "clip_text_encoder_used": False,
            "benchmark_added_adapter": False,
            "replay_enabled": False,
            "retained_csc_components": [
                "class_activation_label_nodes",
                "general_relation_graph",
                "sample_specific_relation_graph",
                "dual_branch_logits",
                "old_model_sigmoid_distillation",
                "max_entropy_confidence_calibration",
            ],
            "backbone_substitution": (
                "TResNet-M spatial map -> CLIP ViT-B/16 patch tokens"
            ),
            "entropy_scope": "all_seen_classes_as_released",
            "branch_combination": "arithmetic_mean_as_released",
            "selection_policy": "fixed_20_epochs_validation_monitoring_only",
            "optimizer_lifecycle": (
                "rebuilt_after_each_expansion_to_update_all_registered_parameters"
            ),
            "old_future_ground_truth_used_for_training": False,
            **asdict(self.options),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active CSC task is required")
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
                "task_sizes": list(self.model.task_sizes),
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
            raise ValueError("CSC checkpoint must be a mapping")
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
                raise ValueError(f"CSC checkpoint {key} differs")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("CSC checkpoint options differ")
        completed = int(payload["completed_task_id"])
        task_sizes = tuple(int(item) for item in payload["task_sizes"])
        expected_sizes = tuple(
            len(self.protocol.current_class_indices(task_id))
            for task_id in range(completed + 1)
        )
        if task_sizes != expected_sizes:
            raise ValueError("CSC checkpoint class expansion differs from protocol")
        if self.model.num_tasks:
            raise RuntimeError("Load CSC checkpoint into an unexpanded model")
        self.model.restore_tasks(task_sizes)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self.training_history = [
            {str(key): float(value) for key, value in dict(row).items()}
            for row in payload.get("training_history", [])
        ]
        self._completed_task_id = completed

    def _growth_for_task(self, task_id: int) -> int:
        if task_id == 0:
            return 0
        old_classes = sum(
            len(self.protocol.current_class_indices(previous))
            for previous in range(task_id)
        )
        added = len(self.protocol.current_class_indices(task_id))
        matrix_growth = (old_classes + added) ** 2 - old_classes**2
        return (
            added * self.options.token_dim
            + matrix_growth  # general relation
            + added * (2 * self.options.graph_dim + 1)
            + added * (self.options.graph_dim + 1)
            + matrix_growth  # fixed diagonal identity parameter
        )

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        named = dict(self.model.named_parameters())
        trainable = sum(
            named[name].numel()
            for name in self._optimizer_parameter_names
            if name in named
        )
        per_task = {
            task_id: self._growth_for_task(task_id)
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
        return MemoryStatistics(
            replay_memory_samples=0,
            replay_memory_bytes=0,
        )


register_method("csc", CSCBenchmarkMethod)
