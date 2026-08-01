"""Protocol-safe EMOTIC adaptation of KRT (ICCV 2023)."""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

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
from .model import CLIPVisualPatchEncoder, KRTModel


@dataclass(frozen=True)
class KRTOptions:
    token_dim: int = 512
    embed_dim: int = 384
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_patch_tokens: int = 196
    use_positional_embedding: bool = True
    epochs: int = 20
    early_stopping_patience: int = 20
    base_learning_rate: float = 4.0e-5
    incremental_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    amp: bool = True
    tf32: bool = True
    pseudo_label: bool = True
    pseudo_initial_threshold: float = 0.8
    pseudo_threshold_step: float = 0.005
    pseudo_threshold_tolerance: float = 0.1
    pseudo_threshold_iterations: int = 100
    token_distillation_weight: float = 100.0
    exemplars_per_class: int = 20
    replay_batch_size: int = 64

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KRTOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown KRT option(s): " + ", ".join(unknown))
        options = cls(**{key: value[key] for key in value})
        integer_positive = (
            options.token_dim,
            options.embed_dim,
            options.num_heads,
            options.max_patch_tokens,
            options.epochs,
            options.early_stopping_patience,
            options.pseudo_threshold_iterations,
            options.exemplars_per_class,
            options.replay_batch_size,
        )
        if any(int(item) <= 0 for item in integer_positive):
            raise ValueError("KRT integer options must be positive")
        if options.embed_dim % options.num_heads:
            raise ValueError("KRT embed_dim must be divisible by num_heads")
        if options.mlp_ratio <= 0:
            raise ValueError("KRT mlp_ratio must be positive")
        if (
            options.base_learning_rate <= 0
            or options.incremental_learning_rate <= 0
            or options.weight_decay < 0
        ):
            raise ValueError("Invalid KRT optimizer options")
        if not 0 < options.pseudo_initial_threshold < 1:
            raise ValueError("KRT pseudo threshold must lie in (0, 1)")
        if (
            options.pseudo_threshold_step <= 0
            or options.pseudo_threshold_tolerance < 0
        ):
            raise ValueError("Invalid KRT dynamic pseudo-label options")
        if options.token_distillation_weight < 0:
            raise ValueError("KRT token distillation weight cannot be negative")
        return options


@dataclass
class ReplayExample:
    image: torch.Tensor
    target_seen_at_capture: torch.Tensor
    sample_id: str
    captured_task_id: int
    visible_through_task_id: int

    def byte_count(self) -> int:
        return (
            self.image.numel() * self.image.element_size()
            + self.target_seen_at_capture.numel()
            * self.target_seen_at_capture.element_size()
            + len(self.sample_id.encode("utf-8"))
            + 16
        )


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id:
        raise ValueError("Task context protocol_id differs from protocol")
    if context.protocol_hash != protocol.protocol_hash:
        raise ValueError("Task context protocol_hash differs from protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("Task context class order differs from protocol")
    if context.track != "A":
        raise ValueError("KRT CLIP adaptation supports Track A only")
    previous = tuple(
        index
        for task_id in range(context.task_id)
        for index in protocol.current_class_indices(task_id)
    )
    if context.seen_class_indices != previous + context.current_class_indices:
        raise ValueError("KRT requires task heads to follow protocol class order")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("KRT training requires protocol-safe TrainBatch values")
    current = len(context.current_class_indices)
    if batch.targets_current.ndim != 2 or batch.targets_current.shape[1] != current:
        raise ValueError("KRT current targets do not match the task")
    if batch.visible_mask.shape != (
        batch.images.shape[0],
        len(context.class_order),
    ):
        raise ValueError("KRT visible mask shape differs from the protocol")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("KRT training batch exposes labels outside current classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("KRT batch IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("KRT prediction requires EvaluationBatch values")
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("KRT evaluation class order differs")
    if (
        batch.targets_seen.ndim != 2
        or batch.targets_seen.shape[1] != len(context.seen_class_indices)
    ):
        raise ValueError("KRT evaluation targets do not match seen classes")
    return batch


class AsymmetricLoss(torch.nn.Module):
    """ASL settings used by the official KRT training entry point."""

    def __init__(
        self,
        gamma_negative: float = 4.0,
        gamma_positive: float = 0.0,
        negative_clip: float = 0.05,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.negative_clip = negative_clip
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        negative = (negative + self.negative_clip).clamp(max=1.0)
        loss = targets * torch.log(positive.clamp(min=self.epsilon))
        loss = loss + (1.0 - targets) * torch.log(
            negative.clamp(min=self.epsilon)
        )
        probability = positive * targets + negative * (1.0 - targets)
        gamma = (
            self.gamma_positive * targets
            + self.gamma_negative * (1.0 - targets)
        )
        weight = torch.pow(1.0 - probability, gamma).detach()
        return -(loss * weight).sum()


class KRTBenchmarkMethod(BenchmarkMethod):
    method_name = "KRT"
    method_family = "Native MLCIL / Replay / Token Distillation"
    backbone = "OpenAI CLIP ViT-B/16 visual encoder (fine-tuned)"
    supported_tracks = ("A",)
    upstream_repository = "https://github.com/witdsl/KRT-MLCIL"
    upstream_commit = "3f79044001edfe9ef94b729cd905a535fe8dd478"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        token_encoder: Optional[torch.nn.Module] = None,
        model: Optional[KRTModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("KRT CLIP adaptation supports Track A only")
        configured = dict(protocol.method_options("krt"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = KRTOptions.from_mapping(configured)
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
            raise ValueError("Inject either KRT model or token encoder, not both")
        if model is None:
            if token_encoder is None:
                token_encoder = self._load_clip_patch_encoder()
            model = KRTModel(
                token_encoder=token_encoder,
                token_dim=self.options.token_dim,
                embed_dim=self.options.embed_dim,
                num_heads=self.options.num_heads,
                mlp_ratio=self.options.mlp_ratio,
                max_patch_tokens=self.options.max_patch_tokens,
                use_positional_embedding=self.options.use_positional_embedding,
            )
        if model.token_dim != self.options.token_dim:
            raise ValueError("KRT model token_dim differs from method options")
        if model.embed_dim != self.options.embed_dim:
            raise ValueError("KRT model embed_dim differs from method options")
        self.model = model.float().to(self.device)
        self.task_context: Optional[TaskContext] = None
        self._teacher: Optional[KRTModel] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names: Tuple[str, ...] = ()
        self._replay_memory: List[ReplayExample] = []
        self._pseudo_threshold: Optional[float] = None
        self._pseudo_target_count = 0.0
        self._pseudo_realized_count = 0.0
        self.training_history: List[Dict[str, float]] = []
        self._criterion = AsymmetricLoss()

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
                f"KRT requires sequential tasks: expected {expected}, "
                f"got {task_context.task_id}"
            )
        old_classes = len(task_context.seen_class_indices) - len(
            task_context.current_class_indices
        )
        if self.model.num_classes != old_classes:
            raise RuntimeError("KRT model state does not match the task boundary")
        self._teacher = None
        if task_context.task_id > 0:
            self._teacher = copy.deepcopy(self.model).to(self.device).eval()
            self._teacher.requires_grad_(False)
        self.model.add_task(len(task_context.current_class_indices))
        self.model.to(self.device)
        self.model.freeze_old_task_parameters()
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        self._pseudo_threshold = None
        self._pseudo_target_count = 0.0
        self._pseudo_realized_count = 0.0
        self.training_history = []

    def _teacher_output(self, images: torch.Tensor) -> Optional[Dict[str, object]]:
        if self._teacher is None:
            return None
        with torch.no_grad():
            return self._teacher(images)

    def _full_current_targets(
        self,
        current_targets: torch.Tensor,
        teacher_output: Optional[Dict[str, object]],
    ) -> torch.Tensor:
        if self.task_context is None:
            raise RuntimeError("KRT target construction requires an active task")
        old_classes = len(self.task_context.seen_class_indices) - len(
            self.task_context.current_class_indices
        )
        if old_classes == 0:
            return current_targets
        old_targets = torch.zeros(
            current_targets.shape[0],
            old_classes,
            device=current_targets.device,
            dtype=current_targets.dtype,
        )
        if (
            self.options.pseudo_label
            and teacher_output is not None
            and self._pseudo_threshold is not None
        ):
            old_logits = teacher_output["logits"]
            old_targets = (
                torch.sigmoid(old_logits.detach().float())
                > self._pseudo_threshold
            ).to(current_targets.dtype)
        return torch.cat([old_targets, current_targets], dim=1)

    def _calibrate_pseudo_threshold(
        self, train_loader: Iterable[TrainBatch]
    ) -> None:
        if self.task_context is None or self._teacher is None:
            self._pseudo_threshold = None
            return
        old_classes = self.model.num_classes - len(
            self.task_context.current_class_indices
        )
        probabilities: List[torch.Tensor] = []
        positive_count = 0.0
        sample_count = 0
        self._teacher.eval()
        with torch.no_grad():
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._images(batch.images)
                with self._autocast():
                    output = self._teacher(images)
                probabilities.append(
                    torch.sigmoid(output["logits"].float()).cpu()
                )
                positive_count += float(batch.targets_current.sum())
                sample_count += int(batch.targets_current.shape[0])
        if not probabilities or sample_count == 0:
            raise ValueError("KRT cannot calibrate pseudo labels without training data")
        current_classes = len(self.task_context.current_class_indices)
        visible_density = positive_count / (sample_count * current_classes)
        target_count = visible_density * old_classes
        all_probabilities = torch.cat(probabilities)
        threshold = self.options.pseudo_initial_threshold
        best_threshold = threshold
        best_count = float((all_probabilities > threshold).sum()) / sample_count
        best_error = abs(best_count - target_count)
        for _ in range(self.options.pseudo_threshold_iterations):
            realized = float((all_probabilities > threshold).sum()) / sample_count
            error = abs(realized - target_count)
            if error < best_error:
                best_error = error
                best_threshold = threshold
                best_count = realized
            if error <= self.options.pseudo_threshold_tolerance:
                best_threshold = threshold
                best_count = realized
                break
            if realized > target_count:
                threshold = min(
                    0.999999,
                    threshold + self.options.pseudo_threshold_step,
                )
            else:
                threshold = max(
                    0.0001,
                    threshold - self.options.pseudo_threshold_step,
                )
        self._pseudo_threshold = float(best_threshold)
        self._pseudo_target_count = float(target_count)
        self._pseudo_realized_count = float(best_count)

    def _optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        decay: List[torch.nn.Parameter] = []
        no_decay: List[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return torch.optim.Adam(
            [
                {"params": decay, "weight_decay": self.options.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            weight_decay=0.0,
        )

    def _refresh_reappearing_memory(
        self, train_loader: Iterable[TrainBatch]
    ) -> None:
        if self.task_context is None or not self._replay_memory:
            return
        memory_by_id = {
            example.sample_id: example for example in self._replay_memory
        }
        seen_width = len(self.task_context.seen_class_indices)
        current_width = len(self.task_context.current_class_indices)
        old_width = seen_width - current_width
        for raw_batch in train_loader:
            batch = _validate_train_batch(raw_batch, self.task_context)
            for row, sample_id in enumerate(batch.sample_ids):
                example = memory_by_id.get(sample_id)
                if example is None:
                    continue
                expanded = torch.zeros(seen_width)
                retained_width = example.target_seen_at_capture.numel()
                expanded[:retained_width] = example.target_seen_at_capture
                expanded[old_width:] = batch.targets_current[row].float().cpu()
                example.target_seen_at_capture = expanded
                example.visible_through_task_id = self.task_context.task_id

    def _token_distillation(
        self,
        output: Dict[str, object],
        teacher_output: Optional[Dict[str, object]],
    ) -> torch.Tensor:
        if teacher_output is None or self.task_context is None:
            return torch.zeros((), device=self.device)
        current_tokens = output["tokens"][:-1]
        old_tokens = teacher_output["tokens"]
        current_flat = torch.cat(current_tokens, dim=1)
        old_flat = torch.cat(old_tokens, dim=1)
        raw = F.cosine_embedding_loss(
            current_flat,
            old_flat,
            torch.ones(current_flat.shape[0], device=self.device),
        )
        seen = len(self.task_context.seen_class_indices)
        current = len(self.task_context.current_class_indices)
        return (
            self.options.token_distillation_weight
            * math.sqrt(seen / current)
            * raw
        )

    def _replay_batches(self, epoch: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.task_context is None or not self._replay_memory:
            return
        generator = torch.Generator().manual_seed(
            self.protocol.seed * 1009 + self.task_context.task_id * 101 + epoch
        )
        order = torch.randperm(len(self._replay_memory), generator=generator)
        seen_classes = len(self.task_context.seen_class_indices)
        for start in range(0, len(order), self.options.replay_batch_size):
            examples = [
                self._replay_memory[int(index)]
                for index in order[start : start + self.options.replay_batch_size]
            ]
            images = torch.stack([example.image for example in examples])
            targets = torch.zeros(len(examples), seen_classes)
            for row, example in enumerate(examples):
                width = example.target_seen_at_capture.numel()
                targets[row, :width] = example.target_seen_at_capture
            yield images, targets

    def _train_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        teacher_output: Optional[Dict[str, object]] = None,
    ) -> Tuple[float, float]:
        images = self._images(images)
        targets = targets.to(self.device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        if teacher_output is None:
            teacher_output = self._teacher_output(images)
        with self._autocast():
            output = self.model(images)
            classification = self._criterion(output["logits"].float(), targets)
            token_loss = self._token_distillation(output, teacher_output)
            loss = classification + token_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return (
            float(classification.detach().cpu()),
            float(token_loss.detach().cpu()),
        )

    def _selection_map(self, val_loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("KRT validation requires an active task")
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
            raise ValueError("KRT validation loader produced no samples")
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
            raise RuntimeError("begin_task must be called before KRT training")
        self._refresh_reappearing_memory(train_loader)
        if self.options.pseudo_label and self._teacher is not None:
            self._calibrate_pseudo_threshold(train_loader)
        learning_rate = (
            self.options.base_learning_rate
            if self.task_context.task_id == 0
            else self.options.incremental_learning_rate
        )
        optimizer = self._optimizer(learning_rate)
        replay_steps = math.ceil(
            len(self._replay_memory) / self.options.replay_batch_size
        )
        try:
            current_steps = len(train_loader)
            steps_per_epoch = current_steps + replay_steps
        except TypeError as exc:
            raise TypeError("KRT training loader must define its length") from exc
        if current_steps <= 0:
            raise ValueError("KRT training loader produced no current-task data")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[learning_rate] * len(optimizer.param_groups),
            steps_per_epoch=steps_per_epoch,
            epochs=self.options.epochs,
            pct_start=0.2,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        best_map = -math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0
        for epoch in range(self.options.epochs):
            self.model.train()
            if self._teacher is not None:
                self._teacher.eval()
            classification_total = 0.0
            token_total = 0.0
            batches = 0
            replay_iterator = iter(self._replay_batches(epoch))
            replay_index = 0
            for current_step, raw_batch in enumerate(train_loader, start=1):
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._images(batch.images)
                teacher_output = self._teacher_output(images)
                targets = self._full_current_targets(
                    batch.targets_current.to(self.device).float(),
                    teacher_output,
                )
                classification, token_loss = self._train_batch(
                    images,
                    targets,
                    optimizer,
                    scaler,
                    teacher_output=teacher_output,
                )
                scheduler.step()
                classification_total += classification
                token_total += token_loss
                batches += 1
                # Approximate the official shuffled concatenated dataset without
                # materializing all transformed current-task images in memory:
                # replay-only minibatches are spread across the epoch.
                while replay_index < replay_steps:
                    replay_position = math.ceil(
                        (replay_index + 1)
                        * current_steps
                        / (replay_steps + 1)
                    )
                    if current_step < replay_position:
                        break
                    replay_images, replay_targets = next(replay_iterator)
                    replay_classification, replay_token_loss = self._train_batch(
                        replay_images,
                        replay_targets,
                        optimizer,
                        scaler,
                    )
                    scheduler.step()
                    classification_total += replay_classification
                    token_total += replay_token_loss
                    batches += 1
                    replay_index += 1
            for replay_images, replay_targets in replay_iterator:
                classification, token_loss = self._train_batch(
                    replay_images, replay_targets, optimizer, scaler
                )
                scheduler.step()
                classification_total += classification
                token_total += token_loss
                batches += 1
            if batches == 0:
                raise ValueError("KRT training loader produced no samples")
            selection_map = self._selection_map(val_loader)
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "classification_loss": classification_total / batches,
                    "token_distillation_loss": token_total / batches,
                    "selection_mAP": selection_map,
                    "pseudo_threshold": (
                        self._pseudo_threshold
                        if self._pseudo_threshold is not None
                        else -1.0
                    ),
                    "pseudo_target_count": self._pseudo_target_count,
                    "pseudo_realized_count": self._pseudo_realized_count,
                    "replay_samples": float(len(self._replay_memory)),
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
            raise RuntimeError("KRT did not produce a selectable checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        self._select_replay_examples(train_loader)
        self._completed_task_id = self.task_context.task_id

    @staticmethod
    def _herding_indices(features: torch.Tensor, count: int) -> List[int]:
        if features.shape[0] == 0 or count <= 0:
            return []
        normalized = F.normalize(features.float(), dim=1)
        mean = normalized.mean(dim=0)
        residual = mean.clone()
        selected: List[int] = []
        available = torch.ones(features.shape[0], dtype=torch.bool)
        for _ in range(min(count, features.shape[0])):
            scores = normalized @ residual
            scores[~available] = -torch.inf
            index = int(scores.argmax())
            selected.append(index)
            available[index] = False
            residual = residual + mean - normalized[index]
        return selected

    def _candidate_rows(
        self, train_loader: Iterable[TrainBatch]
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        if self.task_context is None:
            raise RuntimeError("KRT replay selection requires an active task")
        ids: List[str] = []
        features: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._images(batch.images)
                teacher_output = self._teacher_output(images)
                with self._autocast():
                    output = self.model(images)
                full_targets = self._full_current_targets(
                    batch.targets_current.to(self.device).float(),
                    teacher_output,
                )
                ids.extend(batch.sample_ids)
                features.append(output["pool_embeddings"].float().cpu())
                targets.append(full_targets.cpu())
        if not ids:
            raise ValueError("KRT cannot select replay examples without candidates")
        return ids, torch.cat(features), torch.cat(targets)

    def _select_replay_examples(
        self, train_loader: Iterable[TrainBatch]
    ) -> None:
        if self.task_context is None:
            raise RuntimeError("KRT replay selection requires an active task")
        ids, features, targets = self._candidate_rows(train_loader)
        current_width = len(self.task_context.current_class_indices)
        old_width = targets.shape[1] - current_width
        row_by_id = {sample_id: index for index, sample_id in enumerate(ids)}
        # A multi-label person can re-enter a later task. Update only the newly
        # visible current columns on its retained record; old/future truth is
        # never requested from the data boundary.
        for example in self._replay_memory:
            index = row_by_id.get(example.sample_id)
            if index is None:
                continue
            expanded = torch.zeros(targets.shape[1])
            retained_width = example.target_seen_at_capture.numel()
            expanded[:retained_width] = example.target_seen_at_capture
            expanded[old_width:] = targets[index, old_width:]
            example.target_seen_at_capture = expanded
            example.visible_through_task_id = self.task_context.task_id
        old_ids = {example.sample_id for example in self._replay_memory}
        selected_indices: List[int] = []
        for local_class in range(current_width):
            candidates = [
                index
                for index, sample_id in enumerate(ids)
                if sample_id not in old_ids
                and targets[index, old_width + local_class] > 0
            ]
            if not candidates:
                continue
            local_features = features[candidates]
            for local_index in self._herding_indices(
                local_features, self.options.exemplars_per_class
            ):
                index = candidates[local_index]
                selected_indices.append(index)
        # The official COCO path selects per class and then removes duplicate
        # image IDs without replenishing a class quota. Stable EMOTIC person IDs
        # provide the corresponding identity here.
        target_by_id: Dict[str, torch.Tensor] = {}
        for index in selected_indices:
            target_by_id.setdefault(ids[index], targets[index].clone())
        needed = set(target_by_id)
        images_by_id: Dict[str, torch.Tensor] = {}
        for raw_batch in train_loader:
            batch = _validate_train_batch(raw_batch, self.task_context)
            for row, sample_id in enumerate(batch.sample_ids):
                if sample_id in needed and sample_id not in images_by_id:
                    images_by_id[sample_id] = (
                        batch.images[row].detach().float().cpu().clone()
                    )
            if len(images_by_id) == len(needed):
                break
        missing = sorted(needed.difference(images_by_id))
        if missing:
            raise RuntimeError(
                "KRT replay candidates disappeared between loader passes: "
                + ", ".join(missing[:3])
            )
        for sample_id in target_by_id:
            self._replay_memory.append(
                ReplayExample(
                    image=images_by_id[sample_id],
                    target_seen_at_capture=target_by_id[sample_id],
                    sample_id=sample_id,
                    captured_task_id=self.task_context.task_id,
                    visible_through_task_id=self.task_context.task_id,
                )
            )

    def predict_scores(
        self, data_loader: Iterable[EvaluationBatch]
    ) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede KRT prediction")
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
                    raise ValueError("KRT evaluation loader mixes split hashes")
                with self._autocast():
                    output = self.model(self._images(batch.images))
                scores.append(torch.sigmoid(output["logits"].float()).cpu())
                targets.append(batch.targets_seen.float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("KRT evaluation loader produced no samples")
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
            "strategy": "krt",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_archive_sha256": (
                "2eec8351568358efca50bdaea5d06034dd28bd35d7d41c12e78cc0b1c915daf1"
            ),
            "visual_encoder_trainable": True,
            "clip_text_encoder_used": False,
            "benchmark_added_adapter": False,
            "retained_krt_components": [
                "dynamic_pseudo_labels",
                "task_tokens",
                "class_attention",
                "old_token_distillation",
                "per_task_heads",
                "herding_replay",
            ],
            "backbone_substitution": "TResNet-M spatial map -> CLIP ViT-B/16 patch tokens",
            "pseudo_density_source": "current-task training labels only",
            "pseudo_threshold_timeout_policy": (
                "use closest visited threshold after configured source-style steps"
            ),
            "replay_image_representation": "post-transform float32 tensor",
            "replay_batch_composition": (
                "homogeneous minibatches interleaved across each epoch"
            ),
            "replay_reappearing_sample_policy": (
                "deduplicate ID and update only newly visible current columns"
            ),
            "selection_metric": "current_label_validation_mAP",
            **asdict(self.options),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active KRT task is required")
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
                "head_sizes": list(self.model.head_sizes),
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.model.state_dict().items()
                },
                "replay_memory": [
                    {
                        "image": example.image,
                        "target_seen_at_capture": example.target_seen_at_capture,
                        "sample_id": example.sample_id,
                        "captured_task_id": example.captured_task_id,
                        "visible_through_task_id": (
                            example.visible_through_task_id
                        ),
                    }
                    for example in self._replay_memory
                ],
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
            raise ValueError("KRT checkpoint must be a mapping")
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
                raise ValueError(f"KRT checkpoint {key} differs")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("KRT checkpoint options differ")
        completed = int(payload["completed_task_id"])
        head_sizes = tuple(int(item) for item in payload["head_sizes"])
        expected_sizes = tuple(
            len(self.protocol.current_class_indices(task_id))
            for task_id in range(completed + 1)
        )
        if head_sizes != expected_sizes:
            raise ValueError("KRT checkpoint heads differ from protocol")
        if self.model.num_tasks:
            raise RuntimeError("Load KRT checkpoint into an unexpanded model")
        self.model.restore_tasks(head_sizes)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self._replay_memory = [
            ReplayExample(
                image=dict(row)["image"].float().cpu(),
                target_seen_at_capture=dict(row)[
                    "target_seen_at_capture"
                ].float().cpu(),
                sample_id=str(dict(row)["sample_id"]),
                captured_task_id=int(dict(row)["captured_task_id"]),
                visible_through_task_id=int(
                    dict(row)["visible_through_task_id"]
                ),
            )
            for row in payload.get("replay_memory", [])
        ]
        if len({row.sample_id for row in self._replay_memory}) != len(
            self._replay_memory
        ):
            raise ValueError("KRT checkpoint replay IDs are not unique")
        self.training_history = [
            {str(key): float(value) for key, value in dict(row).items()}
            for row in payload.get("training_history", [])
        ]
        self._completed_task_id = completed

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        named = dict(self.model.named_parameters())
        trainable = sum(
            named[name].numel() for name in self._optimizer_parameter_names
        )
        per_task: Dict[int, int] = {}
        for task_id in range(self.protocol.num_tasks):
            classes = len(self.protocol.current_class_indices(task_id))
            growth = (
                self.options.embed_dim
                + 2 * self.options.embed_dim
                + classes * (self.options.embed_dim + 1)
            )
            per_task[task_id] = 0 if task_id == 0 else growth
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
            replay_memory_samples=len(self._replay_memory),
            replay_memory_bytes=sum(
                example.byte_count() for example in self._replay_memory
            ),
        )


register_method("krt", KRTBenchmarkMethod)
