"""Protocol-safe independent Track-A implementation of L3A (ICML 2025)."""

from __future__ import annotations

import os
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
from .model import L3AModel


@dataclass(frozen=True)
class L3AOptions:
    feature_dim: int = 512
    hidden_dim: int = 4096
    base_epochs: int = 1
    base_learning_rate: float = 4.0e-5
    weight_decay: float = 1.0e-4
    one_cycle_pct_start: float = 0.2
    analytic_repeats: int = 1
    ridge: float = 1.0
    pseudo_label: bool = True
    pseudo_threshold: float = 0.7
    weighted_analytic: bool = True
    amp: bool = True
    tf32: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "L3AOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown L3A option(s): " + ", ".join(unknown))
        options = cls(**{key: value[key] for key in value})
        if (
            options.feature_dim <= 0
            or options.hidden_dim <= 0
            or options.base_epochs <= 0
            or options.analytic_repeats <= 0
        ):
            raise ValueError("L3A dimensions, epochs, and repeats must be positive")
        if options.base_learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("Invalid L3A optimizer options")
        if not 0 < options.one_cycle_pct_start < 1:
            raise ValueError("L3A one_cycle_pct_start must lie in (0, 1)")
        if options.ridge <= 0:
            raise ValueError("L3A ridge must be positive")
        if not 0 < options.pseudo_threshold < 1:
            raise ValueError("L3A pseudo threshold must lie in (0, 1)")
        return options


class AsymmetricLoss(torch.nn.Module):
    """The released L3A base-session ASL objective."""

    def __init__(self) -> None:
        super().__init__()
        self.gamma_negative = 4.0
        self.gamma_positive = 0.0
        self.negative_clip = 0.05
        self.epsilon = 1.0e-8

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        positive = torch.sigmoid(logits)
        negative = (1.0 - positive + self.negative_clip).clamp(max=1.0)
        log_likelihood = targets * torch.log(positive.clamp_min(self.epsilon))
        log_likelihood = log_likelihood + (1.0 - targets) * torch.log(
            negative.clamp_min(self.epsilon)
        )
        probability = positive * targets + negative * (1.0 - targets)
        gamma = (
            self.gamma_positive * targets
            + self.gamma_negative * (1.0 - targets)
        )
        weight = torch.pow(1.0 - probability, gamma).detach()
        return -(log_likelihood * weight).sum()


def weighted_analytic_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    normalized_class_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one L3A batch's A/C contributions and sample weights."""

    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError("L3A analytic features and labels must be matrices")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("L3A analytic features and labels are not aligned")
    if normalized_class_weights.ndim != 1 or (
        normalized_class_weights.shape[0] != labels.shape[1]
    ):
        raise ValueError("L3A normalized class weights do not match labels")
    positive_per_sample = labels.sum(dim=1)
    if torch.any(positive_per_sample <= 0):
        raise RuntimeError("L3A analytic labels require one positive per sample")
    omega = (labels @ normalized_class_weights) / positive_per_sample
    weighted_features = features * omega.unsqueeze(1)
    return (
        features.transpose(0, 1) @ weighted_features,
        features.transpose(0, 1) @ (labels * omega.unsqueeze(1)),
        omega,
    )


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id:
        raise ValueError("Task context protocol_id differs from protocol")
    if context.protocol_hash != protocol.protocol_hash:
        raise ValueError("Task context protocol_hash differs from protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("Task context class order differs from protocol")
    if context.track != "A":
        raise ValueError("L3A CLIP adaptation supports Track A only")
    previous = tuple(
        index
        for task_id in range(context.task_id)
        for index in protocol.current_class_indices(task_id)
    )
    if context.seen_class_indices != previous + context.current_class_indices:
        raise ValueError("L3A requires class expansion in protocol order")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("L3A training requires protocol-safe TrainBatch values")
    current = len(context.current_class_indices)
    if batch.targets_current.ndim != 2 or batch.targets_current.shape[1] != current:
        raise ValueError("L3A current targets do not match the task")
    expected_shape = (batch.images.shape[0], len(context.class_order))
    if tuple(batch.visible_mask.shape) != expected_shape:
        raise ValueError("L3A visible mask shape differs from the protocol")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("L3A training batch exposes labels outside current classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("L3A batch IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("L3A prediction requires EvaluationBatch values")
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("L3A evaluation class order differs")
    expected = len(context.seen_class_indices)
    if batch.targets_seen.ndim != 2 or batch.targets_seen.shape[1] != expected:
        raise ValueError("L3A evaluation targets do not match seen classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("L3A evaluation IDs and images are not aligned")
    return batch


class L3ABenchmarkMethod(BenchmarkMethod):
    method_name = "L3A"
    method_family = "Native MLCIL / Pseudo-Label / Analytic Classifier"
    backbone = "OpenAI CLIP ViT-B/16 visual encoder (Task-0 fine-tuned)"
    supported_tracks = ("A",)
    upstream_repository = "https://github.com/scut-zx/L3A"
    upstream_commit = "1067bbd6124a7aa96136baa555080ba9183ab2ac"
    upstream_tree = "9e6ac470600328f2a393f26e30ea764c3b50102b"
    upstream_archive_sha256 = (
        "307d721d76069a7fe95a10346be1323a842cd9624f8d55200f4cf7cd45462409"
    )

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        model: Optional[L3AModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("L3A CLIP adaptation supports Track A only")
        configured = dict(protocol.method_options("l3a"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = L3AOptions.from_mapping(configured)
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
        if model is not None and feature_extractor is not None:
            raise ValueError("Inject either L3A model or feature extractor, not both")
        if model is None:
            extractor = (
                feature_extractor
                if feature_extractor is not None
                else self._load_clip_visual_encoder()
            )
            model = L3AModel(
                visual_encoder=extractor,
                feature_dim=self.options.feature_dim,
                hidden_dim=self.options.hidden_dim,
            )
        if model.feature_dim != self.options.feature_dim:
            raise ValueError("L3A model feature_dim differs from method options")
        if model.hidden_dim != self.options.hidden_dim:
            raise ValueError("L3A model hidden_dim differs from method options")
        self.model = model.float().to(self.device)
        self.model.visual_encoder.requires_grad_(True)
        self.task_context: Optional[TaskContext] = None
        self._completed_task_id = -1
        self._analytic_a: Optional[torch.Tensor] = None
        self._analytic_c: Optional[torch.Tensor] = None
        self._class_counts: List[float] = []
        self._base_optimizer_parameter_count = 0
        self._base_optimizer_parameter_names: Tuple[str, ...] = ()
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

        clip_model, _ = clip.load(model_path, device="cpu", jit=False)
        visual = clip_model.visual.float()
        output_dim = int(getattr(visual, "output_dim", -1))
        if output_dim != self.options.feature_dim:
            raise ValueError(
                f"CLIP output width {output_dim} != configured "
                f"{self.options.feature_dim}"
            )
        return visual

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def _images(self, images: torch.Tensor) -> torch.Tensor:
        return images.to(self.device, non_blocking=True).float()

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        expected = self._completed_task_id + 1
        if task_context.task_id != expected:
            raise RuntimeError(
                f"L3A requires sequential tasks: expected {expected}, "
                f"got {task_context.task_id}"
            )
        old_classes = len(task_context.seen_class_indices) - len(
            task_context.current_class_indices
        )
        if self.model.num_classes != old_classes:
            raise RuntimeError("L3A model state does not match the task boundary")
        if task_context.task_id == 0:
            self.model.add_base_task(len(task_context.current_class_indices))
            self.model.to(self.device)
            self.model.requires_grad_(True)
            self._base_optimizer_parameter_names = tuple(
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            )
            self._base_optimizer_parameter_count = sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        else:
            self.model.expand_analytic(len(task_context.current_class_indices))
            self.model.to(self.device)
            self.model.requires_grad_(False)
        self.task_context = task_context
        self.training_history = []

    @staticmethod
    def _optimizer_groups(
        model: torch.nn.Module,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        decay = []
        no_decay = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def _train_base_gradient(self, train_loader: Iterable[TrainBatch]) -> None:
        if self.task_context is None or self.task_context.task_id != 0:
            raise RuntimeError("L3A base gradient training requires Task 0")
        try:
            steps_per_epoch = len(train_loader)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("L3A training loader must define its length") from exc
        if steps_per_epoch <= 0:
            raise ValueError("L3A training loader produced no data")
        optimizer = torch.optim.Adam(
            self._optimizer_groups(self.model, self.options.weight_decay),
            lr=self.options.base_learning_rate,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.options.base_learning_rate,
            epochs=self.options.base_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.options.one_cycle_pct_start,
        )
        criterion = AsymmetricLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        for epoch in range(self.options.base_epochs):
            self.model.train()
            loss_total = 0.0
            batches = 0
            optimizer_steps = 0
            skipped_steps = 0
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._images(batch.images)
                targets = batch.targets_current.to(
                    self.device, non_blocking=True
                ).float()
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model.base_logits(images).float()
                    loss = criterion(logits, targets)
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                stepped = float(scaler.get_scale()) >= scale_before
                if stepped:
                    scheduler.step()
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
                loss_total += float(loss.detach().cpu())
                batches += 1
            if batches == 0:
                raise ValueError("L3A base training produced no batches")
            self.training_history.append(
                {
                    "task_id": 0.0,
                    "epoch": float(epoch),
                    "base_gradient_training": 1.0,
                    "base_asl_loss": loss_total / batches,
                    "optimizer_steps": float(optimizer_steps),
                    "skipped_optimizer_steps": float(skipped_steps),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

    def _count_current_labels(
        self,
        train_loader: Iterable[TrainBatch],
    ) -> Tuple[List[float], int]:
        if self.task_context is None:
            raise RuntimeError("L3A label counting requires an active task")
        counts = torch.zeros(
            len(self.task_context.current_class_indices), dtype=torch.float64
        )
        samples = 0
        for raw_batch in train_loader:
            batch = _validate_train_batch(raw_batch, self.task_context)
            counts += batch.targets_current.detach().double().cpu().sum(dim=0)
            samples += int(batch.images.shape[0])
        if samples == 0:
            raise ValueError("L3A training loader produced no samples")
        return [float(value) for value in counts.tolist()], samples

    def _normalized_inverse_sqrt_weights(self) -> torch.Tensor:
        counts = torch.tensor(
            self._class_counts,
            device=self.device,
            dtype=torch.float64,
        )
        inverse = torch.zeros_like(counts)
        positive = counts > 0
        inverse[positive] = counts[positive].reciprocal().pow(0.5)
        denominator = inverse.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("L3A class counts cannot define analytic weights")
        return counts.numel() * inverse / denominator

    def _accumulate_analytic(
        self,
        train_loader: Iterable[TrainBatch],
        old_classes: int,
    ) -> Tuple[int, int]:
        if self.task_context is None or not self.model.analytic_ready:
            raise RuntimeError("L3A analytic accumulation requires an active model")
        if self._analytic_a is None or self._analytic_c is None:
            raise RuntimeError("L3A analytic statistics are not initialized")
        normalized = self._normalized_inverse_sqrt_weights()
        samples = 0
        pseudo_positives = 0
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.options.analytic_repeats):
                for raw_batch in train_loader:
                    batch = _validate_train_batch(raw_batch, self.task_context)
                    images = self._images(batch.images)
                    batch_size = int(images.shape[0])
                    labels = torch.zeros(
                        batch_size,
                        self.model.num_classes,
                        device=self.device,
                        dtype=torch.float64,
                    )
                    if old_classes and self.options.pseudo_label:
                        probabilities = torch.sigmoid(
                            self.model(images)[:, :old_classes].float()
                        )
                        pseudo = probabilities > self.options.pseudo_threshold
                        labels[:, :old_classes] = pseudo.double()
                        pseudo_positives += int(pseudo.sum().item())
                    labels[:, old_classes:] = batch.targets_current.to(
                        self.device, non_blocking=True
                    ).double()
                    if self.options.weighted_analytic:
                        batch_a, batch_c, _ = weighted_analytic_statistics(
                            self.model.analytic_features(images).double(),
                            labels,
                            normalized,
                        )
                    else:
                        if torch.any(labels.sum(dim=1) <= 0):
                            raise RuntimeError(
                                "L3A analytic labels require one positive per sample"
                            )
                        features = self.model.analytic_features(images).double()
                        batch_a = features.transpose(0, 1) @ features
                        batch_c = features.transpose(0, 1) @ labels
                    self._analytic_a.add_(batch_a)
                    self._analytic_c.add_(batch_c)
                    samples += batch_size
        return samples, pseudo_positives

    def _solve_analytic(self) -> None:
        if self._analytic_a is None or self._analytic_c is None:
            raise RuntimeError("L3A analytic statistics are unavailable")
        identity = torch.eye(
            self.options.hidden_dim,
            device=self.device,
            dtype=torch.float64,
        )
        # The official implementation explicitly computes inverse(A + rg I) C.
        solution = torch.linalg.inv(
            self._analytic_a + self.options.ridge * identity
        ) @ self._analytic_c
        self.model.set_analytic_solution(solution.float())

    def _selection_map(self, val_loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("L3A validation requires an active task")
        current = len(self.task_context.current_class_indices)
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                logits = self.model(self._images(batch.images))[:, -current:]
                scores.append(torch.sigmoid(logits.float()).cpu())
                targets.append(batch.targets_current.detach().float().cpu())
        if not scores:
            raise ValueError("L3A validation loader produced no samples")
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
            raise RuntimeError("begin_task must be called before L3A training")
        task_id = self.task_context.task_id
        current_counts, counted_samples = self._count_current_labels(train_loader)
        old_classes = len(self.task_context.seen_class_indices) - len(
            self.task_context.current_class_indices
        )
        if task_id == 0:
            self._train_base_gradient(train_loader)
            self.model.initialize_analytic()
            self.model.to(self.device)
            self._class_counts = current_counts
            self._analytic_a = torch.zeros(
                self.options.hidden_dim,
                self.options.hidden_dim,
                device=self.device,
                dtype=torch.float64,
            )
            self._analytic_c = torch.zeros(
                self.options.hidden_dim,
                self.model.num_classes,
                device=self.device,
                dtype=torch.float64,
            )
        else:
            if self._analytic_a is None or self._analytic_c is None:
                raise RuntimeError("L3A prior analytic state is missing")
            self._class_counts.extend(current_counts)
            self._analytic_c = torch.cat(
                [
                    self._analytic_c,
                    torch.zeros(
                        self.options.hidden_dim,
                        len(self.task_context.current_class_indices),
                        device=self.device,
                        dtype=torch.float64,
                    ),
                ],
                dim=1,
            )
        analytic_samples, pseudo_positives = self._accumulate_analytic(
            train_loader, old_classes
        )
        self._solve_analytic()
        validation_map = self._selection_map(val_loader)
        self.training_history.append(
            {
                "task_id": float(task_id),
                "epoch": float(self.options.base_epochs if task_id == 0 else 0),
                "base_gradient_training": float(task_id == 0),
                "counted_samples": float(counted_samples),
                "analytic_samples": float(analytic_samples),
                "pseudo_positive_labels": float(pseudo_positives),
                "validation_current_mAP": float(validation_map),
            }
        )
        self._completed_task_id = task_id

    def predict_scores(
        self, data_loader: Iterable[EvaluationBatch]
    ) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede L3A prediction")
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
                    raise ValueError("L3A evaluation loader mixes split hashes")
                logits = self.model(self._images(batch.images))
                scores.append(torch.sigmoid(logits.float()).cpu())
                targets.append(batch.targets_seen.detach().float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("L3A evaluation loader produced no samples")
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
        analytic_state_elements = 0
        if self._analytic_a is not None:
            analytic_state_elements += self._analytic_a.numel()
        if self._analytic_c is not None:
            analytic_state_elements += self._analytic_c.numel()
        return {
            "strategy": "l3a",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_tree": self.upstream_tree,
            "upstream_archive_sha256": self.upstream_archive_sha256,
            "upstream_license_status": "no_license_file_observed_reference_only",
            "visual_encoder_trainability": "Task 0 only",
            "clip_text_encoder_used": False,
            "benchmark_added_adapter": False,
            "replay_enabled": False,
            "retained_l3a_components": [
                "task0_asymmetric_loss_gradient_training",
                "bias_free_random_relu_expansion",
                "inverse_sqrt_class_frequency_weighting",
                "weighted_analytic_classifier",
                "positive_old_class_pseudo_labels",
                "cumulative_analytic_statistics",
            ],
            "backbone_substitution": (
                "ImageNet-21k timm ViT-B/16 -> OpenAI CLIP ViT-B/16"
            ),
            "source_configuration_mapping": "official l3a_vit_coco.yaml",
            "selection_policy": "fixed_one_epoch_validation_monitoring_only",
            "amp_skip_policy": (
                "OneCycleLR advances only when GradScaler applies optimizer step"
            ),
            "f1_threshold_mapping": "upstream 0.525 -> benchmark fixed 0.5",
            "old_future_ground_truth_used_for_training": False,
            "analytic_state_dtype": "float64",
            "analytic_state_elements": int(analytic_state_elements),
            "analytic_state_bytes": int(analytic_state_elements * 8),
            **asdict(self.options),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active L3A task is required")
        if self._analytic_a is None or self._analytic_c is None:
            raise RuntimeError("L3A analytic state is missing")
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
                "analytic_a": self._analytic_a.detach().cpu(),
                "analytic_c": self._analytic_c.detach().cpu(),
                "class_counts": list(self._class_counts),
                "base_optimizer_parameter_count": (
                    self._base_optimizer_parameter_count
                ),
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
            raise ValueError("L3A checkpoint must be a mapping")
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
                raise ValueError(f"L3A checkpoint {key} differs")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("L3A checkpoint options differ")
        completed = int(payload["completed_task_id"])
        task_sizes = tuple(int(value) for value in payload["task_sizes"])
        expected_sizes = tuple(
            len(self.protocol.current_class_indices(task_id))
            for task_id in range(completed + 1)
        )
        if task_sizes != expected_sizes:
            raise ValueError("L3A checkpoint class expansion differs from protocol")
        self.model.restore_analytic(task_sizes)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self.model.requires_grad_(False)
        analytic_a = payload["analytic_a"]
        analytic_c = payload["analytic_c"]
        if tuple(analytic_a.shape) != (
            self.options.hidden_dim,
            self.options.hidden_dim,
        ):
            raise ValueError("L3A checkpoint A shape differs")
        if tuple(analytic_c.shape) != (
            self.options.hidden_dim,
            sum(task_sizes),
        ):
            raise ValueError("L3A checkpoint C shape differs")
        self._analytic_a = analytic_a.to(self.device, dtype=torch.float64)
        self._analytic_c = analytic_c.to(self.device, dtype=torch.float64)
        self._class_counts = [float(value) for value in payload["class_counts"]]
        if len(self._class_counts) != sum(task_sizes):
            raise ValueError("L3A checkpoint class counts differ")
        self._base_optimizer_parameter_count = int(
            payload["base_optimizer_parameter_count"]
        )
        self.training_history = [
            {str(key): float(value) for key, value in dict(row).items()}
            for row in payload.get("training_history", [])
        ]
        self._completed_task_id = completed

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        per_task = {
            task_id: (
                0
                if task_id == 0
                else self.options.hidden_dim
                * len(self.protocol.current_class_indices(task_id))
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
            trainable_parameters=self._base_optimizer_parameter_count,
            incremental_parameters=incremental,
            per_task_incremental_parameters=per_task,
        )

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("l3a", L3ABenchmarkMethod)
