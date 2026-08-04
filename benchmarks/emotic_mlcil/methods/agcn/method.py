"""Protocol-safe EMOTIC Track-A adaptation of AGCN (ICME 2022)."""

from __future__ import annotations

import copy
import hashlib
import json
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
from .model import AGCNModel, CorrelationStatistics


@dataclass(frozen=True)
class AGCNOptions:
    visual_dim: int = 512
    embedding_dim: int = 300
    graph_hidden_dim: int = 1024
    epochs: int = 1
    visual_learning_rate: float = 1.0e-4
    graph_learning_rate: float = 3.0e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    classification_weight: float = 0.07
    distillation_weight: float = 0.93
    relationship_weight: float = 1.0e5
    task0_threshold: float = 0.0
    current_threshold: float = 0.4
    cross_threshold: float = 0.3
    task0_graph_scale: float = 0.28
    later_graph_scale: float = 0.25
    reverse_bayes_scale: float = 0.5
    task0_degree_exponent: float = -0.8
    later_degree_exponent: float = -0.5
    amp: bool = False
    tf32: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AGCNOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown AGCN option(s): " + ", ".join(unknown))
        options = cls(**{key: value[key] for key in value})
        if min(options.visual_dim, options.embedding_dim, options.graph_hidden_dim, options.epochs) <= 0:
            raise ValueError("AGCN dimensions and epochs must be positive")
        if options.visual_learning_rate <= 0 or options.graph_learning_rate <= 0:
            raise ValueError("AGCN learning rates must be positive")
        if options.adam_epsilon <= 0:
            raise ValueError("AGCN Adam epsilon must be positive")
        if not (0 <= options.adam_beta1 < 1 and 0 <= options.adam_beta2 < 1):
            raise ValueError("AGCN Adam betas must lie in [0, 1)")
        for name in (
            "classification_weight", "distillation_weight", "relationship_weight",
            "task0_graph_scale", "later_graph_scale", "reverse_bayes_scale",
        ):
            if getattr(options, name) < 0:
                raise ValueError(f"AGCN {name} cannot be negative")
        for name in ("task0_threshold", "current_threshold", "cross_threshold"):
            if not 0 <= getattr(options, name) <= 1:
                raise ValueError(f"AGCN {name} must lie in [0, 1]")
        return options


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id or context.protocol_hash != protocol.protocol_hash:
        raise ValueError("AGCN task context differs from the protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("AGCN class order differs from the protocol")
    if context.track != "A":
        raise ValueError("AGCN CLIP adaptation supports Track A only")
    previous = tuple(
        index
        for task_id in range(context.task_id)
        for index in protocol.current_class_indices(task_id)
    )
    if context.seen_class_indices != previous + context.current_class_indices:
        raise ValueError("AGCN requires class expansion in protocol order")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("AGCN training requires protocol-safe TrainBatch values")
    current = len(context.current_class_indices)
    if batch.targets_current.ndim != 2 or batch.targets_current.shape[1] != current:
        raise ValueError("AGCN current targets do not match the task")
    if batch.visible_mask.shape != (batch.images.shape[0], len(context.class_order)):
        raise ValueError("AGCN visible mask shape differs from the protocol")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("AGCN training batch exposes labels outside current classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("AGCN batch IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("AGCN prediction requires EvaluationBatch values")
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("AGCN evaluation class order differs")
    if batch.targets_seen.ndim != 2 or batch.targets_seen.shape[1] != len(context.seen_class_indices):
        raise ValueError("AGCN evaluation targets do not match seen classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("AGCN evaluation IDs and images are not aligned")
    return batch


class AGCNBenchmarkMethod(BenchmarkMethod):
    method_name = "AGCN"
    method_family = "Native lifelong multi-label / Augmented GCN"
    backbone = "OpenAI CLIP ViT-B/16 visual encoder (fine-tuned)"
    supported_tracks = ("A",)
    upstream_repository = "https://github.com/Kaile-Du/AGCN"
    upstream_commit = "3afe2ecbbef0051c6e841a97c369885011a683f0"
    upstream_tree = "6c25689b81d0807523108a782ee59630d25a9b40"
    upstream_archive_sha256 = "b5843d3ee0964767b48c49ba4b71fdf93c4bf954460669ed2b1cd05f9504f301"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        class_embedding_path: Union[str, Path] = "./pretrained/agcn/emotic_glove_6b_300d.json",
        device: Optional[Union[str, torch.device]] = None,
        visual_encoder: Optional[torch.nn.Module] = None,
        model: Optional[AGCNModel] = None,
        class_embeddings: Optional[torch.Tensor] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        configured = dict(protocol.method_options("agcn"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = AGCNOptions.from_mapping(configured)
        if protocol.track not in self.supported_tracks:
            raise ValueError("AGCN CLIP adaptation supports Track A only")
        self.protocol = protocol
        self.clip_model_path = str(clip_model_path)
        self.class_embedding_path = str(class_embedding_path)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._set_seed(protocol.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.options.tf32
            torch.backends.cudnn.allow_tf32 = self.options.tf32
        self._amp_enabled = self.options.amp and self.device.type == "cuda"
        if model is not None and visual_encoder is not None:
            raise ValueError("Inject either AGCN model or visual encoder, not both")
        if model is None:
            if visual_encoder is None:
                visual_encoder = self._load_clip_visual_encoder()
            model = AGCNModel(
                visual_encoder,
                visual_dim=self.options.visual_dim,
                embedding_dim=self.options.embedding_dim,
                graph_hidden_dim=self.options.graph_hidden_dim,
            )
        if model.visual_dim != self.options.visual_dim:
            raise ValueError("AGCN model visual width differs from options")
        self.model = model.float().to(self.device)
        self._class_embeddings, self._embedding_metadata = self._load_embeddings(class_embeddings)
        self.task_context: Optional[TaskContext] = None
        self._teacher: Optional[AGCNModel] = None
        self._teacher_adjacency: Optional[torch.Tensor] = None
        self._teacher_nodes: Optional[torch.Tensor] = None
        self._adjacency: Optional[torch.Tensor] = None
        self._correlation: Optional[CorrelationStatistics] = None
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

    def _load_clip_visual_encoder(self) -> torch.nn.Module:
        model_path = os.path.abspath(os.path.expanduser(self.clip_model_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CLIP model not found: {model_path}")
        from clip import clip

        clip_model, _ = clip.load(model_path, device="cpu", jit=False)
        visual = clip_model.visual.float()
        if int(getattr(visual, "output_dim", -1)) != self.options.visual_dim:
            raise ValueError("CLIP visual width differs from AGCN options")
        return visual

    def _load_embeddings(self, injected: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if injected is not None:
            values = injected.detach().cpu().float()
            metadata = {"kind": "injected_for_test", "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest()}
        else:
            path = Path(self.class_embedding_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"AGCN GloVe asset not found: {path}. Run prepare_agcn_glove_embeddings.py first."
                )
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            if payload.get("class_order") != list(self.protocol.class_order):
                raise ValueError("AGCN embedding class order differs from protocol")
            values = torch.tensor(payload.get("vectors"), dtype=torch.float32)
            metadata = {
                "kind": "external_glove_6b_300d_mapping",
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_sha256": payload.get("source_sha256"),
                "token_mapping": payload.get("token_mapping"),
            }
        expected = (self.protocol.num_classes, self.options.embedding_dim)
        if tuple(values.shape) != expected or not torch.isfinite(values).all():
            raise ValueError(f"AGCN class embeddings must be finite {expected}")
        return values.to(self.device), metadata

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def _images(self, images: torch.Tensor) -> torch.Tensor:
        return images.to(self.device, non_blocking=True).float()

    def _seen_embeddings(self) -> torch.Tensor:
        if self.task_context is None:
            raise RuntimeError("AGCN has no active task")
        return self._class_embeddings[: len(self.task_context.seen_class_indices)]

    def _build_adjacency(self) -> torch.Tensor:
        if self._correlation is None:
            raise RuntimeError("AGCN correlation state is missing")
        return self._correlation.build_adjacency(
            self._adjacency,
            task0_threshold=self.options.task0_threshold,
            current_threshold=self.options.current_threshold,
            cross_threshold=self.options.cross_threshold,
            task0_scale=self.options.task0_graph_scale,
            later_scale=self.options.later_graph_scale,
            reverse_bayes_scale=self.options.reverse_bayes_scale,
            task0_degree_exponent=self.options.task0_degree_exponent,
            later_degree_exponent=self.options.later_degree_exponent,
        ).detach()

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        expected = self._completed_task_id + 1
        if task_context.task_id != expected:
            raise RuntimeError(f"AGCN requires sequential tasks: expected {expected}, got {task_context.task_id}")
        old_classes = len(task_context.seen_class_indices) - len(task_context.current_class_indices)
        if old_classes == 0 and self._adjacency is not None:
            raise RuntimeError("AGCN Task 0 cannot start from an adjacency")
        if old_classes and (self._adjacency is None or self._adjacency.shape != (old_classes, old_classes)):
            raise RuntimeError("AGCN adjacency does not match the task boundary")
        self._teacher = None
        self._teacher_adjacency = None
        self._teacher_nodes = None
        if old_classes:
            self._teacher = copy.deepcopy(self.model).to(self.device).eval()
            self._teacher.requires_grad_(False)
            self._teacher_adjacency = self._adjacency.detach().clone()
            with torch.no_grad():
                self._teacher_nodes = self._teacher.graph_nodes(
                    self._teacher_adjacency, self._class_embeddings[:old_classes]
                ).detach()
        self.task_context = task_context
        self._correlation = CorrelationStatistics.create(
            old_classes, len(task_context.current_class_indices), device=self.device
        )
        self._optimizer_parameter_names = tuple(name for name, parameter in self.model.named_parameters() if parameter.requires_grad)
        self.training_history = []

    def _selection_map(self, val_loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None or self._adjacency is None:
            raise RuntimeError("AGCN validation requires a trained task")
        self.model.eval()
        old_classes = len(self.task_context.seen_class_indices) - len(self.task_context.current_class_indices)
        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                output = self.model(self._images(batch.images), self._adjacency, self._seen_embeddings())
                scores.append(torch.sigmoid(output["logits"][:, old_classes:].float()).cpu())
                targets.append(batch.targets_current.float().cpu())
        if not scores:
            raise ValueError("AGCN validation loader produced no samples")
        all_scores, all_targets = torch.cat(scores), torch.cat(targets)
        values = [average_precision(all_scores[:, index], all_targets[:, index]) for index in range(all_targets.shape[1])]
        return 100.0 * sum(values) / len(values)

    def train_task(self, train_loader: Iterable[TrainBatch], val_loader: Iterable[TrainBatch]) -> None:
        if self.task_context is None or self._correlation is None:
            raise RuntimeError("begin_task must precede AGCN training")
        visual_optimizer = torch.optim.Adam(
            self.model.visual_encoder.parameters(), lr=self.options.visual_learning_rate,
            betas=(self.options.adam_beta1, self.options.adam_beta2), eps=self.options.adam_epsilon,
        )
        graph_optimizer = torch.optim.Adam(
            self.model.graph.parameters(), lr=self.options.graph_learning_rate,
            betas=(self.options.adam_beta1, self.options.adam_beta2), eps=self.options.adam_epsilon,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        old_classes = len(self.task_context.seen_class_indices) - len(self.task_context.current_class_indices)
        task_samples = 0
        for epoch in range(self.options.epochs):
            self.model.train()
            current_total = distill_total = relationship_total = loss_total = 0.0
            batches = optimizer_steps = skipped_steps = 0
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._images(batch.images)
                current_targets = batch.targets_current.to(self.device, non_blocking=True).float()
                task_samples += int(current_targets.shape[0])
                teacher_sigmoid = teacher_softmax = None
                if self._teacher is not None:
                    with torch.no_grad():
                        teacher_output = self._teacher(images, self._teacher_adjacency, self._class_embeddings[:old_classes])
                        teacher_sigmoid = torch.sigmoid(teacher_output["logits"].float())
                        teacher_softmax = torch.softmax(teacher_output["logits"].float(), dim=1)
                self._correlation.update(current_targets, teacher_softmax)
                adjacency = self._build_adjacency()
                visual_optimizer.zero_grad(set_to_none=True)
                graph_optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.model(images, adjacency, self._seen_embeddings())
                    logits = output["logits"].float()
                    current_loss = F.binary_cross_entropy_with_logits(logits[:, old_classes:], current_targets)
                    distillation_loss = torch.zeros((), device=self.device)
                    relationship_loss = torch.zeros((), device=self.device)
                    if teacher_sigmoid is None:
                        loss = current_loss
                    else:
                        distillation_loss = F.binary_cross_entropy_with_logits(logits[:, :old_classes], teacher_sigmoid)
                        relationship_loss = F.mse_loss(output["graph_nodes"][:old_classes], self._teacher_nodes)
                        loss = (
                            self.options.classification_weight * current_loss
                            + self.options.distillation_weight * distillation_loss
                            + self.options.relationship_weight * relationship_loss
                        )
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.step(visual_optimizer)
                scaler.step(graph_optimizer)
                scaler.update()
                stepped = float(scaler.get_scale()) >= scale_before
                optimizer_steps += int(stepped)
                skipped_steps += int(not stepped)
                current_total += float(current_loss.detach().cpu())
                distill_total += float(distillation_loss.detach().cpu())
                relationship_total += float(relationship_loss.detach().cpu())
                loss_total += float(loss.detach().cpu())
                batches += 1
            if batches == 0:
                raise ValueError("AGCN training loader produced no batches")
            self.training_history.append({
                "epoch": float(epoch),
                "current_loss": current_total / batches,
                "distillation_loss": distill_total / batches,
                "relationship_loss": relationship_total / batches,
                "total_loss": loss_total / batches,
                "optimizer_steps": float(optimizer_steps),
                "skipped_optimizer_steps": float(skipped_steps),
            })
        self._adjacency = self._build_adjacency()
        self.training_history[-1]["validation_current_mAP"] = self._selection_map(val_loader)
        self.training_history[-1]["adjacency_nonzero"] = float(torch.count_nonzero(self._adjacency).item())
        self.training_history[-1]["adjacency_density"] = float(
            torch.count_nonzero(self._adjacency).item() / self._adjacency.numel()
        )
        self.training_history[-1]["acm_current_positive_mass"] = float(
            self._correlation.hard_positive_count.sum().detach().cpu()
        )
        self.training_history[-1]["acm_cross_soft_hard_mass"] = float(
            self._correlation.old_new_soft_hard.sum().detach().cpu()
        )
        self.training_history[-1]["acm_old_soft_mass"] = float(
            self._correlation.old_soft_sum.sum().detach().cpu()
        )
        self.training_history[-1]["acm_samples"] = float(
            self._correlation.sample_count.detach().cpu()
            if old_classes else task_samples
        )
        self._completed_task_id = self.task_context.task_id

    def predict_scores(self, data_loader: Iterable[EvaluationBatch]) -> PredictionOutput:
        if self.task_context is None or self._adjacency is None:
            raise RuntimeError("begin_task and train_task must precede AGCN prediction")
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
                    raise ValueError("AGCN evaluation loader mixes split hashes")
                output = self.model(self._images(batch.images), self._adjacency, self._seen_embeddings())
                scores.append(torch.sigmoid(output["logits"].float()).cpu())
                targets.append(batch.targets_seen.float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("AGCN evaluation loader produced no samples")
        return PredictionOutput(
            scores=torch.cat(scores), targets=torch.cat(targets), sample_ids=sample_ids,
            class_order_hash=self.task_context.class_order_hash, split_hash=split_hash,
        )

    def end_task(self) -> None:
        self.task_context = None
        self._teacher = None
        self._teacher_adjacency = None
        self._teacher_nodes = None
        self._correlation = None

    def training_log_records(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": "agcn",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_tree": self.upstream_tree,
            "upstream_archive_sha256": self.upstream_archive_sha256,
            "upstream_license": "Apache-2.0",
            "upstream_license_note": "root LICENSE is Apache-2.0; GCN.py contains a bare BSD header comment",
            "visual_encoder_trainable": True,
            "clip_text_encoder_used": False,
            "benchmark_added_adapter": False,
            "replay_enabled": False,
            "old_future_ground_truth_used_for_training": False,
            "backbone_substitution": "ImageNet-pretrained ResNet-101 -> OpenAI CLIP ViT-B/16 visual encoder",
            "retained_agcn_components": [
                "glove_6b_300d_label_nodes", "two_layer_300_1024_visualdim_gcn",
                "online_augmented_correlation_matrix", "old_model_sigmoid_distillation",
                "old_graph_node_relationship_mse", "one_epoch_per_task",
            ],
            "acm_old_soft_label_activation": "softmax_as_released_code",
            "distillation_activation": "sigmoid_as_released_code",
            "loss_weight_resolution": "paper_table_3_fills_undefined_released_a_b_c",
            "adam_epsilon_resolution": "1e-8_released_torch_default_paper_reports_1e-4",
            "selection_policy": "fixed_one_epoch_validation_monitoring_only",
            "class_embedding_asset": self._embedding_metadata,
            **asdict(self.options),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0 or self._adjacency is None:
            raise RuntimeError("A completed active AGCN task is required")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1,
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
            "upstream_commit": self.upstream_commit,
            "completed_task_id": self._completed_task_id,
            "model": {name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()},
            "adjacency": self._adjacency.detach().cpu(),
            "embedding_sha256": self._embedding_metadata["sha256"],
            "training_history": self.training_history,
            "options": asdict(self.options),
        }, destination)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        expected = {
            "schema_version": 1, "method": self.method_name,
            "protocol_id": self.protocol.protocol_id, "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash, "upstream_commit": self.upstream_commit,
            "embedding_sha256": self._embedding_metadata["sha256"],
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"AGCN checkpoint {key} differs")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("AGCN checkpoint options differ")
        completed = int(payload["completed_task_id"])
        seen = len(self.protocol.seen_class_indices(completed))
        adjacency = payload["adjacency"]
        if tuple(adjacency.shape) != (seen, seen):
            raise ValueError("AGCN checkpoint adjacency differs from protocol")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self._adjacency = adjacency.float().to(self.device)
        self.training_history = [dict(row) for row in payload.get("training_history", [])]
        self._completed_task_id = completed

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        named = dict(self.model.named_parameters())
        trainable = sum(named[name].numel() for name in self._optimizer_parameter_names if name in named)
        return ParameterStatistics(
            total_parameters=total,
            trainable_parameters=trainable,
            incremental_parameters=0,
            per_task_incremental_parameters={task_id: 0 for task_id in range(self.protocol.num_tasks)},
        )

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("agcn", AGCNBenchmarkMethod)
