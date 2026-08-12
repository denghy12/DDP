"""BENet converted to current-label-only sequential fine-tuning."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from src.helper_functions.detail_report import average_precision

from ...method_base import BenchmarkMethod
from ...protocol import BenchmarkProtocol
from ...registry import register_method
from ...types import EvaluationBatch, MemoryStatistics, ParameterStatistics, PredictionOutput, TaskContext, TrainBatch
from .model import BENetBottomUpTaskHead, BENetFTModel, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY, load_official_benet_core


@dataclass(frozen=True)
class BENetFTOptions:
    epochs: int = 25
    early_stopping_patience: int = 5
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    heatmap_loss_weight: float = 1.0
    size_loss_weight: float = 0.1
    gradient_clip_norm: float = 10.0
    amp: bool = True
    tf32: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BENetFTOptions":
        unknown = sorted(set(value).difference(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown BENet-FT option(s): " + ", ".join(unknown))
        options = cls(**dict(value))
        if options.epochs <= 0 or options.early_stopping_patience <= 0:
            raise ValueError("BENet epoch settings must be positive")
        if options.learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("BENet optimizer settings are invalid")
        if options.heatmap_loss_weight < 0 or options.size_loss_weight < 0:
            raise ValueError("BENet auxiliary loss weights must be non-negative")
        if options.gradient_clip_norm <= 0:
            raise ValueError("BENet gradient clipping must be positive")
        return options


def focal_tag_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Official BENet FocalTagLoss formula, evaluated stably from logits."""

    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("BENet focal-tag inputs are not aligned")
    probabilities = torch.sigmoid(logits)
    # The fixed source explicitly promotes both masks to float64.
    positive = targets.eq(1).to(torch.float64)
    negative = targets.lt(1).to(torch.float64)
    loss = positive * torch.log(probabilities + 1.0e-6) * (1.0 - probabilities).pow(2)
    loss += negative * torch.log(1.0 - probabilities + 1.0e-6) * probabilities.pow(2)
    return -loss.sum(dim=1).mean()


def _heatmap_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = targets.eq(1).to(probabilities.dtype)
    negative = targets.lt(1).to(probabilities.dtype)
    negative_weight = (1.0 - targets).pow(4)
    positive_loss = torch.log(probabilities + 1.0e-6) * (1.0 - probabilities).pow(2) * positive
    negative_loss = torch.log(1.0 - probabilities + 1.0e-6) * probabilities.pow(2) * negative_weight * negative
    positive_count = positive.sum().clamp_min(1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / positive_count


def detection_geometry_loss(
    heatmaps: Iterable[torch.Tensor],
    sizes: Iterable[torch.Tensor],
    box_mask: torch.Tensor,
    heatmap_weight: float,
    size_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hm_total = box_mask.new_zeros(())
    size_total = box_mask.new_zeros(())
    source_h, source_w = box_mask.shape[-2:]
    for heatmap, size in zip(heatmaps, sizes):
        target_hm = torch.zeros_like(heatmap)
        target_size = torch.zeros_like(size)
        active = torch.zeros_like(heatmap)
        for index, mask in enumerate(box_mask):
            points = torch.nonzero(mask > 0.5, as_tuple=False)
            if points.numel() == 0:
                raise ValueError("BENet target-box transport mask is empty")
            y1, x1 = points.min(dim=0).values
            y2, x2 = points.max(dim=0).values + 1
            center_y = min(heatmap.shape[-2] - 1, max(0, int(round(float(y1 + y2) * 0.5 * heatmap.shape[-2] / source_h))))
            center_x = min(heatmap.shape[-1] - 1, max(0, int(round(float(x1 + x2) * 0.5 * heatmap.shape[-1] / source_w))))
            target_hm[index, 0, center_y, center_x] = 1.0
            active[index, 0, center_y, center_x] = 1.0
            target_size[index, 0, center_y, center_x] = float(y2 - y1) * heatmap.shape[-2] / source_h
            target_size[index, 1, center_y, center_x] = float(x2 - x1) * heatmap.shape[-1] / source_w
        hm_total = hm_total + _heatmap_focal_loss(heatmap, target_hm)
        expanded = active.expand_as(size)
        size_total = size_total + ((size - target_size).abs() * expanded).sum() / expanded.sum().clamp_min(1.0)
    total = float(heatmap_weight) * hm_total + float(size_weight) * size_total
    return total, hm_total, size_total


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id or context.protocol_hash != protocol.protocol_hash:
        raise ValueError("BENet task context differs from the protocol")
    if context.class_order_hash != protocol.class_order_hash:
        raise ValueError("BENet class order differs from the protocol")
    if context.track != "B":
        raise ValueError("Native-backbone BENet-FT supports Track B only")


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 11:
        raise ValueError("BENet requires benet_views [N, 11, H, W] images")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("BENet training requires protocol-safe TrainBatch values")
    _validate_images(batch.images)
    if batch.targets_current.shape != (batch.images.shape[0], len(context.current_class_indices)):
        raise ValueError("BENet current targets do not match the task")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("BENet training batch exposes labels outside current classes")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("BENet prediction requires EvaluationBatch values")
    _validate_images(batch.images)
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("BENet evaluation class order differs")
    if batch.targets_seen.shape != (batch.images.shape[0], len(context.seen_class_indices)):
        raise ValueError("BENet evaluation targets do not match seen classes")
    return batch


class BENetFTBenchmarkMethod(BenchmarkMethod):
    method_name = "BENet-FT"
    method_family = "Static EMOTIC model / Sequential Fine-Tuning"
    backbone = "Native BENet HigherHRNet-W32"
    supported_tracks = ("B",)
    upstream_repository = UPSTREAM_REPOSITORY
    upstream_commit = UPSTREAM_COMMIT
    upstream_license = "NOASSERTION (no license file at fixed commit)"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        source_root: Optional[Union[str, Path]] = None,
        pretrained_weights: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[BENetFTModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("Native-backbone BENet-FT supports Track B only")
        configured = dict(protocol.method_options("benet_ft"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = BENetFTOptions.from_mapping(configured)
        self.protocol = protocol
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._set_seed(protocol.seed)
        self._amp_enabled = self.options.amp and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.options.tf32
            torch.backends.cudnn.allow_tf32 = self.options.tf32
        self.source_root = str(Path(source_root).expanduser().resolve()) if source_root is not None else None
        self.pretrained_weights = str(Path(pretrained_weights).expanduser().resolve()) if pretrained_weights is not None else None
        if model is None:
            if self.source_root is None or self.pretrained_weights is None:
                raise FileNotFoundError("BENet-FT requires fixed external source and official HigherHRNet initialization")
            core, block_class, provenance = load_official_benet_core(Path(self.source_root), Path(self.pretrained_weights))
            model = BENetFTModel(core, block_class, provenance)
        self.model = model.float().to(self.device).requires_grad_(True)
        self.task_context: Optional[TaskContext] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names: Tuple[str, ...] = ()
        self.training_history: List[Dict[str, Any]] = []

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
            raise RuntimeError("BENet-FT tasks must be trained sequentially")
        old_classes = len(task_context.seen_class_indices) - len(task_context.current_class_indices)
        if self.model.num_classes != old_classes:
            raise RuntimeError("BENet heads do not match the task boundary")
        self.model.add_head(len(task_context.current_class_indices))
        self.model.to(self.device).requires_grad_(True)
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(name for name, value in self.model.named_parameters() if value.requires_grad)
        self.training_history = []
        print(
            json.dumps(
                {
                    "event": "benet_task_begin",
                    "task_id": task_context.task_id,
                    "current_classes": len(task_context.current_class_indices),
                    "seen_classes": len(task_context.seen_class_indices),
                    "max_epochs": self.options.epochs,
                    "early_stopping_patience": self.options.early_stopping_patience,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _selection_map(self, loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede validation")
        self.model.eval()
        scores, targets = [], []
        with torch.no_grad():
            for raw in loader:
                batch = _validate_train_batch(raw, self.task_context)
                images = batch.images.to(self.device, non_blocking=True).float()
                with self._autocast():
                    probabilities = torch.stack([
                        torch.sigmoid(self.model.current_logits(images, branch))
                        for branch in ("bu", "pc", "context")
                    ]).mean(dim=0)
                scores.append(probabilities.float().cpu())
                targets.append(batch.targets_current.detach().float().cpu())
        if not scores:
            raise ValueError("Validation loader produced no samples")
        all_scores, all_targets = torch.cat(scores), torch.cat(targets)
        return 100.0 * sum(average_precision(all_scores[:, i], all_targets[:, i]) for i in range(all_targets.shape[1])) / all_targets.shape[1]

    def train_task(self, train_loader: Iterable[TrainBatch], val_loader: Iterable[TrainBatch]) -> None:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede training")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options.learning_rate, weight_decay=self.options.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        best_map, best_state, stale = -math.inf, None, 0
        modes = ("det", "bu", "pc", "context")
        task_started = time.perf_counter()
        for epoch in range(self.options.epochs):
            epoch_started = time.perf_counter()
            try:
                total_batches: Optional[int] = len(train_loader)  # type: ignore[arg-type]
            except TypeError:
                total_batches = None
            progress_interval = max(1, math.ceil(total_batches / 10)) if total_batches else 50
            self.model.train()
            totals = {name: 0.0 for name in ("loss", "classification", "heatmap", "size")}
            branch_batches = {name: 0 for name in modes}
            optimizer_steps = skipped_steps = batches = 0
            for batch_index, raw in enumerate(train_loader):
                batch = _validate_train_batch(raw, self.task_context)
                images = batch.images.to(self.device, non_blocking=True).float()
                targets = batch.targets_current.to(self.device, non_blocking=True).float()
                branch = modes[(epoch + batch_index) % len(modes)]
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    if branch == "det":
                        heatmaps, sizes = self.model.detection_outputs(images)
                        loss, hm_loss, size_loss = detection_geometry_loss(
                            heatmaps, sizes, images[:, 9], self.options.heatmap_loss_weight, self.options.size_loss_weight
                        )
                        classification = loss.new_zeros(())
                    else:
                        classification = focal_tag_loss(self.model.current_logits(images, branch), targets)
                        loss = classification
                        hm_loss = loss.new_zeros(())
                        size_loss = loss.new_zeros(())
                old_scale = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                stepped = float(scaler.get_scale()) >= old_scale
                optimizer_steps += int(stepped)
                skipped_steps += int(not stepped)
                totals["loss"] += float(loss.detach().cpu())
                totals["classification"] += float(classification.detach().cpu())
                totals["heatmap"] += float(hm_loss.detach().cpu())
                totals["size"] += float(size_loss.detach().cpu())
                branch_batches[branch] += 1
                batches += 1
                if batches == 1 or batches % progress_interval == 0 or batches == total_batches:
                    elapsed = time.perf_counter() - epoch_started
                    print(
                        json.dumps(
                            {
                                "event": "benet_batch_progress",
                                "task_id": self.task_context.task_id,
                                "epoch": epoch,
                                "batches_completed": batches,
                                "total_batches": total_batches,
                                "epoch_fraction": (batches / total_batches) if total_batches else None,
                                "epoch_elapsed_seconds": elapsed,
                                "seconds_per_batch": elapsed / batches,
                                "estimated_epoch_remaining_seconds": (
                                    elapsed * (total_batches - batches) / batches if total_batches else None
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if batches == 0:
                raise ValueError("Training loader produced no samples")
            validation_map = self._selection_map(val_loader)
            record: Dict[str, Any] = {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss": totals["loss"] / batches,
                "focal_tag_loss": totals["classification"] / batches,
                "heatmap_loss": totals["heatmap"] / batches,
                "size_l1_loss": totals["size"] / batches,
                "validation_current_mAP": validation_map,
                "optimizer_steps": float(optimizer_steps),
                "skipped_optimizer_steps": float(skipped_steps),
                "branch_batches": dict(branch_batches),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "task_elapsed_seconds": time.perf_counter() - task_started,
            }
            self.training_history.append(record)
            if validation_map > best_map:
                best_map = validation_map
                best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            print(
                json.dumps(
                    {
                        "event": "benet_epoch_complete",
                        "task_id": self.task_context.task_id,
                        **record,
                        "best_validation_current_mAP": best_map,
                        "epochs_without_improvement": stale,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if stale >= self.options.early_stopping_patience:
                break
        if best_state is None:
            raise RuntimeError("No BENet epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        self._completed_task_id = self.task_context.task_id
        best_record = max(self.training_history, key=lambda row: float(row["validation_current_mAP"]))
        print(
            json.dumps(
                {
                    "event": "benet_task_complete",
                    "task_id": self.task_context.task_id,
                    "epochs_completed": len(self.training_history),
                    "best_epoch": int(best_record["epoch"]),
                    "best_validation_current_mAP": float(best_record["validation_current_mAP"]),
                    "task_elapsed_seconds": time.perf_counter() - task_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def predict_scores(self, data_loader: Iterable[EvaluationBatch]) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede prediction")
        if self.model.num_classes != len(self.task_context.seen_class_indices):
            raise RuntimeError("BENet model does not contain all seen classes")
        self.model.eval()
        scores, targets, sample_ids = [], [], []
        split_hash = None
        with torch.no_grad():
            for raw in data_loader:
                batch = _validate_eval_batch(raw, self.task_context)
                split_hash = batch.split_hash if split_hash is None else split_hash
                if split_hash != batch.split_hash:
                    raise ValueError("Evaluation loader contains multiple split hashes")
                with self._autocast():
                    probabilities = self.model(batch.images.to(self.device, non_blocking=True).float())
                scores.append(probabilities.float().cpu())
                targets.append(batch.targets_seen.detach().float().cpu())
                sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("Evaluation loader produced no samples")
        return PredictionOutput(torch.cat(scores), torch.cat(targets), sample_ids, self.task_context.class_order_hash, split_hash)

    def end_task(self) -> None:
        self.task_context = None

    def training_log_records(self):
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": "sequential_finetuning",
            "conversion_interface": "BENet-FT-v0.1",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_license": self.upstream_license,
            "source_root": self.source_root,
            "pretrained_weights": self.pretrained_weights,
            "source_provenance": dict(self.model.source_provenance),
            "input_mode": "benet_views",
            "prediction": "mean(sigmoid(BU-center), sigmoid(PC), sigmoid(masked-context))",
            "branch_sampling": "det/bu/person/context equal cyclic allocation",
            "current_label_only": True,
            "old_label_truth_used": False,
            "future_label_truth_used": False,
            "distillation_enabled": False,
            "replay_enabled": False,
            "ewc_enabled": False,
            "benchmark_added_adapter": False,
            "clip_visual_encoder_used": False,
            "clip_text_encoder_used": False,
            "extra_heco_data_used": False,
            "test_selection_used": False,
            "selection_metric": "current_label_validation_mAP",
            **asdict(self.options),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0:
            raise RuntimeError("A completed active task is required for checkpointing")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1,
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
            "completed_task_id": self._completed_task_id,
            "head_sizes": list(self.model.head_sizes),
            "model": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
            "options": asdict(self.options),
            "source_provenance": dict(self.model.source_provenance),
            "training_history": self.training_history,
        }, destination)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported BENet checkpoint")
        for key, value in {
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
        }.items():
            if payload.get(key) != value:
                raise ValueError(f"Checkpoint {key} differs from BENet-FT")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("Checkpoint BENet options differ")
        if dict(payload.get("source_provenance", {})) != dict(self.model.source_provenance):
            raise ValueError("Checkpoint BENet source provenance differs")
        completed = int(payload["completed_task_id"])
        expected = tuple(len(self.protocol.current_class_indices(i)) for i in range(completed + 1))
        if tuple(payload["head_sizes"]) != expected:
            raise ValueError("Checkpoint BENet heads differ from the protocol")
        self.model.restore_heads(expected)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).requires_grad_(True)
        self._completed_task_id = completed
        self.training_history = [dict(row) for row in payload.get("training_history", [])]

    def _head_parameter_count(self, classes: int) -> int:
        bu = BENetBottomUpTaskHead(classes, self.model.block_class)
        tag_head_parameters = (128 * 64 + 64) + (64 * classes + classes)
        return sum(value.numel() for value in bu.parameters()) + 2 * tag_head_parameters

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(value.numel() for value in self.model.parameters())
        parameters = dict(self.model.named_parameters())
        trainable = sum(parameters[name].numel() for name in self._optimizer_parameter_names if name in parameters)
        per_task = {task: (0 if task == 0 else self._head_parameter_count(len(self.protocol.current_class_indices(task)))) for task in range(self.protocol.num_tasks)}
        completed = max(self._completed_task_id, 0)
        return ParameterStatistics(total, trainable, sum(per_task[i] for i in range(1, min(completed + 1, self.protocol.num_tasks))), per_task)

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("benet_ft", BENetFTBenchmarkMethod)
