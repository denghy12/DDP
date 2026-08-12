"""Official-backbone DSCT converted to current-label-only sequential FT."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from argparse import Namespace
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
from .model import DSCTFTModel, dsct_current_task_loss


UPSTREAM_REPOSITORY = "https://github.com/Sampson-Lee/DSCT"
UPSTREAM_COMMIT = "8b0fe36199693ebcd58a4cce8751c166550fc7a5"
UPSTREAM_FILES = {
    "models/deformable_detr_dsct.py": "d843ad2be18307ac62cb0e3bf032fe16c0c90ad80f0ba2ccd9637e8c4b1efc11",
    "models/deformable_transformer_dsct.py": "20d6b7997b96b96839d638275402be8b09cb581be198657b0b5c34350ab7f0a0",
    "models/matcher_dsct.py": "53ff926d246dcf28afe2887b167261cf3374d04692541754e47f620327f5b9dc",
    "datasets/emotic.py": "02d1ab0bd673b23b5ff216249b745a99a232dd3f9a419436037ed3bf639f2066",
    "run.sh": "b0960fa93e3fd02187da97cf5682908e7cb91400f1041395a0944cdc5f016bfc",
}


@dataclass(frozen=True)
class DSCTFTOptions:
    epochs: int = 50
    early_stopping_patience: int = 50
    learning_rate: float = 2.0e-4
    backbone_learning_rate: float = 2.0e-5
    linear_projection_lr_multiplier: float = 0.1
    weight_decay: float = 1.0e-4
    lr_drop_epoch: int = 40
    gradient_clip_norm: float = 0.1
    num_queries: int = 4
    hidden_dim: int = 256
    class_cost: float = 2.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    class_loss_weight: float = 5.0
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    focal_alpha: float = 0.25
    aux_loss: bool = True
    amp: bool = False
    tf32: bool = False
    effective_train_batch_size: int = 4
    per_gpu_micro_batch_size: int = 1
    maximum_data_parallel_replicas: int = 4

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DSCTFTOptions":
        unknown = sorted(set(values).difference(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown DSCT-FT option(s): " + ", ".join(unknown))
        result = cls(**dict(values))
        if min(result.epochs, result.early_stopping_patience, result.lr_drop_epoch,
               result.num_queries, result.hidden_dim, result.effective_train_batch_size,
               result.per_gpu_micro_batch_size, result.maximum_data_parallel_replicas) <= 0:
            raise ValueError("DSCT dimensions and epoch settings must be positive")
        if result.per_gpu_micro_batch_size != 1:
            raise ValueError("DSCT-FT v0.2 requires per-GPU micro-batch size 1")
        if result.effective_train_batch_size != 4:
            raise ValueError("DSCT-FT v0.2 requires effective train batch size 4")
        if result.maximum_data_parallel_replicas != 4:
            raise ValueError("DSCT-FT v0.2 requires four DataParallel replicas")
        if min(result.learning_rate, result.backbone_learning_rate,
               result.linear_projection_lr_multiplier, result.gradient_clip_norm) <= 0:
            raise ValueError("DSCT optimizer settings must be positive")
        if result.weight_decay < 0 or not 0 < result.focal_alpha < 1:
            raise ValueError("DSCT weight decay or focal alpha is invalid")
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream_source(root: Union[str, Path]) -> Dict[str, str]:
    source = Path(root).expanduser().resolve()
    observed = {}
    for relative, expected in UPSTREAM_FILES.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = _sha256(path)
        if observed[relative] != expected:
            raise ValueError(f"DSCT fixed-source hash differs: {relative}")
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=False
    )
    if git.returncode == 0 and git.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError("DSCT source checkout is not the registered commit")
    return observed


def _source_arguments(options: DSCTFTOptions, device: torch.device) -> Namespace:
    return Namespace(
        dataset_file="emotic", device=str(device), backbone="resnet50", dilation=False,
        position_embedding="sine", position_embedding_scale=2 * np.pi,
        num_feature_levels=4, enc_layers=6, dec_layers=6, dim_feedforward=1024,
        hidden_dim=options.hidden_dim, dropout=0.1, nheads=8,
        num_queries=options.num_queries, dec_n_points=4, enc_n_points=4,
        interpolate_factor=0.5, noise=0.2, model="deformable_transformer_dsct",
        detr="deformable_detr_dsct", modality="", dec_n_sp=100, dec_n_sm=50,
        masks=False, aux_loss=options.aux_loss, with_box_refine=False, two_stage=False,
        frozen_weights=None, lr_backbone=options.backbone_learning_rate,
        set_cost_class=options.class_cost, set_cost_em=5.0,
        set_cost_bbox=options.bbox_cost, set_cost_giou=options.giou_cost,
        mask_loss_coef=1.0, dice_loss_coef=1.0, em_loss_coef=5.0,
        cls_loss_coef=options.class_loss_weight, bbox_loss_coef=options.bbox_loss_weight,
        giou_loss_coef=options.giou_loss_weight, focal_alpha=options.focal_alpha,
        binary_flag=1,
    )


def build_upstream_model(
    source_root: Union[str, Path],
    pretrained_weights: Union[str, Path],
    options: DSCTFTOptions,
    device: torch.device,
) -> Tuple[DSCTFTModel, Dict[str, Any]]:
    root = Path(source_root).expanduser().resolve()
    source_hashes = verify_upstream_source(root)
    checkpoint = Path(pretrained_weights).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    existing = sys.modules.get("models")
    if existing is not None:
        module_file = Path(getattr(existing, "__file__", "")).resolve()
        if root not in module_file.parents:
            raise RuntimeError("Top-level 'models' is already imported from another repository")
    sys.path.insert(0, str(root))
    try:
        source_models = importlib.import_module("models")
        source_backbone = importlib.import_module("models.backbone")
        source_backbone.is_main_process = lambda: False
        misc = importlib.import_module("util.misc")
        core, _, _ = source_models.build_model(_source_arguments(options, device))
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError("DSCT pretraining checkpoint does not contain a model state")
    current = core.state_dict()
    compatible = {key: value for key, value in state.items()
                  if key in current and isinstance(value, torch.Tensor) and current[key].shape == value.shape}
    load_result = core.load_state_dict(compatible, strict=False)
    model = DSCTFTModel(core, misc.NestedTensor, hidden_dim=options.hidden_dim)
    provenance = {
        "source_file_sha256": source_hashes,
        "pretrained_weights_path": str(checkpoint),
        "pretrained_weights_sha256": _sha256(checkpoint),
        "pretrained_loaded_tensors": len(compatible),
        "pretrained_missing_keys": list(load_result.missing_keys),
        "pretrained_unexpected_keys": list(load_result.unexpected_keys),
    }
    return model, provenance


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_hash != protocol.protocol_hash or context.class_order_hash != protocol.class_order_hash:
        raise ValueError("DSCT task context differs from the protocol")
    if context.track != "B":
        raise ValueError("DSCT-FT supports Track B only")


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 5:
        raise ValueError("DSCT-FT requires dsct_scene [N, 5, H, W] transport")


class DSCTFTBenchmarkMethod(BenchmarkMethod):
    method_name = "DSCT-FT"
    method_family = "Static EMOTIC model / Sequential Fine-Tuning"
    backbone = "Official DSCT ResNet-50 + Deformable-DETR"
    supported_tracks = ("B",)
    upstream_repository = UPSTREAM_REPOSITORY
    upstream_commit = UPSTREAM_COMMIT
    upstream_license = "README says Apache-2.0; LICENSE absent at fixed commit"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        source_root: Optional[Union[str, Path]] = None,
        pretrained_weights: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[DSCTFTModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track != "B":
            raise ValueError("DSCT-FT supports Track B only")
        configured = dict(protocol.method_options("dsct_ft"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = DSCTFTOptions.from_mapping(configured)
        self.protocol = protocol
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._set_seed(protocol.seed)
        self._amp_enabled = self.options.amp and self.device.type == "cuda"
        self.source_root = str(Path(source_root).expanduser().resolve()) if source_root else None
        self.pretrained_weights = str(Path(pretrained_weights).expanduser().resolve()) if pretrained_weights else None
        if model is None:
            if not self.source_root or not self.pretrained_weights:
                raise FileNotFoundError("DSCT-FT requires fixed source and official Deformable-DETR R50 weights")
            model, self.provenance = build_upstream_model(
                self.source_root, self.pretrained_weights, self.options, self.device
            )
        else:
            self.provenance = {"injected_test_model": True}
        self.model = model.float().to(self.device).requires_grad_(True)
        visible_cuda_devices = torch.cuda.device_count() if self.device.type == "cuda" else 0
        self._execution_gpu_count = max(
            1, min(self.options.maximum_data_parallel_replicas, visible_cuda_devices)
        )
        self._parallel_model: Optional[torch.nn.Module] = None
        self.task_context: Optional[TaskContext] = None
        self._completed_task_id = -1
        self._optimizer_parameter_names: Tuple[str, ...] = ()
        self.training_history: List[Dict[str, float]] = []

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def begin_task(self, task_context: TaskContext) -> None:
        _validate_context(self.protocol, task_context)
        if task_context.task_id != self._completed_task_id + 1:
            raise RuntimeError("DSCT-FT tasks must be trained sequentially")
        old_classes = len(task_context.seen_class_indices) - len(task_context.current_class_indices)
        if self.model.num_classes != old_classes:
            raise RuntimeError("DSCT heads do not match the task boundary")
        self.model.add_head(len(task_context.current_class_indices))
        self.model.to(self.device).requires_grad_(True)
        self._parallel_model = (
            torch.nn.DataParallel(
                self.model,
                device_ids=list(range(self._execution_gpu_count)),
                output_device=0,
            )
            if self.device.type == "cuda" and self._execution_gpu_count > 1
            else self.model
        )
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(name for name, value in self.model.named_parameters() if value.requires_grad)
        self.training_history = []

    def _batch(self, raw: Any, evaluation: bool = False):
        expected_type = EvaluationBatch if evaluation else TrainBatch
        if not isinstance(raw, expected_type):
            raise TypeError(f"DSCT requires {expected_type.__name__}")
        _validate_images(raw.images)
        return raw

    def _loss(self, output: Mapping[str, Any], targets: torch.Tensor):
        current = targets.shape[1]
        total, parts = dsct_current_task_loss(output, targets, current, **{
            "class_cost": self.options.class_cost, "bbox_cost": self.options.bbox_cost,
            "giou_cost": self.options.giou_cost, "class_loss_weight": self.options.class_loss_weight,
            "bbox_loss_weight": self.options.bbox_loss_weight, "giou_loss_weight": self.options.giou_loss_weight,
            "focal_alpha": self.options.focal_alpha,
        })
        for auxiliary in output.get("aux_outputs", []):
            auxiliary = dict(auxiliary); auxiliary["target_boxes"] = output["target_boxes"]
            aux_total, _ = dsct_current_task_loss(auxiliary, targets, current, **{
                "class_cost": self.options.class_cost, "bbox_cost": self.options.bbox_cost,
                "giou_cost": self.options.giou_cost, "class_loss_weight": self.options.class_loss_weight,
                "bbox_loss_weight": self.options.bbox_loss_weight, "giou_loss_weight": self.options.giou_loss_weight,
                "focal_alpha": self.options.focal_alpha,
            })
            total = total + aux_total
        return total, parts

    def _forward(
        self, images: torch.Tensor, *, parallel_training: bool = False
    ) -> Mapping[str, Any]:
        if parallel_training and self._parallel_model is not None:
            return self._parallel_model(images)
        return self.model(images)

    def _parallel_micro_batch_size(self) -> int:
        return self.options.per_gpu_micro_batch_size * self._execution_gpu_count

    def _training_micro_batches(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        width = self._parallel_micro_batch_size()
        for start in range(0, images.shape[0], width):
            stop = min(images.shape[0], start + width)
            yield images[start:stop], targets[start:stop]

    def _selection_map(self, loader: Iterable[TrainBatch]) -> float:
        self.model.eval(); scores, targets = [], []
        with torch.no_grad():
            for raw in loader:
                batch = self._batch(raw)
                images = batch.images.to(self.device).float()
                with self._autocast():
                    output = self._forward(images)
                    logits = self.model.selected_scores(output, batch.targets_current.shape[1])
                scores.append(logits.sigmoid().float().cpu()); targets.append(batch.targets_current.float().cpu())
        if not scores: raise ValueError("Validation loader produced no samples")
        score, target = torch.cat(scores), torch.cat(targets)
        return 100.0 * sum(average_precision(score[:, i], target[:, i]) for i in range(target.shape[1])) / target.shape[1]

    def _optimizer(self):
        backbone, projection, regular = [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            if "backbone.0" in name: backbone.append(parameter)
            elif "reference_points" in name or "sampling_offsets" in name: projection.append(parameter)
            else: regular.append(parameter)
        groups = [{"params": regular, "lr": self.options.learning_rate}]
        if backbone: groups.append({"params": backbone, "lr": self.options.backbone_learning_rate})
        if projection: groups.append({"params": projection, "lr": self.options.learning_rate * self.options.linear_projection_lr_multiplier})
        return torch.optim.AdamW(groups, lr=self.options.learning_rate, weight_decay=self.options.weight_decay)

    def train_task(self, train_loader: Iterable[TrainBatch], val_loader: Iterable[TrainBatch]) -> None:
        if self.task_context is None: raise RuntimeError("begin_task must precede training")
        optimizer = self._optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.options.lr_drop_epoch)
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        best_map, best_state, stale = -math.inf, None, 0
        task_started = time.monotonic()
        for epoch in range(self.options.epochs):
            epoch_started = time.monotonic()
            self.model.train(); sums = {"loss": 0.0, "classification": 0.0, "bbox": 0.0, "giou": 0.0}; batches = steps = skips = 0
            micro_batches = 0
            for raw in train_loader:
                batch = self._batch(raw)
                expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
                expected[:, list(self.task_context.current_class_indices)] = True
                if not torch.equal(batch.visible_mask.bool(), expected):
                    raise ValueError("DSCT training batch exposes labels outside current classes")
                if batch.images.shape[0] > self.options.effective_train_batch_size:
                    raise ValueError("DSCT loader batch exceeds the frozen effective batch size")
                images = batch.images; targets = batch.targets_current
                optimizer.zero_grad(set_to_none=True)
                batch_size = int(images.shape[0])
                batch_sums = {"loss": 0.0, "classification": 0.0, "bbox": 0.0, "giou": 0.0}
                for micro_images, micro_targets in self._training_micro_batches(images, targets):
                    micro_images = micro_images.to(self.device).float()
                    micro_targets = micro_targets.to(self.device).float()
                    weight = float(micro_images.shape[0]) / float(batch_size)
                    with self._autocast():
                        output = self._forward(micro_images, parallel_training=True)
                        loss, parts = self._loss(output, micro_targets)
                    scaler.scale(loss * weight).backward()
                    batch_sums["loss"] += float(loss.detach().cpu()) * weight
                    for key in ("classification", "bbox", "giou"):
                        batch_sums[key] += float(parts[key].detach().cpu()) * weight
                    micro_batches += 1
                    del output, loss, parts, micro_images, micro_targets
                old_scale = float(scaler.get_scale()); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options.gradient_clip_norm)
                scaler.step(optimizer); scaler.update(); stepped = float(scaler.get_scale()) >= old_scale
                steps += int(stepped); skips += int(not stepped); batches += 1
                for key in sums: sums[key] += batch_sums[key]
            if not batches: raise ValueError("Training loader produced no samples")
            validation_map = self._selection_map(val_loader)
            row = {"epoch": float(epoch), "validation_current_mAP": validation_map,
                   "optimizer_steps": float(steps), "skipped_optimizer_steps": float(skips),
                   "micro_batches": float(micro_batches),
                   "execution_gpu_count": float(self._execution_gpu_count),
                   "per_gpu_micro_batch_size": float(self.options.per_gpu_micro_batch_size),
                   "effective_train_batch_size": float(self.options.effective_train_batch_size),
                   "learning_rate": float(optimizer.param_groups[0]["lr"])}
            row.update({key: value / batches for key, value in sums.items()}); self.training_history.append(row)
            if validation_map > best_map:
                best_map = validation_map; stale = 0
                best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
            else: stale += 1
            epoch_seconds = time.monotonic() - epoch_started
            task_elapsed_seconds = time.monotonic() - task_started
            task_eta_seconds = (
                task_elapsed_seconds / float(epoch + 1) * float(self.options.epochs - epoch - 1)
            )
            row.update({"epoch_seconds": epoch_seconds,
                        "task_elapsed_seconds": task_elapsed_seconds,
                        "task_eta_seconds": task_eta_seconds})
            print("DSCT_PROGRESS " + json.dumps({
                "task": self.task_context.task_id,
                "epoch_number": epoch + 1,
                "epochs_planned": self.options.epochs,
                **row,
                "best_validation_current_mAP": best_map,
                "stale_epochs": stale,
            }, sort_keys=True), flush=True)
            scheduler.step()
            if stale >= self.options.early_stopping_patience: break
        if best_state is None: raise RuntimeError("No DSCT epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True); self.model.to(self.device)
        self._completed_task_id = self.task_context.task_id

    def predict_scores(self, data_loader: Iterable[EvaluationBatch]) -> PredictionOutput:
        if self.task_context is None: raise RuntimeError("begin_task must precede prediction")
        self.model.eval(); scores, targets, ids = [], [], []; split_hash = None
        with torch.no_grad():
            for raw in data_loader:
                batch = self._batch(raw, evaluation=True)
                if batch.class_order_hash != self.task_context.class_order_hash: raise ValueError("DSCT evaluation class order differs")
                if split_hash is None: split_hash = batch.split_hash
                elif split_hash != batch.split_hash: raise ValueError("Evaluation loader mixes split hashes")
                with self._autocast(): output = self._forward(batch.images.to(self.device).float()); logits = self.model.selected_scores(output)
                scores.append(logits.sigmoid().float().cpu()); targets.append(batch.targets_seen.float().cpu()); ids.extend(batch.sample_ids)
        if not scores or split_hash is None: raise ValueError("Evaluation loader produced no samples")
        return PredictionOutput(torch.cat(scores), torch.cat(targets), ids, self.task_context.class_order_hash, split_hash)

    def end_task(self) -> None:
        self._parallel_model = None
        self.task_context = None

    def training_log_records(self):
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": "sequential_finetuning", "conversion_interface": "DSCT-FT-v0.2",
            "upstream_repository": self.upstream_repository, "upstream_commit": self.upstream_commit,
            "upstream_license": self.upstream_license, "track": "B", "input_mode": "dsct_scene",
            "current_label_only": True, "old_label_truth_used": False, "future_label_truth_used": False,
            "replay_enabled": False, "distillation_enabled": False, "ewc_enabled": False,
            "benchmark_added_adapter": False, "clip_visual_encoder_used": False,
            "query_selection": "maximum IoU to benchmark target-person bbox (official face_matching)",
            "augmentation": "official horizontal flip and multi-scale resize; target-dropping random crop disabled",
            "checkpoint_selection": "current-label validation mAP; earliest epoch tie-break; test forbidden",
            "execution_mode": "data_parallel_micro_batch_then_accumulate",
            "execution_gpu_count": self._execution_gpu_count,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "parallel_micro_batch_size": self._parallel_micro_batch_size(),
            "optimizer_steps_per_effective_batch": 1,
            **asdict(self.options), **self.provenance,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self.task_context is None or self._completed_task_id < 0: raise RuntimeError("Completed active task required")
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, "protocol_hash": self.protocol.protocol_hash,
                    "class_order_hash": self.protocol.class_order_hash, "options": asdict(self.options),
                    "completed_task_id": self._completed_task_id, "head_sizes": self.model.head_sizes,
                    "model": self.model.state_dict(), "provenance": self.provenance,
                    "training_history": self.training_history}, destination)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = torch.load(path, map_location="cpu")
        if payload.get("protocol_hash") != self.protocol.protocol_hash or payload.get("options") != asdict(self.options):
            raise ValueError("DSCT checkpoint protocol or options differ")
        completed = int(payload["completed_task_id"])
        expected = tuple(len(self.protocol.current_class_indices(task)) for task in range(completed + 1))
        if tuple(payload["head_sizes"]) != expected: raise ValueError("DSCT checkpoint heads differ")
        if self.model.num_classes: raise RuntimeError("Checkpoint loading requires empty heads")
        self.model.restore_heads(expected); self.model.load_state_dict(payload["model"], strict=True); self.model.to(self.device)
        self._completed_task_id = completed; self.training_history = [dict(row) for row in payload.get("training_history", [])]

    def parameter_statistics(self) -> ParameterStatistics:
        parameters = dict(self.model.named_parameters()); total = sum(value.numel() for value in parameters.values())
        trainable = sum(parameters[name].numel() for name in self._optimizer_parameter_names if name in parameters)
        per_task = {task: (0 if task == 0 else len(self.protocol.current_class_indices(task)) * (self.options.hidden_dim + 1)) for task in range(self.protocol.num_tasks)}
        incremental = sum(per_task[task] for task in range(1, min(max(self._completed_task_id, 0) + 1, self.protocol.num_tasks)))
        return ParameterStatistics(total, trainable, incremental, per_task)

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(0, 0)


register_method("dsct_ft", DSCTFTBenchmarkMethod)
