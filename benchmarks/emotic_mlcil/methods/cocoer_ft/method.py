"""Native CocoER converted to a no-memory sequential fine-tuning baseline."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

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
from .model import CocoERFTModel, NativeResNet50GridEncoder, dynamic_bce


UPSTREAM_COMMIT = "dac8fc139e61b87f1bf0b27c581798df2a5a9d38"
OPENAI_CLIP_RN50_SHA256 = (
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CocoERFTOptions:
    epochs: int = 20
    early_stopping_patience: int = 20
    learning_rate: float = 6.0e-5
    lr_step_epochs: int = 3
    lr_gamma: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.96
    weight_decay: float = 0.01
    inside_lr: float = 0.1
    pseudo_threshold: float = 0.3
    grad_distance_weight: float = 0.1
    gradient_clip_norm: float = 10.0
    encoder_blocks: int = 3
    amp: bool = True
    tf32: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CocoERFTOptions":
        unknown = sorted(set(value).difference(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown CocoER-FT option(s): " + ", ".join(unknown))
        options = cls(**{key: value[key] for key in value})
        if min(options.epochs, options.early_stopping_patience, options.lr_step_epochs,
               options.encoder_blocks) <= 0:
            raise ValueError("CocoER epoch/block settings must be positive")
        if options.learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("CocoER optimizer settings are invalid")
        if not (0 <= options.beta1 < 1 and 0 <= options.beta2 < 1):
            raise ValueError("CocoER AdamW betas must lie in [0,1)")
        if not 0 < options.lr_gamma <= 1:
            raise ValueError("CocoER lr_gamma must lie in (0,1]")
        if options.inside_lr <= 0 or options.grad_distance_weight < 0:
            raise ValueError("CocoER competition settings are invalid")
        if not 0 < options.pseudo_threshold < 1:
            raise ValueError("CocoER pseudo threshold must lie in (0,1)")
        return options


class _CLIPRN50ImageEncoder(nn.Module):
    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        dtype = next(self.visual.parameters()).dtype
        return self.visual(images.to(dtype=dtype)).float()


def _load_resnet_state(path: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("CocoER ResNet-50 initialization is not an audited asset bundle")
    expected_metadata = {
        "asset_kind": "cocoer_torchvision_resnet50_imagenet1k_v1",
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V1",
        "source_url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise ValueError(f"CocoER ResNet-50 asset {key} differs")
    source_sha = str(payload.get("source_file_sha256", "")).lower()
    if len(source_sha) != 64 or not source_sha.startswith("0676ba61"):
        raise ValueError("CocoER ResNet-50 source-file SHA-256 differs")
    payload = payload.get("state_dict")
    if not isinstance(payload, Mapping):
        raise ValueError("CocoER ResNet-50 initialization must be a state dict")
    state = {str(key).removeprefix("module."): value for key, value in payload.items()}
    required = {"conv1.weight", "layer4.2.conv3.weight", "fc.weight"}
    if not required.issubset(state):
        raise ValueError("CocoER initialization is not a torchvision ResNet-50 state")
    return state


def _validate_context(protocol: BenchmarkProtocol, context: TaskContext) -> None:
    if context.protocol_id != protocol.protocol_id or context.protocol_hash != protocol.protocol_hash:
        raise ValueError("CocoER task context differs from the protocol")
    if context.class_order_hash != protocol.class_order_hash or context.track != "B":
        raise ValueError("Native CocoER-FT requires the Track-B class contract")


def _validate_train_batch(batch: Any, context: TaskContext) -> TrainBatch:
    if not isinstance(batch, TrainBatch):
        raise TypeError("CocoER training requires protocol-safe TrainBatch values")
    CocoERFTModel.split_inputs(batch.images, batch.geometry)
    expected_shape = (batch.images.shape[0], len(context.current_class_indices))
    if batch.targets_current.shape != expected_shape:
        raise ValueError("CocoER current targets do not match the task")
    expected = torch.zeros_like(batch.visible_mask, dtype=torch.bool)
    expected[:, list(context.current_class_indices)] = True
    if not torch.equal(batch.visible_mask.bool(), expected):
        raise ValueError("CocoER training batch exposes labels outside current classes")
    if len(batch.sample_ids) != batch.images.shape[0]:
        raise ValueError("CocoER training IDs and images are not aligned")
    return batch


def _validate_eval_batch(batch: Any, context: TaskContext) -> EvaluationBatch:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("CocoER prediction requires EvaluationBatch values")
    CocoERFTModel.split_inputs(batch.images, batch.geometry)
    if batch.class_order_hash != context.class_order_hash:
        raise ValueError("CocoER evaluation class order differs")
    if batch.targets_seen.shape != (batch.images.shape[0], len(context.seen_class_indices)):
        raise ValueError("CocoER evaluation targets do not match seen classes")
    return batch


class CocoERFTBenchmarkMethod(BenchmarkMethod):
    method_name = "CocoER-FT"
    method_family = "Static EMOTIC model / Sequential Fine-Tuning"
    backbone = "Native CocoER ImageNet ResNet-50 x3 + frozen OpenAI CLIP RN50"
    supported_tracks = ("B",)
    upstream_repository = "https://github.com/bisno/CocoER"
    upstream_commit = UPSTREAM_COMMIT
    upstream_license = "MIT"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        resnet50_initialization_path: Optional[Union[str, Path]] = None,
        clip_rn50_path: Optional[Union[str, Path]] = None,
        head_box_cache_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[CocoERFTModel] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if protocol.track not in self.supported_tracks:
            raise ValueError("Native CocoER-FT supports Track B only")
        configured = dict(protocol.method_options("cocoer_ft"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.options = CocoERFTOptions.from_mapping(configured)
        self.protocol = protocol
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._set_seed(protocol.seed)
        self._amp_enabled = self.options.amp and self.device.type == "cuda"
        self.resnet50_initialization_path = None
        self.resnet50_initialization_sha256 = None
        self.clip_rn50_path = None
        self.clip_rn50_sha256 = None
        self.head_box_cache_path = (
            str(Path(head_box_cache_path).expanduser().resolve())
            if head_box_cache_path is not None else None
        )
        self.head_box_cache_sha256 = (
            _sha256_file(Path(self.head_box_cache_path))
            if self.head_box_cache_path is not None else None
        )
        if model is None:
            if not resnet50_initialization_path or not clip_rn50_path:
                raise FileNotFoundError(
                    "CocoER-FT requires explicit torchvision ImageNet ResNet-50 and OpenAI CLIP RN50 assets"
                )
            resnet_path = Path(resnet50_initialization_path).expanduser().resolve()
            clip_path = Path(clip_rn50_path).expanduser().resolve()
            if not resnet_path.is_file() or not clip_path.is_file():
                raise FileNotFoundError("CocoER native initialization asset is missing")
            clip_sha256 = _sha256_file(clip_path)
            if clip_sha256 != OPENAI_CLIP_RN50_SHA256:
                raise ValueError("CocoER CLIP RN50 checkpoint SHA-256 differs from OpenAI")
            state = _load_resnet_state(resnet_path)
            from clip import clip

            clip_model, _ = clip.load(str(clip_path), device="cpu")
            image_encoder = _CLIPRN50ImageEncoder(clip_model.visual.float())
            model = CocoERFTModel(
                protocol.num_classes,
                NativeResNet50GridEncoder(dict(state)),
                NativeResNet50GridEncoder(dict(state)),
                NativeResNet50GridEncoder(dict(state)),
                image_encoder,
                encoder_blocks=self.options.encoder_blocks,
                inside_lr=self.options.inside_lr,
                pseudo_threshold=self.options.pseudo_threshold,
            )
            self.resnet50_initialization_path = str(resnet_path)
            self.resnet50_initialization_sha256 = _sha256_file(resnet_path)
            self.clip_rn50_path = str(clip_path)
            self.clip_rn50_sha256 = clip_sha256
        self.model = model.float().to(self.device)
        self.model.clip_image_encoder.requires_grad_(False)
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
            raise RuntimeError("CocoER-FT tasks must be trained sequentially")
        old_classes = len(task_context.seen_class_indices) - len(task_context.current_class_indices)
        if self.model.num_classes != old_classes:
            raise RuntimeError("CocoER heads do not match the task boundary")
        self.model.add_head(len(task_context.current_class_indices))
        self.model.to(self.device)
        self.model.clip_image_encoder.requires_grad_(False)
        self.task_context = task_context
        self._optimizer_parameter_names = tuple(
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )
        self.training_history = []

    def _outputs(self, batch: Union[TrainBatch, EvaluationBatch]):
        return self.model(
            batch.images.to(self.device, non_blocking=True).float(),
            batch.geometry.to(self.device, non_blocking=True).float(),
        )

    def _selection_map(self, loader: Iterable[TrainBatch]) -> float:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede validation")
        self.model.eval()
        scores, targets = [], []
        current = len(self.task_context.current_class_indices)
        for raw_batch in loader:
            batch = _validate_train_batch(raw_batch, self.task_context)
            with self._autocast():
                logits = self._outputs(batch)["logits"][:, -current:]
            scores.append(torch.sigmoid(logits).detach().float().cpu())
            targets.append(batch.targets_current.detach().float().cpu())
        if not scores:
            raise ValueError("Validation loader produced no samples")
        all_scores, all_targets = torch.cat(scores), torch.cat(targets)
        return 100.0 * sum(
            average_precision(all_scores[:, index], all_targets[:, index])
            for index in range(all_targets.shape[1])
        ) / all_targets.shape[1]

    def train_task(self, train_loader: Iterable[TrainBatch], val_loader: Iterable[TrainBatch]) -> None:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede training")
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.options.learning_rate,
            betas=(self.options.beta1, self.options.beta2),
            weight_decay=self.options.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.options.lr_step_epochs, gamma=self.options.lr_gamma
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        current = len(self.task_context.current_class_indices)
        best_map, best_state, stale = -math.inf, None, 0
        for epoch in range(self.options.epochs):
            self.model.train()
            totals = {"loss": 0.0, "global": 0.0, "vi": 0.0, "competition": 0.0,
                      "pseudo_positive_rate": 0.0, "batches": 0.0, "steps": 0.0, "skips": 0.0}
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                targets = batch.targets_current.to(self.device).float()
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self._outputs(batch)
                    global_loss = dynamic_bce(output["logits"][:, -current:], targets)
                    vi_loss = dynamic_bce(output["vi_logits"][:, -current:], targets)
                    pseudo_loss = (
                        output["pseudo_head_loss"] + output["pseudo_body_loss"]
                        + output["pseudo_context_loss"]
                    )
                    loss = 0.2 * (global_loss + vi_loss + pseudo_loss) + (
                        self.options.grad_distance_weight * output["competition_distance"]
                    )
                old_scale = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, self.options.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                stepped = float(scaler.get_scale()) >= old_scale
                totals["steps"] += float(stepped)
                totals["skips"] += float(not stepped)
                totals["loss"] += float(loss.detach().cpu())
                totals["global"] += float(global_loss.detach().cpu())
                totals["vi"] += float(vi_loss.detach().cpu())
                totals["competition"] += float(output["competition_distance"].detach().cpu())
                totals["pseudo_positive_rate"] += float(output["pseudo_positive_rate"].detach().cpu())
                totals["batches"] += 1
            if not totals["batches"]:
                raise ValueError("Training loader produced no samples")
            validation_map = self._selection_map(val_loader)
            row = {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation_current_mAP": validation_map,
                "optimizer_steps": totals["steps"],
                "skipped_optimizer_steps": totals["skips"],
            }
            row.update({key: value / totals["batches"] for key, value in totals.items()
                        if key not in {"batches", "steps", "skips"}})
            self.training_history.append(row)
            if validation_map > best_map:
                best_map = validation_map
                best_state = {name: value.detach().cpu().clone()
                              for name, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            scheduler.step()
            if stale >= self.options.early_stopping_patience:
                break
        if best_state is None:
            raise RuntimeError("No CocoER epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        self._completed_task_id = self.task_context.task_id

    def predict_scores(self, data_loader: Iterable[EvaluationBatch]) -> PredictionOutput:
        if self.task_context is None:
            raise RuntimeError("begin_task must precede prediction")
        self.model.eval()
        scores, targets, sample_ids = [], [], []
        split_hash = None
        for raw_batch in data_loader:
            batch = _validate_eval_batch(raw_batch, self.task_context)
            if split_hash is None:
                split_hash = batch.split_hash
            elif split_hash != batch.split_hash:
                raise ValueError("Evaluation loader contains multiple split hashes")
            with self._autocast():
                logits = self._outputs(batch)["logits"]
            scores.append(torch.sigmoid(logits).detach().float().cpu())
            targets.append(batch.targets_seen.detach().float().cpu())
            sample_ids.extend(batch.sample_ids)
        if not scores or split_hash is None:
            raise ValueError("Evaluation loader produced no samples")
        return PredictionOutput(
            scores=torch.cat(scores), targets=torch.cat(targets), sample_ids=sample_ids,
            class_order_hash=self.task_context.class_order_hash, split_hash=split_hash,
        )

    def end_task(self) -> None:
        self.task_context = None

    def training_log_records(self):
        return tuple(dict(row) for row in self.training_history)

    def resolved_method_config(self) -> Mapping[str, Any]:
        return {
            "strategy": "sequential_finetuning",
            "conversion_interface": "CocoER-FT-v0.1",
            "upstream_repository": self.upstream_repository,
            "upstream_commit": self.upstream_commit,
            "upstream_license": self.upstream_license,
            "input_mode": "cocoer_multilevel",
            "native_backbone": self.backbone,
            "resnet50_initialization_path": self.resnet50_initialization_path,
            "resnet50_initialization_sha256": self.resnet50_initialization_sha256,
            "clip_rn50_path": self.clip_rn50_path,
            "clip_rn50_sha256": self.clip_rn50_sha256,
            "head_box_cache_path": self.head_box_cache_path,
            "head_box_cache_sha256": self.head_box_cache_sha256,
            "full_26_class_gwt_checkpoint_used": False,
            "full_26_class_vi_mapping_checkpoint_used": False,
            "future_leakage_reason": "released GWT and w.pth are trained on all EMOTIC classes",
            "current_label_only": True,
            "old_label_truth_used": False,
            "future_label_truth_used": False,
            "distillation_enabled": False,
            "replay_enabled": False,
            "ewc_enabled": False,
            "benchmark_added_adapter": False,
            "clip_visual_encoder_used": True,
            "clip_visual_backbone": "OpenAI CLIP RN50 (native CocoER component)",
            "clip_text_encoder_used": False,
            "selection_metric": "current_label_validation_mAP",
            "f1_threshold": self.protocol.threshold,
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
            "resnet50_initialization_sha256": self.resnet50_initialization_sha256,
            "clip_rn50_sha256": self.clip_rn50_sha256,
            "head_box_cache_sha256": self.head_box_cache_sha256,
            "training_history": self.training_history,
        }, destination)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported CocoER checkpoint")
        for key, expected in {
            "method": self.method_name,
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
            "resnet50_initialization_sha256": self.resnet50_initialization_sha256,
            "clip_rn50_sha256": self.clip_rn50_sha256,
            "head_box_cache_sha256": self.head_box_cache_sha256,
        }.items():
            if payload.get(key) != expected:
                raise ValueError(f"Checkpoint {key} differs from CocoER-FT")
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("Checkpoint CocoER options differ")
        completed = int(payload["completed_task_id"])
        expected_heads = tuple(
            len(self.protocol.current_class_indices(task_id)) for task_id in range(completed + 1)
        )
        if tuple(payload["head_sizes"]) != expected_heads:
            raise ValueError("Checkpoint heads differ from the protocol")
        if self.model.head_sizes:
            raise RuntimeError("Checkpoint loading requires an unexpanded model")
        self.model.restore_heads(expected_heads)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self.model.clip_image_encoder.requires_grad_(False)
        self._completed_task_id = completed
        self.training_history = [dict(row) for row in payload.get("training_history", [])]

    def parameter_statistics(self) -> ParameterStatistics:
        parameters = dict(self.model.named_parameters())
        total = sum(parameter.numel() for parameter in parameters.values())
        trainable = sum(parameters[name].numel() for name in self._optimizer_parameter_names
                        if name in parameters)
        # Five progressive classifiers are added per task: head/body/context,
        # VI, and global. All consume a 256-D feature.
        per_task = {
            task_id: (0 if task_id == 0 else len(self.protocol.current_class_indices(task_id)) * 5 * 257)
            for task_id in range(self.protocol.num_tasks)
        }
        completed = max(self._completed_task_id, 0)
        return ParameterStatistics(
            total_parameters=total,
            trainable_parameters=trainable,
            incremental_parameters=sum(per_task[index] for index in range(1, min(completed + 1, self.protocol.num_tasks))),
            per_task_incremental_parameters=per_task,
        )

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("cocoer_ft", CocoERFTBenchmarkMethod)
