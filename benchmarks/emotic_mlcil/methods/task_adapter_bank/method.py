"""Formal Track-A evaluation of the strongest full-data Task Adapter Bank.

The DDP prompt learner follows the registered Original-DDP-Tau2 lifecycle.
After each DDP task finishes, a prompt-free auxiliary branch trains one small
residual Adapter from current-task train labels and selects its checkpoint on
current-task validation mAP.  At inference, every class is routed to the
frozen Adapter from its introduction task and the fixed feature-difference
correction is applied to that class's prompted CLS paths.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ddp_internal_adapter import (
    PromptFreePrototypeObjective,
    SharedResidualFeatureAdapter,
)
from emotic_task_adapter_bank import (
    TaskRoutedAdapterBank,
    task_ranges_from_sizes,
)
from evaluation_metrics import mAP
from prototype_adapter import NEGATIVE_TEMPLATES, POSITIVE_TEMPLATES
from prototype_fewshot import masked_bce_with_logits, masked_pos_weight
from train_emotic_ddp_prompt_free_auxiliary import encode_template_ensemble

from ...protocol import BenchmarkProtocol
from ...registry import register_method
from ...types import MemoryStatistics, ParameterStatistics, TaskContext, TrainBatch
from ..original_ddp.method import (
    OriginalDDPBenchmarkMethod,
    _validate_context,
    _validate_train_batch,
)


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskAdapterBankOptions:
    adapter_epochs: int = 50
    adapter_lr: float = 1.0e-3
    adapter_weight_decay: float = 1.0e-4
    adapter_bottleneck_dim: int = 128
    training_residual_scale: float = 0.1
    inference_alpha: float = 0.03
    identity_weight: float = 0.1
    initial_logit_scale: float = 10.0
    adapter_batch_size: int = 1024
    positive_weight_max: float = 20.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskAdapterBankOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError(
                "Unknown Task Adapter Bank option(s): " + ", ".join(unknown)
            )
        options = cls(**dict(value))
        if options.adapter_epochs <= 0 or options.adapter_batch_size <= 0:
            raise ValueError("Adapter epochs and batch size must be positive")
        if options.adapter_lr <= 0 or options.adapter_weight_decay < 0:
            raise ValueError("Adapter optimizer settings are invalid")
        if options.adapter_bottleneck_dim <= 0:
            raise ValueError("Adapter bottleneck dimension must be positive")
        if options.training_residual_scale < 0 or options.inference_alpha < 0:
            raise ValueError("Adapter residual scales must be non-negative")
        if options.identity_weight < 0 or options.initial_logit_scale <= 0:
            raise ValueError("Adapter loss/logit-scale settings are invalid")
        if options.positive_weight_max <= 0:
            raise ValueError("positive_weight_max must be positive")
        return options


class DDPTaskAdapterBankMethod(OriginalDDPBenchmarkMethod):
    """Original DDP plus a protocol-routed, frozen full-data Adapter Bank."""

    method_name = "DDP-Task-Adapter-Bank-Full"
    method_family = "Native MLCIL / DDP with task-routed feature adapters"
    backbone = "OpenAI CLIP ViT-B/16 (frozen)"
    supported_tracks = ("A",)

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[torch.nn.Module] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
        checkpoint_paths: Optional[Mapping[int, Union[str, Path]]] = None,
    ) -> None:
        configured = dict(protocol.method_options("task_adapter_bank"))
        if option_overrides:
            configured.update(dict(option_overrides))
        self.adapter_options = TaskAdapterBankOptions.from_mapping(configured)
        self.task_class_ranges = task_ranges_from_sizes(
            [len(task) for task in protocol.tasks]
        )
        self._task_adapters: Dict[int, SharedResidualFeatureAdapter] = {}
        self._adapter_logit_scales: Dict[int, torch.Tensor] = {}
        self._adapter_history: List[Dict[str, float]] = []
        self._adapter_selection: Dict[int, Dict[str, Any]] = {}
        self.checkpoint_paths = {
            int(task): Path(path).expanduser().resolve()
            for task, path in (checkpoint_paths or {}).items()
        }
        self._source_checkpoint_hashes: Dict[int, str] = {}
        super().__init__(
            protocol,
            clip_model_path=clip_model_path,
            device=device,
            model=model,
            option_overrides=None,
        )

    def begin_task(self, task_context: TaskContext) -> None:
        self.model.disable_task_adapter_bank()
        _validate_context(self.protocol, task_context)
        expected = self._completed_task_id + 1
        if task_context.task_id != expected:
            raise RuntimeError(
                f"Task Adapter Bank requires sequential tasks: expected "
                f"{expected}, got {task_context.task_id}"
            )
        checkpoint_path = self.checkpoint_paths.get(task_context.task_id)
        if checkpoint_path is None:
            raise ValueError(
                f"No frozen Original-DDP checkpoint configured for task "
                f"{task_context.task_id}"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        expected_metadata = {
            "schema_version": 1,
            "method": "Original-DDP-Tau2",
            "protocol_id": self.protocol.protocol_id,
            "protocol_hash": self.protocol.protocol_hash,
            "class_order_hash": self.protocol.class_order_hash,
            "completed_task_id": task_context.task_id,
        }
        for key, expected_value in expected_metadata.items():
            if payload.get(key) != expected_value:
                raise ValueError(
                    f"Frozen Original-DDP checkpoint {key} differs: "
                    f"{payload.get(key)!r} != {expected_value!r}"
                )
        if dict(payload.get("options", {})) != asdict(self.options):
            raise ValueError("Frozen Original-DDP checkpoint options differ")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device)
        self._freeze_non_optimizer_parameters()
        self.task_context = task_context
        self._completed_task_id = task_context.task_id
        self.training_history = []
        self._rebuild_text_feature_cache()
        self._source_checkpoint_hashes[task_context.task_id] = _sha256(
            checkpoint_path
        )

    def _collect_prompt_free_features(
        self,
        loader: Iterable[TrainBatch],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_context is None:
            raise RuntimeError("Task context is required for Adapter extraction")
        features: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for raw_batch in loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._images(batch.images)
                mean = torch.as_tensor(
                    CLIP_MEAN, device=images.device, dtype=images.dtype
                ).view(1, 3, 1, 1)
                std = torch.as_tensor(
                    CLIP_STD, device=images.device, dtype=images.dtype
                ).view(1, 3, 1, 1)
                images = (images - mean) / std
                with self._autocast():
                    encoded = self.model.encode_prompt_free_image(
                        images
                    )
                features.append(encoded.float().cpu())
                targets.append(batch.targets_current.float().cpu())
        if not features:
            raise ValueError("Adapter feature extraction produced no samples")
        return torch.cat(features), torch.cat(targets)

    @staticmethod
    def _validation_map(
        objective: PromptFreePrototypeObjective,
        features: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[float, List[float]]:
        loader = DataLoader(
            TensorDataset(features),
            batch_size=batch_size,
            shuffle=False,
        )
        logits: List[torch.Tensor] = []
        objective.eval()
        with torch.no_grad():
            for (batch,) in loader:
                values, _, _ = objective(batch.to(device))
                logits.append(values.float().cpu())
        score, per_class = mAP(
            targets.numpy(), torch.sigmoid(torch.cat(logits)).numpy()
        )
        return float(score), [100.0 * float(value) for value in per_class]

    def _new_task_adapter(self, task_id: int) -> SharedResidualFeatureAdapter:
        adapter = SharedResidualFeatureAdapter(
            feature_dim=512,
            bottleneck_dim=self.adapter_options.adapter_bottleneck_dim,
            residual_scale=self.adapter_options.training_residual_scale,
        ).to(self.device)
        if task_id > 0:
            if 0 not in self._task_adapters:
                raise RuntimeError("Later task Adapters require the task-0 anchor")
            adapter.load_state_dict(
                copy.deepcopy(self._task_adapters[0].state_dict()), strict=True
            )
        return adapter

    def _train_current_adapter(
        self,
        train_loader: Iterable[TrainBatch],
        val_loader: Iterable[TrainBatch],
    ) -> None:
        if self.task_context is None:
            raise RuntimeError("Task context is required for Adapter training")
        task_id = self.task_context.task_id
        train_features, train_targets = self._collect_prompt_free_features(
            train_loader
        )
        val_features, val_targets = self._collect_prompt_free_features(val_loader)
        current_names = list(self.task_context.current_class_names)
        positive = encode_template_ensemble(
            self.model, current_names, POSITIVE_TEMPLATES, self.device
        )
        negative = encode_template_ensemble(
            self.model, current_names, NEGATIVE_TEMPLATES, self.device
        )
        adapter = self._new_task_adapter(task_id)
        objective = PromptFreePrototypeObjective(
            adapter,
            positive.to(self.device),
            negative.to(self.device),
            initial_logit_scale=self.adapter_options.initial_logit_scale,
        ).to(self.device)
        mask = torch.ones_like(train_targets, dtype=torch.bool)
        active = torch.arange(train_targets.shape[1], dtype=torch.long)
        pos_weight = masked_pos_weight(
            train_targets,
            mask,
            active,
            max_weight=self.adapter_options.positive_weight_max,
        ).to(self.device)
        generator = torch.Generator().manual_seed(
            self.protocol.seed * 1009 + task_id
        )
        offline_loader = DataLoader(
            TensorDataset(train_features, train_targets, mask),
            batch_size=self.adapter_options.adapter_batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            objective.parameters(),
            lr=self.adapter_options.adapter_lr,
            weight_decay=self.adapter_options.adapter_weight_decay,
        )
        initial_score, initial_per_class = self._validation_map(
            objective,
            val_features,
            val_targets,
            self.adapter_options.adapter_batch_size,
            self.device,
        )
        best_score = -math.inf
        best_epoch = -1
        best_adapter: Optional[Dict[str, torch.Tensor]] = None
        best_logit_scale: Optional[torch.Tensor] = None
        best_per_class: Optional[List[float]] = None
        self._adapter_history = []
        for epoch in range(self.adapter_options.adapter_epochs):
            objective.train()
            total = 0.0
            total_classification = 0.0
            total_identity = 0.0
            samples = 0
            for feature_batch, target_batch, mask_batch in offline_loader:
                feature_batch = feature_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                logits, adapted, original = objective(feature_batch)
                classification = masked_bce_with_logits(
                    logits,
                    target_batch,
                    mask_batch,
                    pos_weight=pos_weight,
                )
                identity = 1.0 - (adapted * original).sum(dim=-1).mean()
                loss = (
                    classification
                    + self.adapter_options.identity_weight * identity
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = feature_batch.shape[0]
                samples += count
                total += float(loss.detach()) * count
                total_classification += float(classification.detach()) * count
                total_identity += float(identity.detach()) * count
            selection, per_class = self._validation_map(
                objective,
                val_features,
                val_targets,
                self.adapter_options.adapter_batch_size,
                self.device,
            )
            record = {
                "stage": "task_adapter",
                "task_id": float(task_id),
                "epoch": float(epoch),
                "loss": total / samples,
                "classification_loss": total_classification / samples,
                "identity_loss": total_identity / samples,
                "validation_current_mAP": selection,
            }
            self._adapter_history.append(record)
            if selection > best_score:
                best_score = selection
                best_epoch = epoch
                best_adapter = {
                    key: value.detach().cpu().clone()
                    for key, value in adapter.state_dict().items()
                }
                best_logit_scale = objective.logit_scale.detach().cpu().clone()
                best_per_class = per_class
        if best_adapter is None or best_logit_scale is None:
            raise RuntimeError("Adapter training produced no selectable checkpoint")
        adapter.load_state_dict(best_adapter, strict=True)
        adapter.residual_scale = self.adapter_options.inference_alpha
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        self._task_adapters[task_id] = adapter
        self._adapter_logit_scales[task_id] = best_logit_scale
        self._adapter_selection[task_id] = {
            "initial_validation_current_mAP": initial_score,
            "initial_per_class_ap": initial_per_class,
            "best_epoch": best_epoch,
            "best_validation_current_mAP": best_score,
            "best_per_class_ap": best_per_class,
            "train_samples": int(train_targets.shape[0]),
            "validation_samples": int(val_targets.shape[0]),
            "positive_weight": [float(value) for value in pos_weight.cpu()],
        }

    def _attach_current_bank(self) -> None:
        if self.task_context is None:
            raise RuntimeError("Task context is required for Adapter routing")
        expected = list(range(self.task_context.task_id + 1))
        if sorted(self._task_adapters) != expected:
            raise RuntimeError(
                f"Expected frozen Adapters {expected}, got "
                f"{sorted(self._task_adapters)}"
            )
        bank = TaskRoutedAdapterBank(
            {task: self._task_adapters[task] for task in expected},
            inference_alpha=self.adapter_options.inference_alpha,
            task_class_ranges=self.task_class_ranges,
        ).to(self.device)
        self.model.enable_task_adapter_bank(bank)

    def train_task(
        self,
        train_loader: Iterable[TrainBatch],
        val_loader: Iterable[TrainBatch],
    ) -> None:
        self.model.disable_task_adapter_bank()
        self._train_current_adapter(train_loader, val_loader)
        self.training_history = list(self._adapter_history)
        self._attach_current_bank()

    def predict_scores(self, data_loader):
        self._attach_current_bank()
        return super().predict_scores(data_loader)

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        bank = getattr(self.model, "feature_adapter_bank", None)
        self.model.disable_task_adapter_bank()
        try:
            super().save_checkpoint(path)
        finally:
            if bank is not None:
                self.model.enable_task_adapter_bank(bank)
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        payload["task_adapter_bank"] = {
            "schema_version": 1,
            "task_class_ranges": [list(bounds) for bounds in self.task_class_ranges],
            "adapter_options": asdict(self.adapter_options),
            "adapter_states": {
                str(task): {
                    key: value.detach().cpu()
                    for key, value in adapter.state_dict().items()
                }
                for task, adapter in self._task_adapters.items()
            },
            "auxiliary_logit_scales": {
                str(task): value.detach().cpu()
                for task, value in self._adapter_logit_scales.items()
            },
            "selection": copy.deepcopy(self._adapter_selection),
        }
        torch.save(payload, checkpoint_path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = torch.load(path, map_location="cpu")
        bank_payload = payload.get("task_adapter_bank")
        if not isinstance(bank_payload, Mapping):
            raise ValueError("Checkpoint does not contain Task Adapter Bank state")
        if dict(bank_payload.get("adapter_options", {})) != asdict(
            self.adapter_options
        ):
            raise ValueError("Task Adapter Bank checkpoint options differ")
        expected_ranges = [list(bounds) for bounds in self.task_class_ranges]
        if bank_payload.get("task_class_ranges") != expected_ranges:
            raise ValueError("Task Adapter Bank checkpoint protocol ranges differ")
        super().load_checkpoint(path)
        completed = self._completed_task_id
        states = bank_payload.get("adapter_states", {})
        expected_tasks = list(range(completed + 1))
        if sorted(int(key) for key in states) != expected_tasks:
            raise ValueError("Task Adapter Bank checkpoint tasks are incomplete")
        self._task_adapters = {}
        for task in expected_tasks:
            adapter = self._new_task_adapter(task)
            adapter.load_state_dict(states[str(task)], strict=True)
            adapter.residual_scale = self.adapter_options.inference_alpha
            adapter.eval()
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)
            self._task_adapters[task] = adapter
        self._adapter_logit_scales = {
            int(task): value.detach().cpu()
            for task, value in bank_payload.get(
                "auxiliary_logit_scales", {}
            ).items()
        }
        self._adapter_selection = {
            int(task): dict(value)
            for task, value in bank_payload.get("selection", {}).items()
        }

    def parameter_statistics(self) -> ParameterStatistics:
        bank = getattr(self.model, "feature_adapter_bank", None)
        if bank is not None:
            self.model.disable_task_adapter_bank()
        try:
            base_total = sum(parameter.numel() for parameter in self.model.parameters())
        finally:
            if bank is not None:
                self.model.enable_task_adapter_bank(bank)
        per_adapter = 2 * 512 * self.adapter_options.adapter_bottleneck_dim
        adapter_total = per_adapter * len(self._task_adapters)
        prompt_updated = sum(
            parameter.numel() for parameter in self._optimizer_parameters()
        )
        return ParameterStatistics(
            total_parameters=base_total + adapter_total,
            trainable_parameters=prompt_updated + adapter_total,
            incremental_parameters=adapter_total,
            per_task_incremental_parameters={
                task: per_adapter if task in self._task_adapters else 0
                for task in range(self.protocol.num_tasks)
            },
        )

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)

    def resolved_method_config(self) -> Mapping[str, Any]:
        original = dict(super().resolved_method_config())
        original.update(
            {
                "strategy": "original_ddp_task_adapter_bank_full",
                "benchmark_added_adapter": True,
                "adapter_training_input": "prompt_free_global_clip_cls",
                "adapter_training_label_scope": "current_classes_only",
                "adapter_checkpoint_selection": (
                    "current_task_validation_mAP_earliest_tie"
                ),
                "adapter_inference_input": "prompted_path_cls",
                "adapter_routing": "class_introduction_task",
                "adapter_inference_formula": "feature_difference",
                "ddp_training": "reused_frozen_protocol_checkpoint",
                "ddp_checkpoint_sha256_by_task": {
                    str(task): value
                    for task, value in self._source_checkpoint_hashes.items()
                },
                "test_labels_used_for_selection": False,
                **asdict(self.adapter_options),
            }
        )
        return original


register_method("task_adapter_bank", DDPTaskAdapterBankMethod)
