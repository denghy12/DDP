"""Task-routed Adapter Bank utilities for EMOTIC B5-C3.

Every class is routed to the Adapter trained when that class was introduced.
The routing is class based, not sample based, so one multi-label image may use
several frozen Adapters without requiring an oracle task identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ddp_internal_adapter import (
    SharedResidualFeatureAdapter,
    feature_logit_correction,
)
from prototype_fewshot import sample_multilabel_kshot


TASK_CLASS_RANGES: Tuple[Tuple[int, int], ...] = (
    (0, 5),
    (5, 8),
    (8, 11),
    (11, 14),
    (14, 17),
    (17, 20),
    (20, 23),
    (23, 26),
)
TASK_SEEN_CLASSES: Tuple[int, ...] = tuple(high for _, high in TASK_CLASS_RANGES)
BANK_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1


def task_class_range(task_id: int) -> Tuple[int, int]:
    task_id = int(task_id)
    if not 0 <= task_id < len(TASK_CLASS_RANGES):
        raise ValueError(f"task_id must be in [0, 7], got {task_id}")
    return TASK_CLASS_RANGES[task_id]


def task_class_indices(task_id: int) -> torch.Tensor:
    low, high = task_class_range(task_id)
    return torch.arange(low, high, dtype=torch.long)


def class_task_id(class_id: int) -> int:
    class_id = int(class_id)
    if not 0 <= class_id < TASK_CLASS_RANGES[-1][1]:
        raise ValueError(f"class_id must be in [0, 25], got {class_id}")
    if class_id < 5:
        return 0
    return 1 + (class_id - 5) // 3


def class_to_task_map(seen_classes: int = 26) -> Tuple[int, ...]:
    seen_classes = int(seen_classes)
    if not 1 <= seen_classes <= 26:
        raise ValueError(f"seen_classes must be in [1, 26], got {seen_classes}")
    return tuple(class_task_id(class_id) for class_id in range(seen_classes))


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_task_training_subset(
    labels: torch.Tensor,
    task_id: int,
    training_mode: str,
    seed: int,
    shots_per_class: int = 16,
):
    """Select one task's samples and construct a strict partial-label mask.

    ``full`` keeps every unique person with at least one current-task positive
    and supervises all current-task labels. ``16shot`` reuses the strict
    multi-label sampler: exactly K positive anchors are supervised per current
    class, positive co-labels outside those anchors are ignored, and genuine
    current-task negatives in the selected union remain supervised.
    Old and future classes are always masked.
    """

    if labels.ndim != 2 or labels.shape[1] != 26:
        raise ValueError(
            "labels must have shape [samples, 26], got " f"{tuple(labels.shape)}"
        )
    active = task_class_indices(task_id)
    if training_mode == "full":
        selected = torch.nonzero(
            labels[:, active].sum(dim=1).gt(0), as_tuple=False
        ).flatten()
        source_mask = torch.ones_like(labels[selected], dtype=torch.bool)
        sampling_rows = []
    elif training_mode == "16shot":
        selected, source_mask, sampling_rows = sample_multilabel_kshot(
            labels,
            active,
            int(shots_per_class),
            int(seed),
        )
    else:
        raise ValueError("training_mode must be 'full' or '16shot'")

    if selected.numel() == 0:
        raise RuntimeError(f"Task {task_id} produced an empty training subset")
    selected_labels = labels[selected]
    supervision_mask = torch.zeros_like(selected_labels, dtype=torch.bool)
    supervision_mask[:, active] = source_mask[:, active]
    low, high = task_class_range(task_id)
    old_positive_count = int(selected_labels[:, :low].sum().item()) if low else 0
    future_positive_count = (
        int(selected_labels[:, high:].sum().item()) if high < labels.shape[1] else 0
    )
    current_positive_count = int(selected_labels[:, low:high].sum().item())
    sampling = {
        "mode": "full_data" if training_mode == "full" else "fewshot",
        "task_id": int(task_id),
        "class_range": [low, high],
        "shots_per_class": None if training_mode == "full" else int(shots_per_class),
        "seed": int(seed),
        "unique_training_samples": int(selected.numel()),
        "selected_train_indices": selected.tolist(),
        "current_positive_labels": current_positive_count,
        "ignored_old_positive_labels": old_positive_count,
        "ignored_future_positive_labels": future_positive_count,
        "supervised_entries": int(supervision_mask.sum().item()),
        "sampling": sampling_rows,
    }
    return selected, supervision_mask, sampling


def _checkpoint_task_metadata(checkpoint: Mapping) -> Mapping:
    metadata = checkpoint.get("task_adapter")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint does not contain task_adapter metadata")
    return metadata


def validate_task_adapter_checkpoint(
    checkpoint: Mapping,
    task_id: Optional[int] = None,
    training_mode: Optional[str] = None,
    seed: Optional[int] = None,
    classnames: Optional[Sequence[str]] = None,
) -> Mapping:
    metadata = _checkpoint_task_metadata(checkpoint)
    checkpoint_task = int(metadata["task_id"])
    expected_range = list(task_class_range(checkpoint_task))
    if list(metadata.get("class_range", [])) != expected_range:
        raise ValueError(
            f"Task {checkpoint_task} checkpoint has invalid class_range "
            f"{metadata.get('class_range')}; expected {expected_range}"
        )
    if task_id is not None and checkpoint_task != int(task_id):
        raise ValueError(
            f"Expected task {task_id} Adapter, found task {checkpoint_task}"
        )
    if training_mode is not None and metadata.get("training_mode") != training_mode:
        raise ValueError(
            f"Expected {training_mode} Adapter, found "
            f"{metadata.get('training_mode')}"
        )
    if seed is not None and int(metadata.get("seed", -1)) != int(seed):
        raise ValueError(
            f"Expected seed {seed} Adapter, found {metadata.get('seed')}"
        )
    if classnames is not None and list(checkpoint.get("classnames", [])) != list(
        classnames
    ):
        raise ValueError("Adapter and DDP class orders differ")
    if checkpoint.get("model") is None:
        raise ValueError("Adapter checkpoint does not contain model weights")
    return metadata


class TaskRoutedAdapterBank(nn.Module):
    """A frozen Adapter bank with deterministic class-to-task routing."""

    def __init__(
        self,
        adapters: Mapping[int, nn.Module],
        inference_alpha: float = 0.03,
    ):
        super().__init__()
        if inference_alpha < 0:
            raise ValueError("inference_alpha must be non-negative")
        if not adapters:
            raise ValueError("Adapter bank must contain at least task 0")
        task_ids = sorted(int(task_id) for task_id in adapters)
        if task_ids != list(range(task_ids[-1] + 1)):
            raise ValueError(
                "Adapter bank tasks must be contiguous from task 0; got "
                f"{task_ids}"
            )
        self.adapters = nn.ModuleDict(
            {str(task_id): adapters[task_id] for task_id in task_ids}
        )
        self.inference_alpha = float(inference_alpha)
        for adapter in self.adapters.values():
            if hasattr(adapter, "residual_scale"):
                adapter.residual_scale = self.inference_alpha
            adapter.eval()
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)

    @property
    def max_task(self) -> int:
        return max(int(key) for key in self.adapters.keys())

    @property
    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def feature_difference_correction(
        self,
        prompted_cls_features: torch.Tensor,
        text_features: torch.Tensor,
        seen_classes: int,
        logit_scale: float = 100.0,
    ) -> torch.Tensor:
        """Return path-logit corrections in DDP's [neg..., pos...] order."""

        if prompted_cls_features.ndim != 3:
            raise ValueError("prompted_cls_features must be [batch, 2K, dim]")
        seen_classes = int(seen_classes)
        expected_paths = 2 * seen_classes
        if prompted_cls_features.shape[1] != expected_paths:
            raise ValueError(
                f"Expected {expected_paths} prompted paths, got "
                f"{prompted_cls_features.shape[1]}"
            )
        if text_features.shape != prompted_cls_features.shape[1:]:
            raise ValueError(
                "text_features must match [2K, dim]; got "
                f"{tuple(text_features.shape)}"
            )
        required_max_task = class_task_id(seen_classes - 1)
        if self.max_task < required_max_task:
            raise ValueError(
                f"Seen classes require task {required_max_task}, but bank only "
                f"contains tasks 0..{self.max_task}"
            )

        correction = torch.zeros(
            prompted_cls_features.shape[:2],
            device=prompted_cls_features.device,
            dtype=torch.float32,
        )
        for task_id in range(required_max_task + 1):
            low, task_high = task_class_range(task_id)
            high = min(task_high, seen_classes)
            if low >= high:
                continue
            negative = torch.arange(low, high, device=prompted_cls_features.device)
            positive = torch.arange(
                seen_classes + low,
                seen_classes + high,
                device=prompted_cls_features.device,
            )
            path_indices = torch.cat([negative, positive])
            path_features = prompted_cls_features[:, path_indices, :]
            path_text = text_features[path_indices]
            adapted, original = self.adapters[str(task_id)](path_features)
            task_correction = feature_logit_correction(
                adapted,
                original,
                path_text,
                mode="linear_residual",
                logit_scale=float(logit_scale),
            )
            correction[:, path_indices] = task_correction
        return correction

    def logits_from_features(
        self,
        prompted_cls_features: torch.Tensor,
        base_path_logits: torch.Tensor,
        text_features: torch.Tensor,
        seen_classes: int,
        logit_scale: float = 100.0,
    ) -> torch.Tensor:
        correction = self.feature_difference_correction(
            prompted_cls_features,
            text_features,
            seen_classes,
            logit_scale=logit_scale,
        )
        path_logits = base_path_logits.float() + correction
        return path_logits.reshape(path_logits.shape[0], 2, int(seen_classes))

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        device: torch.device | str,
        max_task: Optional[int] = None,
        classnames: Optional[Sequence[str]] = None,
    ) -> "TaskRoutedAdapterBank":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != BANK_SCHEMA_VERSION:
            raise ValueError("Unsupported Adapter Bank manifest schema")
        manifest_classnames = manifest.get("classnames", [])
        if classnames is not None and list(manifest_classnames) != list(classnames):
            raise ValueError("Adapter Bank and DDP class orders differ")
        available = sorted(int(key) for key in manifest.get("adapters", {}))
        if not available:
            raise ValueError("Adapter Bank manifest contains no Adapters")
        resolved_max = available[-1] if max_task is None else int(max_task)
        expected = list(range(resolved_max + 1))
        if not all(task_id in available for task_id in expected):
            raise ValueError(
                f"Manifest does not contain every task in {expected}; got {available}"
            )
        adapters: Dict[int, SharedResidualFeatureAdapter] = {}
        for task_id in expected:
            entry = manifest["adapters"][str(task_id)]
            checkpoint_path = manifest_path.parent / entry["checkpoint"]
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            actual_hash = file_sha256(checkpoint_path)
            if actual_hash != entry.get("sha256"):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            validate_task_adapter_checkpoint(
                checkpoint,
                task_id=task_id,
                training_mode=manifest["training_mode"],
                seed=manifest["seed"],
                classnames=manifest_classnames,
            )
            checkpoint_args = checkpoint["args"]
            adapter = SharedResidualFeatureAdapter(
                feature_dim=int(checkpoint_args.get("feature_dim", 512)),
                bottleneck_dim=int(checkpoint_args["adapter_dim"]),
                residual_scale=float(manifest["inference_alpha"]),
            )
            adapter.load_state_dict(checkpoint["model"], strict=True)
            adapters[task_id] = adapter.to(device)
        return cls(
            adapters,
            inference_alpha=float(manifest["inference_alpha"]),
        ).to(device)


def build_bank_manifest(
    bank_dir: Union[str, Path],
    training_mode: str,
    seed: int,
    classnames: Sequence[str],
    inference_alpha: float = 0.03,
) -> dict:
    bank_dir = Path(bank_dir)
    adapters = {}
    for task_id in range(len(TASK_CLASS_RANGES)):
        checkpoint_path = bank_dir / f"task{task_id}" / "best_adapter.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = validate_task_adapter_checkpoint(
            checkpoint,
            task_id=task_id,
            training_mode=training_mode,
            seed=seed,
            classnames=classnames,
        )
        adapters[str(task_id)] = {
            "task_id": task_id,
            "class_range": list(task_class_range(task_id)),
            "checkpoint": str(checkpoint_path.relative_to(bank_dir)),
            "sha256": file_sha256(checkpoint_path),
            "best_epoch": checkpoint.get("epoch"),
            "selection_score": checkpoint.get("selection_score"),
            "initialization": metadata.get("initialization"),
        }
    return {
        "schema_version": BANK_SCHEMA_VERSION,
        "name": "EMOTIC B5-C3 task-routed Adapter Bank",
        "training_mode": training_mode,
        "seed": int(seed),
        "inference_formula": "feature_difference",
        "legacy_correction_mode": "linear_residual",
        "inference_alpha": float(inference_alpha),
        "class_specific_gate": False,
        "task_specific_alpha": False,
        "test_used_for_selection": False,
        "classnames": list(classnames),
        "class_to_task": list(class_to_task_map()),
        "adapters": adapters,
    }
