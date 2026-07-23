"""Transfer a prompt-free CLS Adapter Bank to DDP final-token inference.

This module deliberately keeps the legacy Task-routed Adapter Bank and the
trained Final-token Adapter Bank separate.  It accepts only the former's
``feature_difference`` manifest, copies its point-wise MLP weights into
``SharedResidualFinalTokenAdapter`` instances, freezes them, and applies them
either to every final visual token or to the CLS token alone.

Input and output token tensors use DDP's public ``[B, 2K, D, N]`` layout.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import torch
from torch import nn

from emotic_final_token_adapter import SharedResidualFinalTokenAdapter
from emotic_task_adapter_bank import (
    BANK_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    TASK_CLASS_RANGES,
    class_task_id,
    class_to_task_map,
    file_sha256,
    task_class_range,
    validate_task_adapter_checkpoint,
)


CLS_TO_FINAL_TOKEN_TRANSFER_SCHEMA_VERSION = 1
CLS_SOURCE_FORMULA = "feature_difference"
CLS_SOURCE_CORRECTION_MODE = "linear_residual"
TRANSFER_RESIDUAL_SCALE = 0.03
APPLICATION_MODES = ("all_tokens", "cls_only")


def _validate_application_mode(application_mode: str) -> str:
    application_mode = str(application_mode)
    if application_mode not in APPLICATION_MODES:
        raise ValueError(
            f"application_mode must be one of {APPLICATION_MODES}, got "
            f"{application_mode!r}"
        )
    return application_mode


def _validate_cls_manifest(
    manifest: Mapping,
    *,
    classnames: Optional[Sequence[str]],
) -> Sequence[str]:
    if int(manifest.get("schema_version", -1)) != BANK_SCHEMA_VERSION:
        raise ValueError("Unsupported CLS Adapter Bank manifest schema")
    if manifest.get("inference_formula") != CLS_SOURCE_FORMULA:
        raise ValueError(
            "Source manifest must be the legacy feature_difference CLS Bank"
        )
    if manifest.get("legacy_correction_mode") != CLS_SOURCE_CORRECTION_MODE:
        raise ValueError(
            "Source manifest must use the locked linear_residual correction"
        )
    alpha = float(manifest.get("inference_alpha", float("nan")))
    if not math.isclose(
        alpha,
        TRANSFER_RESIDUAL_SCALE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "CLS-to-Final-token transfer requires inference_alpha=0.03; "
            f"found {manifest.get('inference_alpha')}"
        )
    if manifest.get("class_specific_gate") is not False:
        raise ValueError("Class-specific gates are not allowed in this transfer")
    if manifest.get("task_specific_alpha") is not False:
        raise ValueError("Task-specific alpha is not allowed in this transfer")
    if manifest.get("test_used_for_selection") is not False:
        raise ValueError("Source manifest must not use test data for selection")
    if list(manifest.get("class_to_task", [])) != list(class_to_task_map()):
        raise ValueError("Source manifest has an invalid B5-C3 class route")

    manifest_classnames = list(manifest.get("classnames", []))
    if len(manifest_classnames) != TASK_CLASS_RANGES[-1][1]:
        raise ValueError("Source manifest must contain all 26 EMOTIC class names")
    if classnames is not None and manifest_classnames != list(classnames):
        raise ValueError("CLS Adapter Bank and DDP class orders differ")
    return manifest_classnames


class TaskRoutedCLSToFinalTokenTransferBank(nn.Module):
    """Frozen class-routed CLS weights used before original DDP pooling."""

    def __init__(
        self,
        adapters: Mapping[int, nn.Module],
        application_mode: str,
        *,
        source_provenance: Optional[Mapping] = None,
    ):
        super().__init__()
        self.application_mode = _validate_application_mode(application_mode)
        if not adapters:
            raise ValueError("Transfer Adapter Bank must contain task 0")
        task_ids = sorted(int(task_id) for task_id in adapters)
        if task_ids != list(range(task_ids[-1] + 1)):
            raise ValueError(
                "Transfer Adapter Bank tasks must be contiguous from task 0; "
                f"got {task_ids}"
            )
        if task_ids[-1] >= len(TASK_CLASS_RANGES):
            raise ValueError(f"Invalid transfer task id {task_ids[-1]}")

        self.adapters = nn.ModuleDict(
            {str(task_id): adapters[task_id] for task_id in task_ids}
        )
        for adapter in self.adapters.values():
            residual_scale = getattr(adapter, "residual_scale", None)
            if residual_scale is None or not math.isclose(
                float(residual_scale),
                TRANSFER_RESIDUAL_SCALE,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Every transferred Adapter must use residual_scale=0.03"
                )
            adapter.eval()
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)

        provenance = {} if source_provenance is None else dict(source_provenance)
        self._source_provenance = copy.deepcopy(provenance)

    @property
    def max_task(self) -> int:
        return max(int(key) for key in self.adapters.keys())

    @property
    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def residual_scale(self) -> float:
        return TRANSFER_RESIDUAL_SCALE

    @property
    def source_manifest_path(self) -> Optional[str]:
        return self._source_provenance.get("source_manifest")

    @property
    def source_manifest_sha256(self) -> Optional[str]:
        return self._source_provenance.get("source_manifest_sha256")

    @property
    def source_checkpoint_hashes(self) -> Dict[int, str]:
        entries = self._source_provenance.get("source_checkpoints", {})
        return {
            int(task_id): str(entry["sha256"])
            for task_id, entry in entries.items()
        }

    @property
    def source_checkpoint_paths(self) -> Dict[int, str]:
        entries = self._source_provenance.get("source_checkpoints", {})
        return {
            int(task_id): str(entry["path"])
            for task_id, entry in entries.items()
        }

    def transfer_provenance(self) -> dict:
        """Return a JSON-serializable, mutation-safe provenance record."""

        return copy.deepcopy(self._source_provenance)

    def adapt_token_features(
        self,
        token_features: torch.Tensor,
        seen_classes: int,
        return_aux: bool = False,
    ):
        """Apply routed CLS weights and preserve DDP's token tensor layout."""

        if token_features.ndim != 4:
            raise ValueError("token_features must be [batch, 2K, dim, tokens]")
        seen_classes = int(seen_classes)
        if seen_classes <= 0:
            raise ValueError("seen_classes must be positive")
        if token_features.shape[1] != 2 * seen_classes:
            raise ValueError(
                f"Expected {2 * seen_classes} paths, got {token_features.shape[1]}"
            )
        if token_features.shape[-1] <= 0:
            raise ValueError("token_features must contain at least one token")
        required_max_task = class_task_id(seen_classes - 1)
        if self.max_task < required_max_task:
            raise ValueError(
                f"Seen classes require task {required_max_task}, but bank only "
                f"contains tasks 0..{self.max_task}"
            )

        # Point-wise Adapters consume [B, paths, tokens, D].
        source = token_features.permute(0, 1, 3, 2).float()
        adapted = source.clone()
        diagnostics = []
        token_slice = (
            slice(None) if self.application_mode == "all_tokens" else slice(0, 1)
        )
        adapted_token_count = (
            int(source.shape[2]) if self.application_mode == "all_tokens" else 1
        )

        for task_id in range(required_max_task + 1):
            low, task_high = task_class_range(task_id)
            high = min(task_high, seen_classes)
            if low >= high:
                continue
            negative = torch.arange(low, high, device=source.device)
            positive = torch.arange(
                seen_classes + low,
                seen_classes + high,
                device=source.device,
            )
            path_indices = torch.cat([negative, positive])
            task_tokens = source[:, path_indices, token_slice, :]
            task_adapted, task_original, task_residual = self.adapters[
                str(task_id)
            ](task_tokens)
            adapted[:, path_indices, token_slice, :] = task_adapted
            if return_aux:
                diagnostics.append(
                    {
                        "task_id": task_id,
                        "mean_residual_norm": float(
                            task_residual.detach().float().norm(dim=-1).mean()
                        ),
                        "mean_feature_change": float(
                            (task_adapted.detach() - task_original.detach())
                            .norm(dim=-1)
                            .mean()
                        ),
                        "adapted_tokens_per_path": adapted_token_count,
                    }
                )

        result = adapted.permute(0, 1, 3, 2).contiguous()
        if result.shape != token_features.shape:
            raise RuntimeError("CLS transfer changed the DDP token tensor shape")
        if return_aux:
            return result, {
                "application_mode": self.application_mode,
                "residual_scale": TRANSFER_RESIDUAL_SCALE,
                "tasks": diagnostics,
            }
        return result

    @classmethod
    def from_cls_manifest(
        cls,
        manifest_path: Union[str, Path],
        *,
        application_mode: str,
        device: Union[str, torch.device] = "cpu",
        max_task: Optional[int] = None,
        classnames: Optional[Sequence[str]] = None,
    ) -> "TaskRoutedCLSToFinalTokenTransferBank":
        """Load and explicitly reinterpret a legacy CLS Adapter Bank."""

        application_mode = _validate_application_mode(application_mode)
        manifest_path = Path(manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_classnames = _validate_cls_manifest(
            manifest,
            classnames=classnames,
        )

        raw_adapters = manifest.get("adapters", {})
        available = sorted(int(key) for key in raw_adapters)
        if not available:
            raise ValueError("CLS Adapter Bank manifest contains no Adapters")
        if available != list(range(available[-1] + 1)):
            raise ValueError(
                "CLS manifest tasks must be contiguous from task 0; got "
                f"{available}"
            )
        if available[-1] >= len(TASK_CLASS_RANGES):
            raise ValueError(f"Invalid source task id {available[-1]}")

        resolved_max = available[-1] if max_task is None else int(max_task)
        if not 0 <= resolved_max < len(TASK_CLASS_RANGES):
            raise ValueError(
                f"max_task must be in [0, {len(TASK_CLASS_RANGES) - 1}]"
            )
        expected = list(range(resolved_max + 1))
        if not all(task_id in available for task_id in expected):
            raise ValueError(
                f"Manifest does not contain every task in {expected}; got "
                f"{available}"
            )

        training_mode = manifest.get("training_mode")
        seed = manifest.get("seed")
        if training_mode not in ("full", "16shot"):
            raise ValueError(f"Unsupported source training_mode {training_mode!r}")
        if seed is None:
            raise ValueError("Source manifest does not contain a seed")

        adapters: Dict[int, SharedResidualFinalTokenAdapter] = {}
        source_checkpoints = {}
        common_feature_dim = None
        for task_id in expected:
            entry = raw_adapters[str(task_id)]
            if int(entry.get("task_id", -1)) != task_id:
                raise ValueError(f"Source manifest entry {task_id} has wrong task_id")
            if list(entry.get("class_range", [])) != list(
                task_class_range(task_id)
            ):
                raise ValueError(
                    f"Source manifest entry {task_id} has invalid class_range"
                )
            checkpoint_path = (
                manifest_path.parent / str(entry["checkpoint"])
            ).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            actual_hash = file_sha256(checkpoint_path)
            if actual_hash != entry.get("sha256"):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            metadata = validate_task_adapter_checkpoint(
                checkpoint,
                task_id=task_id,
                training_mode=training_mode,
                seed=int(seed),
                classnames=manifest_classnames,
            )
            if (
                int(metadata.get("schema_version", -1))
                != CHECKPOINT_SCHEMA_VERSION
            ):
                raise ValueError("Unsupported source CLS checkpoint schema")
            checkpoint_args = checkpoint.get("args")
            if not isinstance(checkpoint_args, Mapping):
                raise ValueError("Source CLS checkpoint does not contain args")
            if checkpoint_args.get("formula") != CLS_SOURCE_FORMULA:
                raise ValueError("Source checkpoint is not a CLS feature Adapter")
            if (
                checkpoint_args.get("correction_mode")
                != CLS_SOURCE_CORRECTION_MODE
            ):
                raise ValueError(
                    "Source checkpoint does not use linear_residual correction"
                )

            feature_dim = int(checkpoint_args.get("feature_dim", 512))
            bottleneck_dim = int(checkpoint_args["adapter_dim"])
            if common_feature_dim is None:
                common_feature_dim = feature_dim
            elif feature_dim != common_feature_dim:
                raise ValueError(
                    "All source CLS Adapters must share one feature dimension"
                )
            adapter = SharedResidualFinalTokenAdapter(
                feature_dim=feature_dim,
                bottleneck_dim=bottleneck_dim,
                residual_scale=TRANSFER_RESIDUAL_SCALE,
            )
            adapter.load_state_dict(checkpoint["model"], strict=True)
            adapters[task_id] = adapter.to(device)
            source_checkpoints[str(task_id)] = {
                "path": str(checkpoint_path),
                "sha256": actual_hash,
            }

        provenance = {
            "schema_version": CLS_TO_FINAL_TOKEN_TRANSFER_SCHEMA_VERSION,
            "transfer": "prompt_free_cls_weights_to_final_tokens",
            "application_mode": application_mode,
            "residual_scale": TRANSFER_RESIDUAL_SCALE,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "source_manifest_schema_version": int(manifest["schema_version"]),
            "source_training_mode": training_mode,
            "source_seed": int(seed),
            "source_inference_formula": manifest["inference_formula"],
            "source_checkpoint_selection": "existing_validation_selected_weights",
            "test_used_for_transfer_selection": False,
            "source_checkpoints": source_checkpoints,
        }
        return cls(
            adapters,
            application_mode=application_mode,
            source_provenance=provenance,
        ).to(device)

