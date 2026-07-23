"""Final-token Adapter utilities for EMOTIC B5-C3.

The Adapter operates on all 197 projected ViT tokens of every DDP +/- path.
After adaptation, the original DDP token-text similarities, shared attention
pooling, and two-way path logits are recomputed without an auxiliary logit
head or score fusion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from emotic_task_adapter_bank import (
    TASK_CLASS_RANGES,
    class_task_id,
    class_to_task_map,
    task_class_range,
)


FINAL_TOKEN_CHECKPOINT_SCHEMA_VERSION = 2
FINAL_TOKEN_BANK_SCHEMA_VERSION = 2
FINAL_TOKEN_FORMULA = "adapt_197_tokens_then_original_ddp_pooling"


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ddp_pool_final_tokens(
    token_features: torch.Tensor,
    text_features: torch.Tensor,
):
    """Apply the exact original DDP token-attention pooling equations.

    Args:
        token_features: normalized final visual tokens in ``[B, 2K, D, N]``.
        text_features: normalized path text features in ``[2K, D]`` and DDP's
            existing ``[negative paths, positive paths]`` order.

    Returns:
        ``(pooled_features, path_logits, token_logits, path_weights)``.

    The reference half used to construct the shared attention intentionally
    matches the existing DDP implementation.  This helper centralizes that
    behavior so the baseline and Final-token route cannot silently diverge.
    """

    if token_features.ndim != 4:
        raise ValueError(
            "token_features must be [batch, 2K, dim, tokens], got "
            f"{tuple(token_features.shape)}"
        )
    if text_features.ndim != 2:
        raise ValueError(
            "text_features must be [2K, dim], got "
            f"{tuple(text_features.shape)}"
        )
    batch, path_count, feature_dim, token_count = token_features.shape
    if path_count == 0 or path_count % 2:
        raise ValueError("Final-token DDP paths must contain equal +/- halves")
    if text_features.shape != (path_count, feature_dim):
        raise ValueError(
            "text_features must match token paths and dimension; got "
            f"{tuple(text_features.shape)} for tokens "
            f"{tuple(token_features.shape)}"
        )
    if token_count <= 0 or batch <= 0:
        raise ValueError("token_features must contain samples and tokens")

    seen_classes = path_count // 2
    token_logits = 20 * torch.einsum(
        "bkdn,kd->bkn", token_features, text_features
    )
    reference_weights = F.softmax(token_logits[:, seen_classes:, :], dim=-1)
    path_weights = torch.cat([reference_weights, reference_weights], dim=1)
    pooled_features = torch.einsum(
        "bkdn,bkn->bkd", token_features, path_weights
    )
    path_logits = 5 * (token_logits * path_weights).sum(dim=-1)
    return pooled_features, path_logits, token_logits, path_weights


def final_token_drift_tensors(
    adapted_tokens: torch.Tensor,
    original_tokens: torch.Tensor,
    text_features: torch.Tensor,
):
    """Return per-example diagnostics without modifying the training loss.

    ``adapted_tokens`` and ``original_tokens`` use ``[B, 2K, N, D]``.  The
    returned tensors retain their natural sample/path/token axes so callers
    can aggregate means, percentiles, and maxima over a complete split.
    Attention KL follows ``KL(original || adapted)`` on the positive-path
    attention that DDP shares with the matching negative path.
    """

    if adapted_tokens.shape != original_tokens.shape:
        raise ValueError("adapted and original final tokens must match")
    if adapted_tokens.ndim != 4:
        raise ValueError("final tokens must be [batch, paths, tokens, dim]")
    batch, path_count, token_count, feature_dim = adapted_tokens.shape
    if path_count == 0 or path_count % 2:
        raise ValueError("final-token diagnostics require equal +/- paths")
    if text_features.shape != (path_count, feature_dim):
        raise ValueError("text features do not match final-token paths")
    if batch <= 0 or token_count <= 1:
        raise ValueError("final-token diagnostics require samples and patches")

    original_ddp = original_tokens.float().permute(0, 1, 3, 2).contiguous()
    adapted_ddp = adapted_tokens.float().permute(0, 1, 3, 2).contiguous()
    (
        original_pooled,
        original_logits,
        original_token_logits,
        original_weights,
    ) = ddp_pool_final_tokens(original_ddp, text_features.float())
    (
        adapted_pooled,
        adapted_logits,
        adapted_token_logits,
        adapted_weights,
    ) = ddp_pool_final_tokens(adapted_ddp, text_features.float())
    seen_classes = path_count // 2
    # Compute KL from log-softmax logits rather than log(clamped softmax).
    # DDP's token logits span roughly [-20, 20], so some probabilities can be
    # far below 1e-12.  Clamping those probabilities before log creates a
    # discontinuous, extremely steep backward path when a non-zero Task-0
    # Adapter initializes later tasks.  This formulation is mathematically
    # equivalent to KL(original || adapted) and keeps its gradient finite.
    original_log_attention = F.log_softmax(
        original_token_logits[:, seen_classes:, :].float(), dim=-1
    )
    adapted_log_attention = F.log_softmax(
        adapted_token_logits[:, seen_classes:, :].float(), dim=-1
    )
    original_attention = original_log_attention.exp()
    attention_kl = (
        original_attention
        * (original_log_attention - adapted_log_attention)
    ).sum(dim=-1).clamp_min(0.0)
    pooled_cosine_drift = (
        1.0
        - F.cosine_similarity(
            adapted_pooled.float(), original_pooled.float(), dim=-1
        )
    ).clamp_min(0.0)
    path_logit_delta = adapted_logits.float() - original_logits.float()
    token_delta_l2 = (
        adapted_tokens.float() - original_tokens.float()
    ).norm(dim=-1)
    return {
        "attention_kl_original_to_adapted": attention_kl,
        "pooled_feature_cosine_drift": pooled_cosine_drift,
        "path_logit_delta": path_logit_delta,
        "token_delta_l2": token_delta_l2,
    }


def final_token_pooling_aware_losses(
    adapted_tokens: torch.Tensor,
    original_tokens: torch.Tensor,
    text_features: torch.Tensor,
    margin_beta: float = 1.0,
):
    """Compute differentiable DDP-structure preservation losses.

    The frozen original DDP route is the teacher.  The margin term preserves
    the two-way class evidence ``positive path - negative path`` rather than
    penalizing a harmless common shift of both path logits.
    """

    if margin_beta <= 0:
        raise ValueError("margin_beta must be positive")
    drift = final_token_drift_tensors(
        adapted_tokens, original_tokens, text_features
    )
    path_delta = drift["path_logit_delta"]
    if path_delta.shape[1] % 2:
        raise ValueError("path-logit drift must contain equal +/- halves")
    margin_delta = (
        path_delta[:, path_delta.shape[1] // 2 :]
        - path_delta[:, : path_delta.shape[1] // 2]
    )
    margin_loss = F.smooth_l1_loss(
        margin_delta,
        torch.zeros_like(margin_delta),
        beta=float(margin_beta),
    )
    return {
        "pooling_loss": drift["pooled_feature_cosine_drift"].mean(),
        "attention_loss": drift[
            "attention_kl_original_to_adapted"
        ].mean(),
        "margin_loss": margin_loss,
        "margin_delta": margin_delta,
    }


class SharedResidualFinalTokenAdapter(nn.Module):
    """One point-wise residual MLP shared by all tokens and +/- paths."""

    def __init__(
        self,
        feature_dim: int = 512,
        bottleneck_dim: int = 128,
        residual_scale: float = 0.03,
    ):
        super().__init__()
        if feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("feature_dim and bottleneck_dim must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(self.feature_dim, self.bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.bottleneck_dim, self.feature_dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, token_features: torch.Tensor):
        """Adapt tokens whose final axis is the CLIP feature dimension.

        The output is normalized before it re-enters DDP pooling.  The up
        projection is zero initialized, so the route starts as a numerical
        identity while the normalization branch remains differentiable.  Do
        not replace this with a data-dependent exact-identity ``torch.where``:
        selecting the original input when the residual is zero disconnects the
        zero-initialized up projection from the classification loss forever.
        """

        if token_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got "
                f"{token_features.shape[-1]}"
            )
        original = token_features.float()
        adapter_input = F.normalize(original, dim=-1)
        residual = self.up(self.activation(self.down(adapter_input)))
        scaled_residual = self.residual_scale * residual
        adapted = F.normalize(original + scaled_residual, dim=-1)
        return adapted, original, residual


def final_token_identity_loss(
    adapted_tokens: torch.Tensor,
    original_tokens: torch.Tensor,
) -> torch.Tensor:
    if adapted_tokens.shape != original_tokens.shape:
        raise ValueError("adapted and original token tensors must match")
    return 1.0 - F.cosine_similarity(
        adapted_tokens.float(), original_tokens.float(), dim=-1
    ).mean()


def validate_final_token_checkpoint(
    checkpoint: Mapping,
    task_id: Optional[int] = None,
    training_mode: Optional[str] = None,
    seed: Optional[int] = None,
    classnames: Optional[Sequence[str]] = None,
) -> Mapping:
    metadata = checkpoint.get("final_token_adapter")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint does not contain final_token_adapter metadata")
    if int(metadata.get("schema_version", -1)) != FINAL_TOKEN_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported Final-token checkpoint schema")
    checkpoint_task = int(metadata["task_id"])
    expected_range = list(task_class_range(checkpoint_task))
    if list(metadata.get("class_range", [])) != expected_range:
        raise ValueError(
            f"Task {checkpoint_task} has invalid class_range; "
            f"expected {expected_range}"
        )
    if metadata.get("formula") != FINAL_TOKEN_FORMULA:
        raise ValueError("Checkpoint is not a Final-token Adapter checkpoint")
    if task_id is not None and checkpoint_task != int(task_id):
        raise ValueError(f"Expected task {task_id}, found task {checkpoint_task}")
    if training_mode is not None and metadata.get("training_mode") != training_mode:
        raise ValueError(
            f"Expected mode {training_mode}, found {metadata.get('training_mode')}"
        )
    if seed is not None and int(metadata.get("seed", -1)) != int(seed):
        raise ValueError(f"Expected seed {seed}, found {metadata.get('seed')}")
    if classnames is not None and list(checkpoint.get("classnames", [])) != list(
        classnames
    ):
        raise ValueError("Adapter and DDP class orders differ")
    if checkpoint.get("model") is None:
        raise ValueError("Final-token checkpoint does not contain model weights")
    return metadata


class TaskRoutedFinalTokenAdapterBank(nn.Module):
    """Route both +/- token paths of each class to its introduction Adapter."""

    def __init__(self, adapters: Mapping[int, nn.Module]):
        super().__init__()
        if not adapters:
            raise ValueError("Final-token Adapter Bank must contain task 0")
        task_ids = sorted(int(task_id) for task_id in adapters)
        if task_ids != list(range(task_ids[-1] + 1)):
            raise ValueError(
                "Final-token bank tasks must be contiguous from task 0; got "
                f"{task_ids}"
            )
        self.adapters = nn.ModuleDict(
            {str(task_id): adapters[task_id] for task_id in task_ids}
        )
        for adapter in self.adapters.values():
            adapter.eval()
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)

    @property
    def max_task(self) -> int:
        return max(int(key) for key in self.adapters.keys())

    @property
    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def adapt_token_features(
        self,
        token_features: torch.Tensor,
        seen_classes: int,
        return_aux: bool = False,
    ):
        if token_features.ndim != 4:
            raise ValueError("token_features must be [batch, 2K, dim, tokens]")
        seen_classes = int(seen_classes)
        if token_features.shape[1] != 2 * seen_classes:
            raise ValueError(
                f"Expected {2 * seen_classes} paths, got {token_features.shape[1]}"
            )
        required_max_task = class_task_id(seen_classes - 1)
        if self.max_task < required_max_task:
            raise ValueError(
                f"Seen classes require task {required_max_task}, but bank only "
                f"contains tasks 0..{self.max_task}"
            )

        # Adapters consume [B, paths, tokens, D].
        source = token_features.permute(0, 1, 3, 2).float()
        adapted = source.clone()
        residual_norms = []
        for task_id in range(required_max_task + 1):
            low, task_high = task_class_range(task_id)
            high = min(task_high, seen_classes)
            if low >= high:
                continue
            negative = torch.arange(low, high, device=source.device)
            positive = torch.arange(
                seen_classes + low, seen_classes + high, device=source.device
            )
            path_indices = torch.cat([negative, positive])
            task_tokens = source[:, path_indices, :, :]
            task_adapted, task_original, task_residual = self.adapters[
                str(task_id)
            ](task_tokens)
            adapted[:, path_indices, :, :] = task_adapted
            if return_aux:
                residual_norms.append(
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
                    }
                )
        result = adapted.permute(0, 1, 3, 2).contiguous()
        if return_aux:
            return result, {"tasks": residual_norms}
        return result

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        device: Union[str, torch.device],
        max_task: Optional[int] = None,
        classnames: Optional[Sequence[str]] = None,
    ) -> "TaskRoutedFinalTokenAdapterBank":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != FINAL_TOKEN_BANK_SCHEMA_VERSION:
            raise ValueError("Unsupported Final-token bank manifest schema")
        if manifest.get("inference_formula") != FINAL_TOKEN_FORMULA:
            raise ValueError("Manifest is not a Final-token Adapter Bank")
        manifest_classnames = list(manifest.get("classnames", []))
        if classnames is not None and manifest_classnames != list(classnames):
            raise ValueError("Final-token bank and DDP class orders differ")
        available = sorted(int(key) for key in manifest.get("adapters", {}))
        if not available:
            raise ValueError("Final-token bank manifest contains no Adapters")
        resolved_max = available[-1] if max_task is None else int(max_task)
        expected = list(range(resolved_max + 1))
        if not all(task_id in available for task_id in expected):
            raise ValueError(
                f"Manifest does not contain every task in {expected}; got {available}"
            )

        adapters: Dict[int, SharedResidualFinalTokenAdapter] = {}
        for task_id in expected:
            entry = manifest["adapters"][str(task_id)]
            checkpoint_path = manifest_path.parent / entry["checkpoint"]
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if file_sha256(checkpoint_path) != entry.get("sha256"):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            validate_final_token_checkpoint(
                checkpoint,
                task_id=task_id,
                training_mode=manifest["training_mode"],
                seed=manifest["seed"],
                classnames=manifest_classnames,
            )
            checkpoint_args = checkpoint["args"]
            adapter = SharedResidualFinalTokenAdapter(
                feature_dim=int(checkpoint_args.get("feature_dim", 512)),
                bottleneck_dim=int(checkpoint_args["adapter_dim"]),
                residual_scale=float(manifest["residual_scale"]),
            )
            adapter.load_state_dict(checkpoint["model"], strict=True)
            adapters[task_id] = adapter.to(device)
        return cls(adapters).to(device)


def build_final_token_bank_manifest(
    bank_dir: Union[str, Path],
    training_mode: str,
    seed: int,
    classnames: Sequence[str],
    residual_scale: float = 0.03,
    selection: str = "fixed_last_epoch_no_validation_selection",
    training_protocol: Optional[dict] = None,
) -> dict:
    bank_dir = Path(bank_dir)
    adapters = {}
    for task_id in range(len(TASK_CLASS_RANGES)):
        checkpoint_path = bank_dir / f"task{task_id}" / "final_adapter.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = validate_final_token_checkpoint(
            checkpoint,
            task_id=task_id,
            training_mode=training_mode,
            seed=seed,
            classnames=classnames,
        )
        checkpoint_scale = float(checkpoint["args"]["residual_scale"])
        if checkpoint_scale != float(residual_scale):
            raise ValueError(
                f"Task {task_id} residual scale {checkpoint_scale} does not "
                f"match locked scale {residual_scale}"
            )
        checkpoint_steps = int(checkpoint["args"].get("max_optimizer_steps", 0))
        if selection == "fixed_optimizer_steps_no_validation_selection":
            expected_steps = int((training_protocol or {}).get("optimizer_steps", 0))
            if expected_steps <= 0 or checkpoint_steps != expected_steps:
                raise ValueError(
                    f"Task {task_id} max optimizer steps {checkpoint_steps} does "
                    f"not match locked value {expected_steps}"
                )
        adapters[str(task_id)] = {
            "task_id": task_id,
            "class_range": list(task_class_range(task_id)),
            "checkpoint": str(checkpoint_path.relative_to(bank_dir)),
            "sha256": file_sha256(checkpoint_path),
            "final_epoch": checkpoint.get("epoch"),
            "optimizer_steps": checkpoint.get("optimizer_steps"),
            "reporting_val_mAP": checkpoint.get("reporting_val_mAP"),
            "initialization": metadata.get("initialization"),
            "ddp_checkpoint": metadata.get("ddp_checkpoint"),
            "parameter_count": int(
                sum(value.numel() for value in checkpoint["model"].values())
            ),
        }
    return {
        "schema_version": FINAL_TOKEN_BANK_SCHEMA_VERSION,
        "name": "EMOTIC B5-C3 task-routed Final-token Adapter Bank",
        "training_mode": training_mode,
        "seed": int(seed),
        "inference_formula": FINAL_TOKEN_FORMULA,
        "residual_scale": float(residual_scale),
        "token_count": 197,
        "adapter_parameter_count_each": 131072,
        "adapter_bank_parameter_count": int(
            sum(entry["parameter_count"] for entry in adapters.values())
        ),
        "selection": selection,
        "training_protocol": dict(training_protocol or {}),
        "decision_threshold": 0.5,
        "class_specific_gate": False,
        "task_specific_alpha": False,
        "test_used_for_selection": False,
        "classnames": list(classnames),
        "class_to_task": list(class_to_task_map()),
        "adapters": adapters,
    }
