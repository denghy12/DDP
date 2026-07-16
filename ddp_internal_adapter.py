import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CORRECTION_MODES = (
    "linear_residual",
    "cosine_difference",
    "feature_correction",
)


def norm_preserving_feature_correction(
    pooled_features,
    feature_delta,
    eps=1e-12,
):
    """Apply an Adapter offset while preserving each pooled-feature norm."""
    if pooled_features.shape != feature_delta.shape:
        raise ValueError(
            "pooled_features and feature_delta must have identical shapes; "
            f"got {tuple(pooled_features.shape)} and {tuple(feature_delta.shape)}"
        )
    if pooled_features.ndim != 3:
        raise ValueError(
            "Expected pooled_features [batch, paths, dim], got "
            f"{tuple(pooled_features.shape)}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    pooled = pooled_features.float()
    shifted = pooled + feature_delta.float()
    original_norm = pooled.norm(dim=-1, keepdim=True)
    shifted_norm = shifted.norm(dim=-1, keepdim=True)
    pooled_direction = pooled / original_norm.clamp_min(float(eps))
    shifted_direction = shifted / shifted_norm.clamp_min(float(eps))
    direction = torch.where(
        shifted_norm > float(eps), shifted_direction, pooled_direction
    )
    corrected = original_norm * direction
    zero_delta = feature_delta.float().abs().amax(dim=-1, keepdim=True).eq(0)
    return torch.where(zero_delta, pooled, corrected)


def feature_logit_correction(
    adapted,
    original,
    text_features,
    mode="linear_residual",
    logit_scale=100.0,
    pooled_features=None,
):
    """Convert an Adapter feature change into per-path DDP logit corrections.

    ``linear_residual`` exactly preserves the historical implementation:
    scale * text^T(adapted - original).

    ``cosine_difference`` measures the change in normalized CLIP similarity:
    scale * [cos(text, adapted) - cos(text, original)].

    ``feature_correction`` adds the Adapter offset to the original DDP pooled
    representation, preserves its norm, and measures the change produced by
    the original DDP dot-product head. Adding this numerical difference to the
    cached DDP logits is algebraically the same as applying that head directly
    to the corrected pooled representation.
    """
    if mode not in CORRECTION_MODES:
        raise ValueError(
            f"Unknown correction mode '{mode}'; expected one of "
            f"{CORRECTION_MODES}"
        )
    if adapted.shape != original.shape:
        raise ValueError(
            "adapted and original features must have identical shapes; "
            f"got {tuple(adapted.shape)} and {tuple(original.shape)}"
        )
    if adapted.ndim != 3 or text_features.ndim != 2:
        raise ValueError(
            "Expected adapted/original [batch, paths, dim] and text "
            f"[paths, dim], got {tuple(adapted.shape)} and "
            f"{tuple(text_features.shape)}"
        )
    if adapted.shape[1:] != text_features.shape:
        raise ValueError(
            "Feature paths/dimension must match text features; got "
            f"{tuple(adapted.shape[1:])} and {tuple(text_features.shape)}"
        )
    if logit_scale <= 0:
        raise ValueError("logit_scale must be positive")

    text = text_features.float()
    if mode == "linear_residual":
        return float(logit_scale) * torch.einsum(
            "bkd,kd->bk", adapted.float() - original.float(), text
        )

    if mode == "feature_correction":
        if pooled_features is None:
            raise ValueError(
                "feature_correction requires the original DDP pooled_features"
            )
        if pooled_features.shape != original.shape:
            raise ValueError(
                "pooled_features must match Adapter path features; got "
                f"{tuple(pooled_features.shape)} and {tuple(original.shape)}"
            )
        pooled = pooled_features.float()
        corrected = norm_preserving_feature_correction(
            pooled,
            adapted.float() - original.float(),
        )
        return float(logit_scale) * torch.einsum(
            "bkd,kd->bk", corrected - pooled, text
        )

    text = F.normalize(text, dim=-1)
    adapted_similarity = torch.einsum(
        "bkd,kd->bk", F.normalize(adapted.float(), dim=-1), text
    )
    original_similarity = torch.einsum(
        "bkd,kd->bk", F.normalize(original.float(), dim=-1), text
    )
    return float(logit_scale) * (adapted_similarity - original_similarity)


class SharedResidualFeatureAdapter(nn.Module):
    """One residual mapping shared by every DDP class and +/- path."""

    def __init__(self, feature_dim=512, bottleneck_dim=128, residual_scale=0.1):
        super().__init__()
        if feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("feature_dim and bottleneck_dim must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(feature_dim, bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {features.shape[-1]}"
            )
        original = features.float()
        adapter_input = F.normalize(original, dim=-1)
        residual = self.up(self.activation(self.down(adapter_input)))
        adapted = original + self.residual_scale * residual
        return adapted, original


class PromptFreePrototypeObjective(nn.Module):
    """Training-only prototype head for the DDP-owned global CLIP route.

    The head owns no image or text encoder.  It trains the same shared Adapter
    that is later attached to prompted DDP CLS features, while the fixed
    positive/negative prototypes and the learnable logit scale are discarded
    after auxiliary training.
    """

    def __init__(
        self,
        adapter,
        positive_prototypes,
        negative_prototypes,
        initial_logit_scale=10.0,
    ):
        super().__init__()
        if positive_prototypes.shape != negative_prototypes.shape:
            raise ValueError(
                "Positive and negative prototypes must have equal shapes"
            )
        if positive_prototypes.ndim != 2:
            raise ValueError(
                "Prototypes must have shape [num_classes, feature_dim]"
            )
        if positive_prototypes.shape[1] != adapter.feature_dim:
            raise ValueError(
                "Prototype dimension must match the shared Adapter dimension"
            )
        if initial_logit_scale <= 0:
            raise ValueError("initial_logit_scale must be positive")

        self.adapter = adapter
        self.logit_scale = nn.Parameter(
            torch.tensor(
                math.log(float(initial_logit_scale)), dtype=torch.float32
            )
        )
        self.register_buffer(
            "positive_prototypes",
            F.normalize(positive_prototypes.float(), dim=-1),
        )
        self.register_buffer(
            "negative_prototypes",
            F.normalize(negative_prototypes.float(), dim=-1),
        )

    def forward(self, global_features):
        adapted, original = self.adapter(global_features)
        adapted_normalized = F.normalize(adapted.float(), dim=-1)
        original_normalized = F.normalize(original.float(), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        positive_similarity = adapted_normalized @ self.positive_prototypes.T
        negative_similarity = adapted_normalized @ self.negative_prototypes.T
        logits = scale * (positive_similarity - negative_similarity)
        return logits, adapted_normalized, original_normalized


def feature_identity_loss(adapted, original):
    return 1.0 - F.cosine_similarity(adapted, original, dim=-1).mean()


def masked_ddp_bce(logits, targets, supervision_mask, pos_weight=None):
    """Masked BCE on DDP's positive-minus-negative two-way logit margin."""
    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError("DDP logits must have shape [batch, 2, classes]")
    margin = logits[:, 1, :] - logits[:, 0, :]
    if targets.shape != margin.shape or supervision_mask.shape != margin.shape:
        raise ValueError(
            "targets and supervision_mask must match the DDP margin shape; "
            f"got margin={tuple(margin.shape)}, targets={tuple(targets.shape)}, "
            f"mask={tuple(supervision_mask.shape)}"
        )
    losses = F.binary_cross_entropy_with_logits(
        margin,
        targets.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    mask = supervision_mask.to(dtype=losses.dtype)
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("supervision_mask contains no supervised entries")
    return (losses * mask).sum() / denominator
