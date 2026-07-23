"""Masked asymmetric losses for strict EMOTIC incremental supervision.

The functions in this module operate on logits.  Old and future labels are
removed exclusively through ``supervision_mask`` before reduction; class
frequency statistics are computed from the same visible entries.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


LOSS_NAMES = ("weighted_bce", "asl", "bal_paper", "bal_release")


def visible_class_statistics(
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Count supervised positives and negatives for each active class."""

    if targets.shape != supervision_mask.shape or targets.ndim != 2:
        raise ValueError(
            "targets and supervision_mask must have equal [samples, classes] "
            f"shapes; got {tuple(targets.shape)} and "
            f"{tuple(supervision_mask.shape)}"
        )
    mask = supervision_mask.bool()
    positive = targets.gt(0.5)
    positives = (positive & mask).sum(dim=0).float()
    negatives = ((~positive) & mask).sum(dim=0).float()
    supervised = mask.sum(dim=0).float()
    if (supervised <= 0).any():
        missing = torch.nonzero(supervised <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Classes without supervised entries: {missing}")
    return {
        "positives": positives,
        "negatives": negatives,
        "supervised": supervised,
    }


def balanced_positive_class_weights(
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
    power: float = 1.6,
) -> torch.Tensor:
    """Return the BAL inverse-positive-frequency weights in ``(0, 1]``.

    The common factor ``N`` in ``(N / N_c)^power`` cancels after max
    normalization, so only visible positive counts are required.
    """

    if power < 0:
        raise ValueError("power must be non-negative")
    statistics = visible_class_statistics(targets, supervision_mask)
    positives = statistics["positives"]
    if (positives <= 0).any():
        missing = torch.nonzero(positives <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"BAL requires a visible positive for every class: {missing}")
    inverse = positives.pow(-float(power))
    return inverse / inverse.max().clamp_min(torch.finfo(inverse.dtype).tiny)


class MaskedAsymmetricLoss(nn.Module):
    """ASL/BAL with strict partial-label masking and stable element reduction."""

    def __init__(
        self,
        gamma_neg: float = 9.8,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        smoothing_num_classes: int = 26,
        positive_weight_mode: str = "paper",
        normalize_logits_by_batch_max: bool = False,
        detach_focal_weight: bool = True,
    ):
        super().__init__()
        if gamma_neg < 0 or gamma_pos < 0:
            raise ValueError("gamma_neg and gamma_pos must be non-negative")
        if clip < 0 or clip >= 1:
            raise ValueError("clip must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if smoothing_num_classes <= 0:
            raise ValueError("smoothing_num_classes must be positive")
        if positive_weight_mode not in ("paper", "one_plus"):
            raise ValueError("positive_weight_mode must be 'paper' or 'one_plus'")

        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)
        self.smoothing_num_classes = int(smoothing_num_classes)
        self.positive_weight_mode = positive_weight_mode
        self.normalize_logits_by_batch_max = bool(normalize_logits_by_batch_max)
        self.detach_focal_weight = bool(detach_focal_weight)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            if class_weights.ndim != 1:
                raise ValueError("class_weights must have shape [classes]")
            self.register_buffer("class_weights", class_weights.detach().float())

    def _normalized_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.normalize_logits_by_batch_max:
            return logits
        maximum = logits.detach().max()
        if not torch.isfinite(maximum) or maximum.abs() <= self.eps:
            raise FloatingPointError(
                "LM-CLIP batch-max logit normalization requires a finite, "
                "non-zero maximum"
            )
        return logits / maximum

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        supervision_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape or targets.shape != supervision_mask.shape:
            raise ValueError(
                "logits, targets, and supervision_mask must have equal shapes; "
                f"got {tuple(logits.shape)}, {tuple(targets.shape)}, and "
                f"{tuple(supervision_mask.shape)}"
            )
        if logits.ndim != 2:
            raise ValueError("Expected [batch, classes] tensors")
        if self.class_weights is not None and self.class_weights.numel() != logits.shape[1]:
            raise ValueError(
                f"Expected {logits.shape[1]} class weights, got "
                f"{self.class_weights.numel()}"
            )

        raw_targets = targets.float()
        anti_targets = 1.0 - raw_targets
        positive_targets = raw_targets
        if self.label_smoothing > 0:
            positive_targets = (
                (1.0 - self.label_smoothing) * raw_targets
                + self.label_smoothing / self.smoothing_num_classes
            )

        probabilities = torch.sigmoid(self._normalized_logits(logits.float()))
        positive_probability = probabilities
        negative_probability = 1.0 - probabilities
        if self.clip > 0:
            negative_probability = (negative_probability + self.clip).clamp(max=1.0)

        positive_loss = positive_targets * torch.log(
            positive_probability.clamp_min(self.eps)
        )
        if self.class_weights is not None:
            weights = self.class_weights.to(
                device=logits.device, dtype=positive_loss.dtype
            )
            if self.positive_weight_mode == "one_plus":
                weights = 1.0 + weights
            positive_loss = positive_loss * weights.unsqueeze(0)
        negative_loss = anti_targets * torch.log(
            negative_probability.clamp_min(self.eps)
        )

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = (
                positive_probability * positive_targets
                + negative_probability * anti_targets
            )
            gamma = (
                self.gamma_pos * positive_targets
                + self.gamma_neg * anti_targets
            )
            focal_weight = (1.0 - pt).clamp_min(0.0).pow(gamma)
            if self.detach_focal_weight:
                focal_weight = focal_weight.detach()
            element_loss = -(positive_loss + negative_loss) * focal_weight
        else:
            element_loss = -(positive_loss + negative_loss)

        mask = supervision_mask.to(dtype=element_loss.dtype)
        denominator = mask.sum()
        if denominator.item() <= 0:
            raise ValueError("supervision_mask contains no supervised entries")
        loss = (element_loss * mask).sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("ASL/BAL produced a non-finite loss")
        return loss


def build_asymmetric_loss(
    loss_name: str,
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
    gamma_neg: float,
    gamma_pos: float,
    clip: float,
    bal_weight_power: float,
    bal_label_smoothing: float,
    smoothing_num_classes: int,
) -> tuple[MaskedAsymmetricLoss, Dict]:
    """Build one loss and return JSON-serializable train-only diagnostics."""

    if loss_name not in ("asl", "bal_paper", "bal_release"):
        raise ValueError(f"Unsupported asymmetric loss: {loss_name}")
    statistics = visible_class_statistics(targets, supervision_mask)
    weights = None
    positive_weight_mode = "paper"
    normalize_logits = False
    smoothing = 0.0
    if loss_name.startswith("bal"):
        weights = balanced_positive_class_weights(
            targets, supervision_mask, power=bal_weight_power
        )
        smoothing = float(bal_label_smoothing)
        if loss_name == "bal_release":
            positive_weight_mode = "one_plus"
            normalize_logits = True

    loss = MaskedAsymmetricLoss(
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=clip,
        class_weights=weights,
        label_smoothing=smoothing,
        smoothing_num_classes=smoothing_num_classes,
        positive_weight_mode=positive_weight_mode,
        normalize_logits_by_batch_max=normalize_logits,
    )
    diagnostics = {
        "loss_name": loss_name,
        "gamma_neg": float(gamma_neg),
        "gamma_pos": float(gamma_pos),
        "clip": float(clip),
        "bal_weight_power": (
            float(bal_weight_power) if loss_name.startswith("bal") else None
        ),
        "label_smoothing": smoothing,
        "smoothing_num_classes": int(smoothing_num_classes),
        "positive_weight_mode": positive_weight_mode,
        "normalize_logits_by_batch_max": normalize_logits,
        "visible_positives": statistics["positives"].tolist(),
        "visible_negatives": statistics["negatives"].tolist(),
        "supervised_entries": statistics["supervised"].tolist(),
        "class_weights": None if weights is None else weights.tolist(),
    }
    return loss, diagnostics
