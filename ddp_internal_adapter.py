import torch
import torch.nn as nn
import torch.nn.functional as F


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
