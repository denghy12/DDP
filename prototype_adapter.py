import math

import torch
import torch.nn as nn
import torch.nn.functional as F


POSITIVE_TEMPLATES = (
    "a photo of a person clearly feeling {}.",
    "this photo shows {} emotion.",
    "a person expressing {}.",
)

NEGATIVE_TEMPLATES = (
    "a photo of a person not feeling {}.",
    "this photo does not show {} emotion.",
    "a person without an expression of {}.",
)


class ResidualPrototypeAdapter(nn.Module):
    """Shared residual adapter with fixed positive/negative text prototypes."""

    def __init__(
        self,
        positive_prototypes,
        negative_prototypes,
        bottleneck_dim=128,
        residual_scale=0.1,
        initial_logit_scale=10.0,
    ):
        super().__init__()
        if positive_prototypes.shape != negative_prototypes.shape:
            raise ValueError("Positive and negative prototypes must have equal shapes")
        if positive_prototypes.ndim != 2:
            raise ValueError("Prototypes must have shape [num_classes, feature_dim]")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if initial_logit_scale <= 0:
            raise ValueError("initial_logit_scale must be positive")

        feature_dim = positive_prototypes.shape[1]
        self.down = nn.Linear(feature_dim, bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim, bias=False)
        self.residual_scale = float(residual_scale)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(initial_logit_scale), dtype=torch.float32)
        )

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

        self.register_buffer(
            "positive_prototypes", F.normalize(positive_prototypes.float(), dim=-1)
        )
        self.register_buffer(
            "negative_prototypes", F.normalize(negative_prototypes.float(), dim=-1)
        )

    def adapt(self, image_features):
        image_features = F.normalize(image_features.float(), dim=-1)
        residual = self.up(self.activation(self.down(image_features)))
        adapted = image_features + self.residual_scale * residual
        return F.normalize(adapted, dim=-1), image_features

    def forward(self, image_features):
        adapted, original = self.adapt(image_features)
        positive_similarity = adapted @ self.positive_prototypes.t()
        negative_similarity = adapted @ self.negative_prototypes.t()
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (positive_similarity - negative_similarity)
        return logits, adapted, original


def protocol_class_indices(protocol, total_classes, base_classes):
    if protocol == "all26":
        return torch.arange(total_classes, dtype=torch.long)
    if protocol == "base5":
        if not 0 < base_classes < total_classes:
            raise ValueError("base5 requires 0 < base_classes < total_classes")
        return torch.arange(base_classes, dtype=torch.long)
    raise ValueError(f"Unknown prototype protocol: {protocol}")

