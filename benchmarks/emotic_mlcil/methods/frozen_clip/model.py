"""Shared Track-A classifier used by continual-learning baselines."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ResidualFeatureAdapter(nn.Module):
    """Small shared residual mapping on top of frozen CLIP image features."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("feature_dim and bottleneck_dim must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(self.feature_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.bottleneck_dim, self.feature_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "Adapter features must have shape [N, feature_dim], got "
                f"{tuple(features.shape)}"
            )
        normalized = F.normalize(features.float(), dim=-1)
        residual = self.up(self.activation(self.down(normalized)))
        return normalized + self.residual_scale * residual


class GrowingMultiLabelClassifier(nn.Module):
    """One shared adapter followed by protocol-ordered task head blocks."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.adapter = ResidualFeatureAdapter(
            feature_dim=feature_dim,
            bottleneck_dim=bottleneck_dim,
            residual_scale=residual_scale,
        )
        self.heads = nn.ModuleList()
        self._head_sizes = []

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    def add_head(self, classes: int) -> nn.Linear:
        if classes <= 0:
            raise ValueError("A task head must contain at least one class")
        for head in self.heads:
            head.requires_grad_(False)
        head = nn.Linear(self.feature_dim, int(classes))
        nn.init.normal_(head.weight, std=0.01)
        nn.init.zeros_(head.bias)
        self.heads.append(head)
        self._head_sizes.append(int(classes))
        return head

    def restore_heads(self, head_sizes: Iterable[int]) -> None:
        if self.heads:
            raise RuntimeError("restore_heads requires an empty classifier")
        for classes in head_sizes:
            self.add_head(int(classes))

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No task head has been added")
        adapted = self.forward_features(features)
        return torch.cat([head(adapted) for head in self.heads], dim=1)

    def current_logits(self, features: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No task head has been added")
        adapted = self.forward_features(features)
        return self.heads[-1](adapted)
