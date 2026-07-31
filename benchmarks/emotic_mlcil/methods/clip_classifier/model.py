"""Shared CLIP-visual classifier without benchmark-added adapters."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class GrowingMultiLabelClassifier(nn.Module):
    """A trainable visual encoder followed by protocol-ordered linear heads."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        feature_dim: int,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.visual_encoder = visual_encoder
        self.feature_dim = int(feature_dim)
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

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.visual_encoder, "encode_image"):
            features = self.visual_encoder.encode_image(images)
        else:
            features = self.visual_encoder(images)
        if not isinstance(features, torch.Tensor):
            raise TypeError("Visual encoder output must be a tensor")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "Visual encoder must return [N, feature_dim], got "
                f"{tuple(features.shape)}"
            )
        return F.normalize(features.float(), dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No task head has been added")
        features = self.encode_images(images)
        return torch.cat([head(features) for head in self.heads], dim=1)

    def current_logits(self, images: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No task head has been added")
        features = self.encode_images(images)
        return self.heads[-1](features)
