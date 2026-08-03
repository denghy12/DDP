"""Model lifecycle for the independent Track-A L3A implementation."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn


class L3AModel(nn.Module):
    """CLIP visual tower followed by L3A's two-stage classifier lifecycle.

    Task 0 first uses ``base_classifier`` for ordinary gradient training.  The
    temporary head is then discarded and replaced by the released random
    expansion, ReLU, and bias-free analytic classifier.  Later tasks only grow
    the analytic output width.
    """

    def __init__(
        self,
        visual_encoder: nn.Module,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("L3A feature dimensions must be positive")
        self.visual_encoder = visual_encoder
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.base_classifier: Optional[nn.Linear] = None
        self.random_projection: Optional[nn.Linear] = None
        self.analytic_classifier: Optional[nn.Linear] = None
        self._task_sizes = []

    @property
    def task_sizes(self) -> Tuple[int, ...]:
        return tuple(self._task_sizes)

    @property
    def num_classes(self) -> int:
        if self.analytic_classifier is not None:
            return int(self.analytic_classifier.out_features)
        if self.base_classifier is not None:
            return int(self.base_classifier.out_features)
        return 0

    @property
    def analytic_ready(self) -> bool:
        return (
            self.random_projection is not None
            and self.analytic_classifier is not None
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual_encoder(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("L3A visual encoder must return [N, D] features")
        if features.shape[1] != self.feature_dim:
            raise ValueError("L3A visual feature width differs from configuration")
        return features

    def add_base_task(self, classes: int) -> None:
        if classes <= 0:
            raise ValueError("L3A base task must introduce classes")
        if self.num_classes or self._task_sizes:
            raise RuntimeError("L3A base task has already been initialized")
        self.base_classifier = nn.Linear(self.feature_dim, int(classes))
        self._task_sizes.append(int(classes))

    def base_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.base_classifier is None or self.analytic_ready:
            raise RuntimeError("L3A temporary base classifier is not active")
        return self.base_classifier(self.encode_images(images))

    def initialize_analytic(self) -> None:
        if self.base_classifier is None or len(self._task_sizes) != 1:
            raise RuntimeError("L3A analytic alignment requires the base task")
        classes = self.base_classifier.out_features
        device = self.base_classifier.weight.device
        dtype = self.base_classifier.weight.dtype
        self.random_projection = nn.Linear(
            self.feature_dim,
            self.hidden_dim,
            bias=False,
        ).to(device=device, dtype=dtype)
        self.analytic_classifier = nn.Linear(
            self.hidden_dim,
            classes,
            bias=False,
        ).to(device=device, dtype=dtype)
        # The released code replaces the gradient-trained head entirely.
        self.base_classifier = None
        self.visual_encoder.requires_grad_(False)
        self.random_projection.requires_grad_(False)
        self.analytic_classifier.requires_grad_(False)

    def expand_analytic(self, classes: int) -> None:
        if classes <= 0:
            raise ValueError("L3A incremental task must introduce classes")
        if not self.analytic_ready or self.analytic_classifier is None:
            raise RuntimeError("L3A analytic classifier is not initialized")
        old = self.analytic_classifier
        expanded = nn.Linear(
            self.hidden_dim,
            old.out_features + int(classes),
            bias=False,
        ).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[: old.out_features].copy_(old.weight)
        expanded.requires_grad_(False)
        self.analytic_classifier = expanded
        self._task_sizes.append(int(classes))

    def restore_analytic(self, task_sizes: Iterable[int]) -> None:
        sizes = tuple(int(value) for value in task_sizes)
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError("L3A checkpoint task sizes are invalid")
        if self.num_classes or self._task_sizes:
            raise RuntimeError("Restore L3A checkpoint into a fresh model")
        self.add_base_task(sizes[0])
        self.initialize_analytic()
        for classes in sizes[1:]:
            self.expand_analytic(classes)

    def analytic_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.random_projection is None:
            raise RuntimeError("L3A random expansion is not initialized")
        return torch.relu(self.random_projection(self.encode_images(images)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.analytic_classifier is None:
            return self.base_logits(images)
        return self.analytic_classifier(self.analytic_features(images))

    def set_analytic_solution(self, solution: torch.Tensor) -> None:
        """Install W shaped [hidden_dim, seen_classes]."""

        if self.analytic_classifier is None:
            raise RuntimeError("L3A analytic classifier is not initialized")
        expected = (self.hidden_dim, self.num_classes)
        if tuple(solution.shape) != expected:
            raise ValueError(
                f"L3A analytic solution shape {tuple(solution.shape)} != {expected}"
            )
        with torch.no_grad():
            self.analytic_classifier.weight.copy_(
                solution.transpose(0, 1).to(
                    device=self.analytic_classifier.weight.device,
                    dtype=self.analytic_classifier.weight.dtype,
                )
            )
