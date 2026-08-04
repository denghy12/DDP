"""Independent AGCN operators for the EMOTIC Track-A adaptation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def graph_normalize(adjacency: torch.Tensor, exponent: float) -> torch.Tensor:
    """Match the released ``gen_P`` row-degree transformation."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("AGCN adjacency must be square")
    degrees = adjacency.sum(dim=1).float()
    if torch.any(degrees <= 0):
        raise ValueError("AGCN adjacency must have positive row degree")
    degree_matrix = torch.diag(torch.pow(degrees, exponent))
    return (adjacency.float() @ degree_matrix).t() @ degree_matrix


def threshold_scale(
    correlation: torch.Tensor,
    *,
    threshold: float,
    scale: float,
    add_identity: bool,
) -> torch.Tensor:
    """Threshold, column-scale, and optionally self-connect a graph block."""

    if correlation.ndim != 2:
        raise ValueError("AGCN correlation block must be a matrix")
    binary = (correlation >= threshold).to(dtype=torch.float32)
    scaled = binary * scale / (binary.sum(dim=0, keepdim=True) + 1.0e-6)
    if add_identity:
        if scaled.shape[0] != scaled.shape[1]:
            raise ValueError("Only square AGCN blocks can receive identity edges")
        scaled = scaled + torch.eye(
            scaled.shape[0], device=scaled.device, dtype=scaled.dtype
        )
    return scaled


class GraphConvolution(nn.Module):
    """Bias-free graph convolution used by the released two-layer AGCN."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("AGCN graph dimensions must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if node_features.ndim != 2:
            raise ValueError("AGCN node features must be [classes, features]")
        if adjacency.shape != (node_features.shape[0], node_features.shape[0]):
            raise ValueError("AGCN adjacency and node counts differ")
        return adjacency @ (node_features @ self.weight)


class AGCNGraphNetwork(nn.Module):
    """Two graph-convolution layers with the released LeakyReLU."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self, adjacency: torch.Tensor, node_features: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.activation(self.gc1(node_features, adjacency))
        return self.gc2(hidden, adjacency)


class AGCNModel(nn.Module):
    """Trainable CLIP visual tower plus AGCN-generated class weights."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        *,
        visual_dim: int,
        embedding_dim: int,
        graph_hidden_dim: int,
    ) -> None:
        super().__init__()
        if visual_dim <= 0:
            raise ValueError("AGCN visual feature width must be positive")
        self.visual_encoder = visual_encoder
        self.visual_dim = int(visual_dim)
        self.graph = AGCNGraphNetwork(
            embedding_dim, graph_hidden_dim, visual_dim
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual_encoder(images)
        if features.ndim != 2 or features.shape[1] != self.visual_dim:
            raise ValueError("AGCN visual encoder must return [N, visual_dim]")
        return features.float()

    def graph_nodes(
        self, adjacency: torch.Tensor, node_features: torch.Tensor
    ) -> torch.Tensor:
        nodes = self.graph(adjacency.float(), node_features.float())
        if nodes.shape != (node_features.shape[0], self.visual_dim):
            raise RuntimeError("AGCN graph output shape differs from class layout")
        return nodes.float()

    def forward(
        self,
        images: torch.Tensor,
        adjacency: torch.Tensor,
        node_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual_features = self.encode_images(images)
        graph_nodes = self.graph_nodes(adjacency, node_features)
        logits = visual_features @ graph_nodes.t()
        return {
            "logits": logits,
            "visual_features": visual_features,
            "graph_nodes": graph_nodes,
        }


@dataclass
class CorrelationStatistics:
    """Online sufficient statistics for one AGCN task."""

    old_classes: int
    current_classes: int
    hard_cooccurrence: torch.Tensor
    hard_positive_count: torch.Tensor
    cross_hard_positive_count: torch.Tensor
    old_new_soft_hard: torch.Tensor
    old_soft_sum: torch.Tensor
    sample_count: torch.Tensor

    @classmethod
    def create(
        cls,
        old_classes: int,
        current_classes: int,
        *,
        device: torch.device,
    ) -> "CorrelationStatistics":
        if old_classes < 0 or current_classes <= 0:
            raise ValueError("Invalid AGCN class counts")
        hard_epsilon = 1.0e-6
        cross_epsilon = 1.0e-10
        return cls(
            old_classes=int(old_classes),
            current_classes=int(current_classes),
            hard_cooccurrence=torch.zeros(
                current_classes, current_classes, device=device
            ),
            hard_positive_count=torch.full(
                (current_classes,), hard_epsilon, device=device
            ),
            cross_hard_positive_count=torch.full(
                (current_classes,), cross_epsilon, device=device
            ),
            old_new_soft_hard=torch.zeros(
                old_classes, current_classes, device=device
            ),
            old_soft_sum=torch.full((old_classes,), cross_epsilon, device=device),
            sample_count=torch.tensor(cross_epsilon, device=device),
        )

    @torch.no_grad()
    def update(
        self,
        current_targets: torch.Tensor,
        old_acm_soft_labels: Optional[torch.Tensor],
    ) -> None:
        targets = current_targets.detach().float()
        if targets.ndim != 2 or targets.shape[1] != self.current_classes:
            raise ValueError("AGCN current targets differ from correlation state")
        positives = (targets > 0.5).float()
        self.hard_positive_count += positives.sum(dim=0)
        self.cross_hard_positive_count += positives.sum(dim=0)
        cooccurrence = positives.t() @ positives
        cooccurrence.fill_diagonal_(0.0)
        self.hard_cooccurrence += cooccurrence
        if self.old_classes == 0:
            if old_acm_soft_labels is not None:
                raise ValueError("Task 0 cannot receive old AGCN soft labels")
            return
        if old_acm_soft_labels is None:
            raise ValueError("Incremental AGCN correlation requires old soft labels")
        soft = old_acm_soft_labels.detach().float()
        if soft.shape != (targets.shape[0], self.old_classes):
            raise ValueError("AGCN old soft-label shape differs from state")
        self.old_new_soft_hard += soft.t() @ positives
        self.old_soft_sum += soft.sum(dim=0)
        self.sample_count += float(targets.shape[0])

    def build_adjacency(
        self,
        previous_adjacency: Optional[torch.Tensor],
        *,
        task0_threshold: float,
        current_threshold: float,
        cross_threshold: float,
        task0_scale: float,
        later_scale: float,
        reverse_bayes_scale: float,
        task0_degree_exponent: float,
        later_degree_exponent: float,
    ) -> torch.Tensor:
        current_correlation = (
            self.hard_cooccurrence / self.hard_positive_count.unsqueeze(0)
        )
        if self.old_classes == 0:
            current_block = threshold_scale(
                current_correlation,
                threshold=task0_threshold,
                scale=task0_scale,
                add_identity=True,
            )
            return graph_normalize(current_block, task0_degree_exponent)
        if previous_adjacency is None or previous_adjacency.shape != (
            self.old_classes,
            self.old_classes,
        ):
            raise ValueError("Incremental AGCN requires the previous adjacency")
        old_new = self.old_new_soft_hard / self.cross_hard_positive_count.unsqueeze(0)
        new_prior = self.cross_hard_positive_count / self.sample_count
        old_prior = self.old_soft_sum / self.sample_count
        new_old = (
            old_new.t()
            * new_prior.unsqueeze(1)
            / old_prior.clamp_min(1.0e-12).unsqueeze(0)
            * reverse_bayes_scale
        )
        old_new = threshold_scale(
            old_new,
            threshold=cross_threshold,
            scale=later_scale,
            add_identity=False,
        )
        new_old = threshold_scale(
            new_old,
            threshold=cross_threshold,
            scale=later_scale,
            add_identity=False,
        )
        current_block = threshold_scale(
            current_correlation,
            threshold=current_threshold,
            scale=later_scale,
            add_identity=True,
        )
        current_block = graph_normalize(
            current_block, later_degree_exponent
        )
        seen = self.old_classes + self.current_classes
        adjacency = torch.zeros(
            seen,
            seen,
            device=previous_adjacency.device,
            dtype=torch.float32,
        )
        adjacency[: self.old_classes, : self.old_classes] = previous_adjacency
        adjacency[: self.old_classes, self.old_classes :] = old_new
        adjacency[self.old_classes :, : self.old_classes] = new_old
        adjacency[self.old_classes :, self.old_classes :] = current_block
        return adjacency
