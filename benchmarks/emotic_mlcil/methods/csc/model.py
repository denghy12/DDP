"""Independent Track-A implementation of CSC's dynamically expanding CI-GCN.

Only the source spatial backbone is substituted: final OpenAI CLIP ViT patch
tokens replace the TResNet feature map. The class-activation nodes, general and
sample-specific graph relations, graph classifier, and two-branch averaging
follow the audited CSC paper/release contract.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class CLIPVisualPatchEncoder(nn.Module):
    """Return every final ViT token from an OpenAI CLIP visual tower."""

    def __init__(self, visual_encoder: nn.Module) -> None:
        super().__init__()
        required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
            "proj",
        )
        missing = [name for name in required if not hasattr(visual_encoder, name)]
        if missing:
            raise TypeError(
                "CSC Track A requires an OpenAI CLIP VisionTransformer; missing "
                + ", ".join(missing)
            )
        self.visual_encoder = visual_encoder
        self.output_dim = int(getattr(visual_encoder, "output_dim", -1))
        if self.output_dim <= 0:
            raise ValueError("CLIP visual output_dim must be positive")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.visual_encoder
        x = images.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1)
        x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        return x


class CSCModel(nn.Module):
    """CLIP-backed CSC model with seen-class dynamic graph expansion."""

    def __init__(
        self,
        token_encoder: nn.Module,
        token_dim: int,
        graph_dim: int,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or graph_dim <= 0:
            raise ValueError("CSC token and graph dimensions must be positive")
        self.token_encoder = token_encoder
        self.token_dim = int(token_dim)
        self.graph_dim = int(graph_dim)

        # Source conv_transform: token-wise here because CLIP tokens replace the
        # TResNet spatial grid.
        self.feature_projection = nn.Linear(token_dim, graph_dim)
        self.general_weight = nn.Conv1d(graph_dim, graph_dim, 1)
        self.global_projection = nn.Conv1d(graph_dim, graph_dim, 1)
        self.global_norm = nn.BatchNorm1d(graph_dim)
        self.specific_weight = nn.Conv1d(graph_dim, graph_dim, 1)
        self.activation = nn.LeakyReLU(0.2)

        # These layers grow with the seen-label axis. They are absent until the
        # first protocol task is introduced.
        self.class_activation: Optional[nn.Linear] = None
        self.general_relation: Optional[nn.Conv1d] = None
        self.specific_relation: Optional[nn.Conv1d] = None
        self.graph_classifier: Optional[nn.Conv1d] = None
        self.register_parameter("identity_mask", None)
        self._task_sizes: List[int] = []

    @property
    def task_sizes(self) -> Tuple[int, ...]:
        return tuple(self._task_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._task_sizes)

    @property
    def num_tasks(self) -> int:
        return len(self._task_sizes)

    def _new_layers(
        self, classes: int
    ) -> Tuple[nn.Linear, nn.Conv1d, nn.Conv1d, nn.Conv1d, nn.Parameter]:
        device = self.feature_projection.weight.device
        dtype = self.feature_projection.weight.dtype
        class_activation = nn.Linear(
            self.token_dim, classes, bias=False
        ).to(device=device, dtype=dtype)
        general_relation = nn.Conv1d(
            classes, classes, 1, bias=False
        ).to(device=device, dtype=dtype)
        specific_relation = nn.Conv1d(
            self.graph_dim * 2, classes, 1
        ).to(device=device, dtype=dtype)
        graph_classifier = nn.Conv1d(
            self.graph_dim, classes, 1
        ).to(device=device, dtype=dtype)
        identity_mask = nn.Parameter(
            torch.eye(classes, device=device, dtype=dtype),
            requires_grad=False,
        )
        return (
            class_activation,
            general_relation,
            specific_relation,
            graph_classifier,
            identity_mask,
        )

    def add_task(self, classes: int) -> None:
        if classes <= 0:
            raise ValueError("A CSC task must introduce at least one class")
        old_classes = self.num_classes
        new_classes = old_classes + int(classes)
        (
            class_activation,
            general_relation,
            specific_relation,
            graph_classifier,
            identity_mask,
        ) = self._new_layers(new_classes)

        if old_classes:
            if (
                self.class_activation is None
                or self.general_relation is None
                or self.specific_relation is None
                or self.graph_classifier is None
                or self.identity_mask is None
            ):
                raise RuntimeError("CSC dynamic layers are incomplete")
            with torch.no_grad():
                class_activation.weight[:old_classes].copy_(
                    self.class_activation.weight
                )
                general_relation.weight[:old_classes, :old_classes].copy_(
                    self.general_relation.weight
                )
                specific_relation.weight[:old_classes].copy_(
                    self.specific_relation.weight
                )
                specific_relation.bias[:old_classes].copy_(
                    self.specific_relation.bias
                )
                graph_classifier.weight[:old_classes].copy_(
                    self.graph_classifier.weight
                )
                graph_classifier.bias[:old_classes].copy_(
                    self.graph_classifier.bias
                )
                identity_mask[:old_classes, :old_classes].copy_(
                    self.identity_mask
                )

        self.class_activation = class_activation
        self.general_relation = general_relation
        self.specific_relation = specific_relation
        self.graph_classifier = graph_classifier
        self.identity_mask = identity_mask
        self._task_sizes.append(int(classes))

    def restore_tasks(self, task_sizes: Iterable[int]) -> None:
        if self.num_tasks:
            raise RuntimeError("restore_tasks requires an unexpanded CSC model")
        for classes in task_sizes:
            self.add_task(int(classes))

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.token_encoder(images)
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("CSC token encoder must return a tensor")
        if encoded.ndim == 2:
            patches = encoded.unsqueeze(1)
        elif encoded.ndim == 3:
            # OpenAI CLIP returns its CLS token first. A one-token injected
            # encoder is accepted for deterministic unit tests.
            patches = encoded[:, 1:] if encoded.shape[1] > 1 else encoded
        else:
            raise ValueError("CSC token encoder must return [N,D] or [N,T,D]")
        if patches.shape[-1] != self.token_dim:
            raise ValueError("CSC token encoder feature width differs")
        if patches.shape[1] == 0:
            raise ValueError("CSC requires at least one visual patch token")
        return patches.float()

    def _checked_dynamic_layers(
        self,
    ) -> Tuple[nn.Linear, nn.Conv1d, nn.Conv1d, nn.Conv1d]:
        if (
            self.class_activation is None
            or self.general_relation is None
            or self.specific_relation is None
            or self.graph_classifier is None
            or self.identity_mask is None
        ):
            raise RuntimeError("No CSC task has been added")
        return (
            self.class_activation,
            self.general_relation,
            self.specific_relation,
            self.graph_classifier,
        )

    def _normalized_global(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.global_projection(value)
        # The official loader drops the last batch, avoiding a single value per
        # BatchNorm channel. The Core keeps every sample; for the singleton case
        # use the accumulated running statistics instead of leaking/dropping it.
        if self.training and projected.shape[0] * projected.shape[2] == 1:
            return F.batch_norm(
                projected,
                self.global_norm.running_mean,
                self.global_norm.running_var,
                self.global_norm.weight,
                self.global_norm.bias,
                training=False,
                momentum=0.0,
                eps=self.global_norm.eps,
            )
        return self.global_norm(projected)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        (
            class_activation,
            general_relation,
            specific_relation,
            graph_classifier,
        ) = self._checked_dynamic_layers()
        patches = self._patch_tokens(images)

        # Source CAM classifier and spatial max pooling.
        activation_maps = class_activation(patches)  # [B, patches, classes]
        classification_logits = activation_maps.max(dim=1).values
        masks = torch.softmax(activation_maps, dim=1)

        projected = self.feature_projection(patches)  # [B, patches, graph]
        label_nodes = torch.einsum("btg,btc->bgc", projected, masks)

        general = general_relation(label_nodes.transpose(1, 2))
        general = self.activation(general).transpose(1, 2)
        general = self.activation(self.general_weight(general))
        general_nodes = label_nodes + general

        global_node = F.adaptive_avg_pool1d(general_nodes, 1)
        global_node = self.activation(self._normalized_global(global_node))
        global_node = global_node.expand(-1, -1, self.num_classes)
        relation_input = torch.cat([global_node, general_nodes], dim=1)
        sample_relation = torch.sigmoid(specific_relation(relation_input))

        specific = torch.matmul(general_nodes, sample_relation)
        specific = self.activation(specific)
        specific = self.activation(self.specific_weight(specific))
        graph_nodes = label_nodes + specific

        graph_matrix = graph_classifier(graph_nodes)
        graph_logits = torch.diagonal(graph_matrix, dim1=1, dim2=2)
        logits = (classification_logits + graph_logits) / 2.0
        return {
            "logits": logits,
            "classification_logits": classification_logits,
            "graph_logits": graph_logits,
            "activation_maps": activation_maps,
            "label_nodes": label_nodes,
            "general_nodes": general_nodes,
            "graph_nodes": graph_nodes,
            "sample_relation": sample_relation,
        }
