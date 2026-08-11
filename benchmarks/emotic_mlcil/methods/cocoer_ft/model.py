"""Protocol-safe native CocoER model used by the sequential-FT conversion."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class NativeResNet50GridEncoder(nn.Module):
    """Released ImageNet ResNet-50 tower through layer4, projected to 256-D."""

    def __init__(self, state_dict: Optional[dict] = None, width: int = 256) -> None:
        super().__init__()
        from torchvision.models import resnet50

        source = resnet50(weights=None)
        if state_dict is not None:
            source.load_state_dict(state_dict, strict=True)
        self.features = nn.Sequential(
            source.conv1,
            source.bn1,
            source.relu,
            source.maxpool,
            source.layer1,
            source.layer2,
            source.layer3,
            source.layer4,
        )
        self.projection = nn.Conv2d(2048, width, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.projection(self.features(images))
        if features.shape[-2:] != (7, 7):
            raise ValueError("CocoER ResNet-50 towers must produce a 7x7 grid")
        return features.flatten(2).transpose(1, 2)


class CrossLevelEncoder(nn.Module):
    """Pre-norm cross-level attention block matching the released GWT width."""

    def __init__(self, width: int = 256, heads: int = 4, expansion: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.1, batch_first=True)
        self.attention_dropout = nn.Dropout(0.1)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(width * expansion, width),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_position: torch.Tensor,
        memory_position: torch.Tensor,
    ) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            normalized_query + query_position,
            normalized_memory + memory_position,
            memory,
            need_weights=False,
        )
        query = query + self.attention_dropout(attended)
        return query + self.ffn(self.ffn_norm(query))


class ProgressiveFeatureHead(nn.Module):
    """Released 49x256→256 HFE projection with protocol-ordered task heads."""

    def __init__(self, width: int = 256, grid: int = 7) -> None:
        super().__init__()
        self.width = int(width)
        self.feature = nn.Sequential(
            nn.GELU(),
            nn.Linear(grid * grid * width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.heads = nn.ModuleList()
        self._head_sizes = []

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    def add_head(self, classes: int) -> None:
        head = nn.Linear(self.width, int(classes))
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        self.heads.append(head)
        self._head_sizes.append(int(classes))

    def encode(self, grid: torch.Tensor) -> torch.Tensor:
        return self.feature(grid.flatten(1))

    def logits(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("CocoER task heads have not been initialized")
        return torch.cat([head(feature) for head in self.heads], dim=1)


class ProgressiveLinearHeads(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = nn.ModuleList()
        self._head_sizes = []

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    def add_head(self, classes: int) -> None:
        head = nn.Linear(self.width, int(classes))
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        self.heads.append(head)
        self._head_sizes.append(int(classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("CocoER task heads have not been initialized")
        return torch.cat([head(features) for head in self.heads], dim=1)


def dynamic_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Released dynamic class-weight BCE, generalized to visible columns only."""

    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("CocoER logits and targets must be aligned matrices")
    positives = targets.sum(dim=0)
    weights = torch.full_like(positives, 1.0e-4)
    present = positives != 0
    weights[present] = 1.0 / torch.log(positives[present] + 1.2)
    return F.binary_cross_entropy_with_logits(
        logits, targets, weight=weights.unsqueeze(0).expand_as(logits), reduction="none"
    ).sum(dim=-1).mean()


def _normalized_gradient(value: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(value, feature, retain_graph=True, create_graph=False)[0]
    return F.normalize(gradient, dim=1, eps=1.0e-12)


class CocoERFTModel(nn.Module):
    """Native CocoER visual stack with a protocol-safe expanding label interface.

    The release's all-26-class GWT/VI checkpoints are intentionally not loaded.
    Generic ImageNet and OpenAI CLIP RN50 initialization is supplied separately.
    """

    grid_size = 7

    def __init__(
        self,
        total_classes: int,
        head_encoder: nn.Module,
        body_encoder: nn.Module,
        context_encoder: nn.Module,
        clip_image_encoder: nn.Module,
        *,
        width: int = 256,
        clip_width: int = 1024,
        vi_hidden_width: Optional[int] = None,
        encoder_blocks: int = 3,
        inside_lr: float = 0.1,
        pseudo_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        if total_classes <= 0 or encoder_blocks <= 0:
            raise ValueError("CocoER dimensions must be positive")
        self.total_classes = int(total_classes)
        self.width = int(width)
        self.clip_width = int(clip_width)
        self.inside_lr = float(inside_lr)
        self.pseudo_threshold = float(pseudo_threshold)
        self.head_encoder = head_encoder
        self.body_encoder = body_encoder
        self.context_encoder = context_encoder
        self.clip_image_encoder = clip_image_encoder
        self.clip_image_encoder.requires_grad_(False)

        self.context_position = nn.Parameter(torch.empty(1, 49, width))
        nn.init.uniform_(self.context_position)
        self.head_blocks = nn.ModuleList([CrossLevelEncoder(width) for _ in range(encoder_blocks)])
        self.body_blocks = nn.ModuleList([CrossLevelEncoder(width) for _ in range(encoder_blocks)])
        self.context_blocks = nn.ModuleList([CrossLevelEncoder(width) for _ in range(encoder_blocks)])
        self.output_norm = nn.LayerNorm(width)

        self.head_branch = ProgressiveFeatureHead(width)
        self.body_branch = ProgressiveFeatureHead(width)
        self.context_branch = ProgressiveFeatureHead(width)

        self.mapping = nn.Linear(clip_width, clip_width, bias=False)
        nn.init.eye_(self.mapping.weight)
        repeated = clip_width * total_classes
        hidden = int(vi_hidden_width or repeated // 4)
        self.vi_feature = nn.Sequential(
            nn.GELU(), nn.Linear(repeated, hidden), nn.GELU(),
            nn.Linear(hidden, width), nn.GELU(),
        )
        self.vi_heads = ProgressiveLinearHeads(width)
        self.global_feature = nn.Sequential(
            nn.GELU(), nn.LayerNorm(4 * width), nn.Linear(4 * width, width),
            nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width),
            nn.GELU(), nn.LayerNorm(width),
        )
        self.global_heads = ProgressiveLinearHeads(width)
        self._head_sizes = []

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    def add_head(self, classes: int) -> None:
        if classes <= 0:
            raise ValueError("A CocoER task head must contain classes")
        for branch in (
            self.head_branch, self.body_branch, self.context_branch,
            self.vi_heads, self.global_heads,
        ):
            branch.add_head(classes)
        self._head_sizes.append(int(classes))

    def restore_heads(self, sizes: Iterable[int]) -> None:
        if self._head_sizes:
            raise RuntimeError("restore_heads requires an empty CocoER model")
        for size in sizes:
            self.add_head(int(size))

    @staticmethod
    def split_inputs(images: torch.Tensor, geometry: torch.Tensor):
        if images.ndim != 5 or images.shape[1:] != (3, 3, 224, 224):
            raise ValueError("CocoER requires [N,3,3,224,224] context/body/head images")
        if geometry is None or geometry.shape != (images.shape[0], 2, 4):
            raise ValueError("CocoER requires [N,2,4] body/head geometry")
        return images[:, 0], images[:, 1], images[:, 2], geometry[:, 0], geometry[:, 1]

    def _roi_position(self, boxes: torch.Tensor) -> torch.Tensor:
        # Source coordinates are scaled to the 224 canvas; sample the learned
        # 7x7 context embedding at the corresponding dense ROI grid.
        batch = boxes.shape[0]
        start = boxes[:, :2]
        end = boxes[:, 2:]
        axis = (torch.arange(7, device=boxes.device, dtype=boxes.dtype) + 0.5) / 7.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        unit = torch.stack((xx.flatten(), yy.flatten()), dim=1).unsqueeze(0)
        points = start.unsqueeze(1) + unit * (end - start).unsqueeze(1)
        grid = points / 112.0 - 1.0
        source = self.context_position.expand(batch, -1, -1).transpose(1, 2).reshape(
            batch, self.width, 7, 7
        )
        sampled = F.grid_sample(
            source, grid.view(batch, 7, 7, 2), mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        return sampled.flatten(2).transpose(1, 2)

    def _encode_clip(self, context: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.clip_image_encoder(context)
        if features.shape != (context.shape[0], self.clip_width):
            raise ValueError("CocoER CLIP RN50 encoder returned an unexpected shape")
        features = F.normalize(features.float(), dim=-1)
        mapped = self.mapping(features).unsqueeze(1).expand(-1, self.total_classes, -1)
        return self.vi_feature(mapped.flatten(1))

    def _encode_branches(
        self, context: torch.Tensor, body: torch.Tensor, head: torch.Tensor,
        body_boxes: torch.Tensor, head_boxes: torch.Tensor,
    ):
        head_grid = self.head_encoder(head)
        body_grid = self.body_encoder(body)
        context_grid = self.context_encoder(context)
        expected = (context.shape[0], 49, self.width)
        if any(value.shape != expected for value in (head_grid, body_grid, context_grid)):
            raise ValueError("CocoER native towers must return [N,49,256] grids")
        context_position = self.context_position.expand(context.shape[0], -1, -1)
        head_position = self._roi_position(head_boxes)
        body_position = self._roi_position(body_boxes)
        for block in self.head_blocks:
            head_grid = block(head_grid, context_grid, head_position, context_position)
        for block in self.body_blocks:
            body_grid = block(body_grid, context_grid, body_position, context_position)
        for block in self.context_blocks:
            context_grid = block(context_grid, context_grid, context_position, context_position)
        return tuple(
            branch.encode(self.output_norm(grid))
            for branch, grid in (
                (self.head_branch, head_grid),
                (self.body_branch, body_grid),
                (self.context_branch, context_grid),
            )
        )

    def forward(self, images: torch.Tensor, geometry: torch.Tensor) -> Dict[str, torch.Tensor]:
        context, body, head, body_boxes, head_boxes = self.split_inputs(images, geometry)
        head_feature, body_feature, context_feature = self._encode_branches(
            context, body, head, body_boxes, head_boxes
        )
        vi_feature = self._encode_clip(context)
        head_logits = self.head_branch.logits(head_feature)
        body_logits = self.body_branch.logits(body_feature)
        context_logits = self.context_branch.logits(context_feature)
        vi_logits = self.vi_heads(vi_feature)

        pseudo = (torch.sigmoid(vi_logits.detach()) > self.pseudo_threshold).float()
        branch_features = (head_feature, body_feature, context_feature)
        branch_modules = (self.head_branch, self.body_branch, self.context_branch)
        if all(feature.requires_grad for feature in branch_features):
            branch_losses = tuple(
                dynamic_bce(logits, pseudo)
                for logits in (head_logits, body_logits, context_logits)
            )
            gradients = tuple(
                _normalized_gradient(loss, feature)
                for loss, feature in zip(branch_losses, branch_features)
            )
        else:
            # Benchmark prediction runs under no_grad. Recreate only the tiny
            # feature-level competition graph; the visual towers stay frozen.
            with torch.enable_grad():
                proxy_features = tuple(
                    feature.detach().requires_grad_(True) for feature in branch_features
                )
                proxy_logits = tuple(
                    branch.logits(feature)
                    for branch, feature in zip(branch_modules, proxy_features)
                )
                proxy_losses = tuple(
                    dynamic_bce(logits, pseudo) for logits in proxy_logits
                )
                gradients = tuple(
                    _normalized_gradient(loss, feature)
                    for loss, feature in zip(proxy_losses, proxy_features)
                )
                branch_losses = tuple(loss.detach() for loss in proxy_losses)
        common = torch.stack(gradients).mean(dim=0).detach()
        refined = tuple(
            feature - self.inside_lr * common
            for feature in (head_feature, body_feature, context_feature)
        )
        fused = self.global_feature(torch.cat((*refined, vi_feature), dim=1))
        return {
            "logits": self.global_heads(fused),
            "head_logits": head_logits,
            "body_logits": body_logits,
            "context_logits": context_logits,
            "vi_logits": vi_logits,
            "competition_distance": common.norm(dim=1).mean(),
            "pseudo_positive_rate": pseudo.mean(),
            "pseudo_head_loss": branch_losses[0].detach(),
            "pseudo_body_loss": branch_losses[1].detach(),
            "pseudo_context_loss": branch_losses[2].detach(),
        }
