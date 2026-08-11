"""Source-faithful EmotionCLIP visual tower with expanding task heads."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_type = value.dtype
        value = F.layer_norm(
            value, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return value.to(original_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                (
                    ("c_fc", nn.Linear(width, hidden)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(hidden, width)),
                )
            )
        )
        self.ln_2 = LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(value)
        attended = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        value = value + attended
        return value + self.mlp(self.ln_2(value))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            ResidualAttentionBlock(width, heads, mlp_ratio) for _ in range(layers)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            value = block(value)
        return value


class EmotionCLIPVisualTransformer(nn.Module):
    """Exact EmotionCLIP subject-token ViT operator (default ViT-B/32)."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 32,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("EmotionCLIP image size must be divisible by patch size")
        self.image_size = (int(image_size), int(image_size))
        self.patch_size = (int(patch_size), int(patch_size))
        self.grid_size = image_size // patch_size
        self.output_dim = int(output_dim)
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size * self.grid_size + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.avg_pool = nn.AvgPool2d(kernel_size=self.patch_size)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("EmotionCLIP image must have shape [N, 3, H, W]")
        if mask.shape != (image.shape[0], image.shape[2], image.shape[3]):
            raise ValueError("EmotionCLIP mask must have shape [N, H, W]")
        value = self.conv1(image).reshape(image.shape[0], -1, self.grid_size ** 2)
        value = value.permute(0, 2, 1)
        cls = self.class_embedding.to(value.dtype) + torch.zeros(
            value.shape[0], 1, value.shape[-1], dtype=value.dtype, device=value.device
        )
        value = torch.cat((cls, value), dim=1)
        value = value + self.positional_embedding.to(value.dtype)
        subject = self.avg_pool(mask.unsqueeze(1).float())
        subject = subject.reshape(subject.shape[0], 1, -1).permute(0, 2, 1)
        subject = subject * self.positional_embedding[1:].to(value.dtype)
        subject = subject.sum(dim=1, keepdim=True)
        value = torch.cat((value, subject), dim=1)
        value = self.ln_pre(value).permute(1, 0, 2)
        value = self.transformer(value).permute(1, 0, 2)
        return self.ln_post(value[:, 0, :]) @ self.proj


class EmotionCLIPFTModel(nn.Module):
    def __init__(self, visual_encoder: nn.Module) -> None:
        super().__init__()
        output_dim = int(getattr(visual_encoder, "output_dim", -1))
        if output_dim <= 0:
            raise ValueError("EmotionCLIP visual encoder must expose output_dim")
        self.visual_encoder = visual_encoder
        self.feature_dim = output_dim
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
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        self.heads.append(head)
        self._head_sizes.append(int(classes))
        return head

    def restore_heads(self, sizes: Iterable[int]) -> None:
        if self.heads:
            raise RuntimeError("restore_heads requires an empty model")
        for size in sizes:
            self.add_head(int(size))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("EmotionCLIP-FT requires [N, 4, 224, 224] RGB+mask input")
        features = self.visual_encoder(inputs[:, :3], inputs[:, 3])
        return F.normalize(features.float(), dim=-1)

    def current_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No EmotionCLIP task head exists")
        return self.heads[-1](self.encode(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No EmotionCLIP task head exists")
        features = self.encode(inputs)
        return torch.cat(tuple(head(features) for head in self.heads), dim=1)

    def load_official_visual_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.visual_encoder.state_dict())
        observed = set(state)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"EmotionCLIP visual checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.visual_encoder.load_state_dict(dict(state), strict=True)
