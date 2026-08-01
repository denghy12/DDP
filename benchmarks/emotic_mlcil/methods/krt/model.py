"""KRT task-token/ICA model driven by OpenAI CLIP visual patch tokens.

The task-token block, per-task classifiers, and their initialization follow the
official KRT release.  Only the source spatial backbone is replaced: projected
CLIP ViT patch tokens take the place of the TResNet feature map.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


def _truncated_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    nn.init.trunc_normal_(tensor, std=std)


class CLIPVisualPatchEncoder(nn.Module):
    """Expose all final ViT tokens from the repository's OpenAI CLIP tower."""

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
                "KRT Track A requires an OpenAI CLIP VisionTransformer; missing "
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


class KRTClassifier(nn.Module):
    """Official KRT continual classifier: LayerNorm followed by Linear."""

    def __init__(self, embed_dim: int, classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(token))


class ClassAttention(nn.Module):
    """The ClassAttention operation shipped in the official KRT source."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("KRT embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, count, width = tokens.shape
        query = self.q(tokens[:, 0]).reshape(
            batch, 1, self.num_heads, width // self.num_heads
        ).permute(0, 2, 1, 3)
        key = self.k(tokens).reshape(
            batch, count, self.num_heads, width // self.num_heads
        ).permute(0, 2, 1, 3)
        value = self.v(tokens).reshape(
            batch, count, self.num_heads, width // self.num_heads
        ).permute(0, 2, 1, 3)
        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(
            dim=-1
        )
        output = (attention @ value).transpose(1, 2).reshape(batch, 1, width)
        return self.proj(output), attention


class ClassAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = ClassAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        update, attention = self.attention(self.norm1(tokens))
        cls_token = tokens[:, :1] + update
        cls_token = cls_token + self.mlp(self.norm2(cls_token))
        return cls_token, attention


class KRTModel(nn.Module):
    """CLIP-backed KRT model with one token and classifier per task."""

    def __init__(
        self,
        token_encoder: nn.Module,
        token_dim: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        max_patch_tokens: int = 196,
        use_positional_embedding: bool = True,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or embed_dim <= 0 or max_patch_tokens <= 0:
            raise ValueError("KRT token dimensions must be positive")
        self.token_encoder = token_encoder
        self.token_dim = int(token_dim)
        self.embed_dim = int(embed_dim)
        self.max_patch_tokens = int(max_patch_tokens)
        self.use_positional_embedding = bool(use_positional_embedding)
        self.last_proj = nn.Linear(token_dim, embed_dim)
        nn.init.kaiming_normal_(
            self.last_proj.weight,
            mode="fan_out",
            nonlinearity="leaky_relu",
        )
        nn.init.zeros_(self.last_proj.bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        _truncated_normal_(self.cls_token)
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(
                torch.zeros(1, max_patch_tokens, embed_dim)
            )
            _truncated_normal_(self.positional_embedding)
        else:
            self.register_parameter("positional_embedding", None)
        self.tab = ClassAttentionBlock(embed_dim, num_heads, mlp_ratio)
        self.task_tokens = nn.ParameterList()
        self.heads = nn.ModuleList()
        self._head_sizes: List[int] = []

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    @property
    def num_tasks(self) -> int:
        return len(self._head_sizes)

    def add_task(self, classes: int) -> None:
        if classes <= 0:
            raise ValueError("A KRT task must introduce at least one class")
        if self.task_tokens:
            task_token = copy.deepcopy(self.task_tokens[-1])
            _truncated_normal_(task_token)
        else:
            task_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            _truncated_normal_(task_token)
        self.task_tokens.append(task_token)
        self.heads.append(KRTClassifier(self.embed_dim, int(classes)))
        self._head_sizes.append(int(classes))

    def restore_tasks(self, head_sizes: Iterable[int]) -> None:
        if self.num_tasks:
            raise RuntimeError("restore_tasks requires an unexpanded KRT model")
        for classes in head_sizes:
            self.add_task(int(classes))

    def freeze_old_task_parameters(self) -> None:
        """Match KRT's ``freeze_task: [old_task_tokens, old_heads]``."""

        self.requires_grad_(True)
        for token in self.task_tokens[:-1]:
            token.requires_grad_(False)
        for head in self.heads[:-1]:
            head.requires_grad_(False)

    def encode_patch_tokens(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.token_encoder(images)
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("KRT token encoder must return a tensor")
        if encoded.ndim == 2:
            if encoded.shape[1] != self.token_dim:
                raise ValueError("KRT token encoder feature width differs")
            pooled = encoded
            patches = encoded.unsqueeze(1)
        elif encoded.ndim == 3:
            if encoded.shape[2] != self.token_dim:
                raise ValueError("KRT token encoder token width differs")
            if encoded.shape[1] < 2:
                pooled = encoded[:, 0]
                patches = encoded
            else:
                pooled = encoded[:, 0]
                patches = encoded[:, 1:]
        else:
            raise ValueError("KRT token encoder must return [N,D] or [N,T,D]")
        if patches.shape[1] > self.max_patch_tokens:
            raise ValueError("KRT received more patch tokens than configured")
        patches = self.last_proj(patches.float())
        if self.positional_embedding is not None:
            patches = patches + self.positional_embedding[:, : patches.shape[1]]
        return patches, pooled.float()

    def forward(self, images: torch.Tensor) -> Dict[str, object]:
        if not self.num_tasks:
            raise RuntimeError("No KRT task has been added")
        patches, pooled = self.encode_patch_tokens(images)
        batch = images.shape[0]
        tokens: List[torch.Tensor] = []
        attentions: List[torch.Tensor] = []
        for task_token in self.task_tokens:
            inputs = torch.cat(
                [
                    self.cls_token.expand(batch, -1, -1),
                    task_token.expand(batch, -1, -1),
                    patches,
                ],
                dim=1,
            )
            output, attention = self.tab(inputs)
            tokens.append(output[:, 0])
            attentions.append(attention)
        logits = torch.cat(
            [head(tokens[index]) for index, head in enumerate(self.heads)],
            dim=1,
        )
        return {
            "logits": logits,
            "tokens": tokens,
            "attentions": attentions,
            "pool_embeddings": pooled,
        }

    def current_logits(self, images: torch.Tensor) -> torch.Tensor:
        output = self(images)
        return self.heads[-1](output["tokens"][-1])
