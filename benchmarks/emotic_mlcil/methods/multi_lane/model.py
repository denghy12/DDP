"""Independent Track-A MULTI-LANE model over frozen OpenAI CLIP blocks.

The implementation follows the fixed official release at ``5ee982c`` without
copying its source.  OpenAI CLIP replaces the released ImageNet ViT, while the
selector aggregation, task K/V prompts, drop-and-replace pathway, task-slice
copying, shared classifier, and concat inference remain method-specific.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MultiLaneModel(nn.Module):
    """Frozen CLIP visual tower with preallocated MULTI-LANE task pathways."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        task_sizes: Sequence[int],
        num_selectors: int = 10,
        num_prompts: int = 10,
        num_prompt_layers: int = 5,
        normalize: str = "pre-head",
    ) -> None:
        super().__init__()
        if not task_sizes or any(int(size) <= 0 for size in task_sizes):
            raise ValueError("MULTI-LANE task sizes must be positive")
        if num_selectors <= 0 or num_prompts <= 0:
            raise ValueError("MULTI-LANE selector/prompt counts must be positive")
        if normalize not in {"none", "pre-head"}:
            raise ValueError("MULTI-LANE normalize must be none or pre-head")
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
                "MULTI-LANE Track A requires an OpenAI CLIP VisionTransformer; "
                "missing " + ", ".join(missing)
            )
        if not hasattr(visual_encoder.transformer, "resblocks"):
            raise TypeError("CLIP visual transformer must expose residual blocks")
        blocks = list(visual_encoder.transformer.resblocks)
        if not blocks:
            raise ValueError("CLIP visual transformer has no residual blocks")
        if not 0 <= num_prompt_layers <= len(blocks):
            raise ValueError("MULTI-LANE prompt-layer count is invalid")

        width = int(visual_encoder.conv1.out_channels)
        output_dim = int(getattr(visual_encoder, "output_dim", -1))
        if width <= 0 or output_dim <= 0:
            raise ValueError("CLIP visual dimensions must be positive")
        first_attention = blocks[0].attn
        num_heads = int(first_attention.num_heads)
        if width % num_heads:
            raise ValueError("CLIP width must be divisible by attention heads")
        for block in blocks:
            block_width = int(block.attn.embed_dim)
            if block_width != width or int(block.attn.num_heads) != num_heads:
                raise ValueError("CLIP visual residual blocks are inconsistent")

        self.visual_encoder = visual_encoder
        self.visual_encoder.requires_grad_(False)
        self.width = width
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.num_selectors = int(num_selectors)
        self.num_prompts = int(num_prompts)
        self.num_prompt_layers = int(num_prompt_layers)
        self.normalize = normalize
        self._task_sizes = tuple(int(size) for size in task_sizes)
        self._current_task_id = -1

        selectors = torch.randn(
            len(self._task_sizes), self.num_selectors, self.width
        )
        nn.init.orthogonal_(selectors)
        self.selectors = nn.Parameter(selectors)

        prompts = nn.ParameterList()
        for _ in range(self.num_prompt_layers):
            value = torch.randn(
                2,
                len(self._task_sizes),
                self.num_prompts,
                self.num_heads,
                self.head_dim,
            )
            nn.init.orthogonal_(value)
            prompts.append(nn.Parameter(value))
        self.prompts = prompts

        self.head = nn.Linear(self.output_dim, self.num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        mask = torch.zeros(len(self._task_sizes), self.num_classes)
        offset = 0
        for task_id, size in enumerate(self._task_sizes):
            mask[task_id, offset : offset + size] = 1.0
            offset += size
        self.register_buffer("task_class_mask", mask, persistent=True)

    @property
    def task_sizes(self) -> Tuple[int, ...]:
        return self._task_sizes

    @property
    def num_tasks(self) -> int:
        return len(self._task_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._task_sizes)

    @property
    def current_task_id(self) -> int:
        return self._current_task_id

    @property
    def seen_classes(self) -> int:
        if self._current_task_id < 0:
            return 0
        return sum(self._task_sizes[: self._current_task_id + 1])

    def activate_task(self, task_id: int) -> None:
        expected = self._current_task_id + 1
        if task_id != expected:
            raise RuntimeError(
                f"MULTI-LANE tasks must be sequential: expected {expected}, "
                f"got {task_id}"
            )
        if not 0 <= task_id < self.num_tasks:
            raise ValueError("MULTI-LANE task id is outside the protocol")
        if task_id > 0:
            with torch.no_grad():
                self.selectors[task_id].copy_(self.selectors[task_id - 1])
                for prompt in self.prompts:
                    prompt[:, task_id].copy_(prompt[:, task_id - 1])
        self._current_task_id = int(task_id)

    def restore_task(self, task_id: int) -> None:
        if not -1 <= task_id < self.num_tasks:
            raise ValueError("MULTI-LANE restored task id is invalid")
        self._current_task_id = int(task_id)

    def _visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.visual_encoder
        x = images.to(dtype=visual.conv1.weight.dtype)
        with torch.no_grad():
            x = visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1)
            x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
            positional = visual.positional_embedding.to(x.dtype)
            if positional.shape[0] != x.shape[1]:
                raise ValueError(
                    "CLIP positional-token count differs from the input patches"
                )
            x = visual.ln_pre(x + positional)
        return x

    def _lane_ids(self, all_seen_lanes: bool) -> List[int]:
        if self._current_task_id < 0:
            raise RuntimeError("No MULTI-LANE task is active")
        if all_seen_lanes:
            return list(range(self._current_task_id + 1))
        return [self._current_task_id]

    def _initial_lane_tokens(
        self, batch_size: int, lane_ids: Sequence[int]
    ) -> torch.Tensor:
        selector = self.selectors[list(lane_ids)]
        selector = selector.unsqueeze(1).expand(-1, batch_size, -1, -1)
        cls = self.visual_encoder.class_embedding.to(selector.dtype)
        cls = cls.reshape(1, 1, 1, -1).expand(
            len(lane_ids), batch_size, 1, -1
        )
        return torch.cat([cls, selector], dim=2)

    def _prompt_attention(
        self,
        block: nn.Module,
        task_tokens: torch.Tensor,
        lane_ids: Sequence[int],
        layer_id: int,
    ) -> torch.Tensor:
        task_count, batch, token_count, width = task_tokens.shape
        attention = block.attn
        qkv = F.linear(
            task_tokens,
            attention.in_proj_weight,
            attention.in_proj_bias,
        )
        qkv = qkv.reshape(
            task_count,
            batch,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(3, 0, 1, 4, 2, 5)
        query, key, value = qkv.unbind(0)
        if layer_id < self.num_prompt_layers:
            prompt = self.prompts[layer_id][:, list(lane_ids)]
            prompt = prompt.permute(0, 1, 3, 2, 4)
            prompt = prompt.unsqueeze(2).expand(-1, -1, batch, -1, -1, -1)
            key = torch.cat([prompt[0], key], dim=-2)
            value = torch.cat([prompt[1], value], dim=-2)
        weights = torch.matmul(query, key.transpose(-2, -1))
        weights = torch.softmax(weights * (self.head_dim**-0.5), dim=-1)
        output = torch.matmul(weights, value)
        output = output.permute(0, 1, 3, 2, 4).reshape(
            task_count, batch, token_count, width
        )
        return F.linear(
            output,
            attention.out_proj.weight,
            attention.out_proj.bias,
        )

    def _lane_block(
        self,
        block: nn.Module,
        image_tokens: torch.Tensor,
        lane_tokens: torch.Tensor,
        lane_ids: Sequence[int],
        layer_id: int,
    ) -> torch.Tensor:
        # The released block applies its first LayerNorm before both selector
        # aggregation and prompt attention.  Keep the residual stream itself
        # unnormalized, as in the original pre-norm transformer.
        normalized_image = block.ln_1(image_tokens)
        normalized_lane = block.ln_1(lane_tokens)
        task_cls = normalized_lane[:, :, :1]
        selectors = normalized_lane[:, :, 1:]
        similarity = torch.einsum(
            "tbsc,bnc->tbsn", selectors, normalized_image.detach()
        ) * (self.width**-0.5)
        selected = torch.einsum(
            "bnc,tbsn->tbsc",
            normalized_image.detach(),
            torch.softmax(similarity, dim=-1),
        )
        summarized = torch.cat([task_cls, selected], dim=2)
        attention_output = self._prompt_attention(
            block,
            summarized,
            lane_ids,
            layer_id,
        )
        # Released drop-and-replace: retain the attended CLS update and put the
        # selector tokens back before the residual addition.
        update = torch.cat([attention_output[:, :, :1], selectors], dim=2)
        lane_tokens = lane_tokens + update
        lane_tokens = lane_tokens + block.mlp(block.ln_2(lane_tokens))
        return lane_tokens

    def encode_lanes(
        self, images: torch.Tensor, all_seen_lanes: bool
    ) -> torch.Tensor:
        lane_ids = self._lane_ids(all_seen_lanes)
        image_tokens = self._visual_tokens(images)
        lane_tokens = self._initial_lane_tokens(images.shape[0], lane_ids)
        for layer_id, block in enumerate(self.visual_encoder.transformer.resblocks):
            lane_tokens = self._lane_block(
                block,
                image_tokens,
                lane_tokens,
                lane_ids,
                layer_id,
            )
            with torch.no_grad():
                image_tokens = block(image_tokens.permute(1, 0, 2)).permute(
                    1, 0, 2
                )
        lane_tokens = self.visual_encoder.ln_post(lane_tokens)
        if self.visual_encoder.proj is not None:
            lane_tokens = lane_tokens @ self.visual_encoder.proj
        lane_cls = lane_tokens[:, :, 0].permute(1, 0, 2).float()
        if self.normalize == "pre-head":
            lane_cls = F.normalize(lane_cls, dim=-1)
        return lane_cls

    def lane_logits(
        self, images: torch.Tensor, all_seen_lanes: bool
    ) -> torch.Tensor:
        return self.head(self.encode_lanes(images, all_seen_lanes))

    def current_logits(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.current_all_logits(images)
        start = sum(self._task_sizes[: self._current_task_id])
        stop = start + self._task_sizes[self._current_task_id]
        return logits[:, start:stop]

    def current_all_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.lane_logits(images, all_seen_lanes=False)[:, 0]

    def seen_logits(self, images: torch.Tensor) -> torch.Tensor:
        lane_ids = self._lane_ids(all_seen_lanes=True)
        logits = self.lane_logits(images, all_seen_lanes=True)
        masks = self.task_class_mask[lane_ids].to(dtype=logits.dtype)
        combined = torch.sum(logits * masks.unsqueeze(0), dim=1)
        return combined[:, : self.seen_classes]

    def optimizer_parameters(self) -> Iterable[nn.Parameter]:
        yield self.selectors
        yield from self.prompts
        yield from self.head.parameters()

    def optimizer_parameter_names(self) -> Tuple[str, ...]:
        names = ["selectors"]
        names.extend(f"prompts.{index}" for index in range(len(self.prompts)))
        names.extend(("head.weight", "head.bias"))
        return tuple(names)

    def assert_visual_frozen(self) -> None:
        unexpected = [
            name
            for name, parameter in self.visual_encoder.named_parameters()
            if parameter.requires_grad
        ]
        if unexpected:
            raise RuntimeError(
                "MULTI-LANE visual encoder unexpectedly became trainable: "
                + ", ".join(unexpected)
            )
