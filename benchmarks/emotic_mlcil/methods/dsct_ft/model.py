"""Protocol-safe expanding-head wrapper for the official DSCT detector."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _patch_ms_deform_attention_amp(module: nn.Module) -> int:
    """Keep the legacy CUDA extension in FP32 inside an AMP model.

    The fixed extension dispatches only float/double.  Casting at the narrow
    extension boundary preserves the upstream operator while allowing the
    surrounding ResNet/Transformer projections to use Tensor Cores.
    """

    matched = 0
    patched_classes = set()
    for child in module.modules():
        if child.__class__.__name__ not in {"MSDeformAttn", "MSDeformAttnCtx"}:
            continue
        matched += 1
        module_class = child.__class__
        if module_class in patched_classes or getattr(
            module_class, "_dsct_amp_bridge_installed", False
        ):
            continue
        original = module_class.forward

        def forward_fp32(self, *args, _original=original, **kwargs):
            cast_args = tuple(
                value.float()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value
                for value in args
            )
            cast_kwargs = {
                key: (
                    value.float()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )
                for key, value in kwargs.items()
            }
            with torch.cuda.amp.autocast(enabled=False):
                result = _original(self, *cast_args, **cast_kwargs)
            return result

        forward_fp32._dsct_amp_bridge = True
        module_class.forward = forward_fp32
        module_class._dsct_amp_bridge_installed = True
        patched_classes.add(module_class)
    return matched


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (center_x - width / 2, center_y - height / 2,
         center_x + width / 2, center_y + height / 2), dim=-1
    )


def generalized_box_iou_aligned(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Aligned generalized IoU for valid xyxy boxes."""

    left_top = torch.maximum(first[:, :2], second[:, :2])
    right_bottom = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)
    area_first = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(dim=1)
    area_second = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(dim=1)
    union = (area_first + area_second - intersection).clamp(min=1.0e-8)
    iou = intersection / union
    enclosure_left_top = torch.minimum(first[:, :2], second[:, :2])
    enclosure_right_bottom = torch.maximum(first[:, 2:], second[:, 2:])
    enclosure = (enclosure_right_bottom - enclosure_left_top).clamp(min=0).prod(dim=1).clamp(min=1.0e-8)
    return iou - (enclosure - union) / enclosure


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_target = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = cross_entropy * ((1.0 - probability_target) ** gamma)
    if alpha >= 0:
        alpha_target = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_target * loss
    return loss.mean(1).sum() / max(1, logits.shape[0])


def target_boxes_from_transport(images: torch.Tensor) -> torch.Tensor:
    """Recover normalized cxcywh target boxes from [RGB, box, valid]."""

    if images.ndim != 4 or images.shape[1] != 5:
        raise ValueError("DSCT transport must be [N, 5, H, W]")
    boxes = []
    for box_mask, valid_mask in zip(images[:, 3] > 0.5, images[:, 4] > 0.5):
        valid_rows = valid_mask.any(dim=1).nonzero().flatten()
        valid_cols = valid_mask.any(dim=0).nonzero().flatten()
        box_rows = box_mask.any(dim=1).nonzero().flatten()
        box_cols = box_mask.any(dim=0).nonzero().flatten()
        if not len(valid_rows) or not len(valid_cols) or not len(box_rows) or not len(box_cols):
            raise ValueError("DSCT transport contains an empty valid region or target box")
        height = float(valid_rows[-1].item() + 1)
        width = float(valid_cols[-1].item() + 1)
        x1, x2 = float(box_cols[0]), float(box_cols[-1].item() + 1)
        y1, y2 = float(box_rows[0]), float(box_rows[-1].item() + 1)
        boxes.append(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                      (x2 - x1) / width, (y2 - y1) / height))
    return images.new_tensor(boxes)


def target_sizes_from_transport(images: torch.Tensor) -> torch.Tensor:
    sizes = []
    for valid_mask in images[:, 4] > 0.5:
        rows = valid_mask.any(dim=1).nonzero().flatten()
        columns = valid_mask.any(dim=0).nonzero().flatten()
        if not len(rows) or not len(columns):
            raise ValueError("DSCT transport has an empty valid region")
        sizes.append((float(rows[-1].item() + 1), float(columns[-1].item() + 1)))
    return images.new_tensor(sizes)


def target_query_indices(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Official inference mapping: choose the subject query with maximum IoU."""

    predicted = box_cxcywh_to_xyxy(predicted_boxes)
    target = box_cxcywh_to_xyxy(target_boxes)
    if target_sizes is None:
        target_sizes = predicted_boxes.new_ones((predicted_boxes.shape[0], 2))
    selected = []
    for batch_index in range(predicted.shape[0]):
        height, width = target_sizes[batch_index]
        scale = torch.stack((width, height, width, height))
        boxes = predicted[batch_index] * scale
        reference = target[batch_index].unsqueeze(0) * scale
        left_top = torch.maximum(boxes[:, :2], reference[:, :2])
        right_bottom = torch.minimum(boxes[:, 2:], reference[:, 2:])
        intersection = (right_bottom - left_top + 1).clamp(min=0).prod(dim=1)
        area_boxes = (boxes[:, 2:] - boxes[:, :2] + 1).clamp(min=0).prod(dim=1)
        area_reference = (reference[:, 2:] - reference[:, :2] + 1).clamp(min=0).prod(dim=1)
        iou = intersection / (area_boxes + area_reference - intersection).clamp(min=1.0e-8)
        selected.append(iou.argmax())
    return torch.stack(selected)


class DSCTTaskClassifier(nn.Module):
    """Shared subjectness plus protocol-ordered expanding emotion heads."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = nn.ModuleList()
        self.subjectness = nn.Linear(self.hidden_dim, 1)
        self._head_sizes = []
        prior_bias = -torch.log(torch.tensor(99.0)).item()
        nn.init.constant_(self.subjectness.bias, prior_bias)

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    def add_head(self, classes: int) -> nn.Linear:
        if classes <= 0:
            raise ValueError("DSCT task heads must be non-empty")
        head = nn.Linear(self.hidden_dim, int(classes))
        nn.init.xavier_uniform_(head.weight)
        nn.init.constant_(head.bias, -torch.log(torch.tensor(99.0)).item())
        self.heads.append(head)
        self._head_sizes.append(int(classes))
        return head

    def restore_heads(self, sizes: Iterable[int]) -> None:
        if self.heads:
            raise RuntimeError("restore_heads requires an empty classifier")
        for size in sizes:
            self.add_head(int(size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        emotions = [head(features) for head in self.heads]
        if not emotions:
            raise RuntimeError("No DSCT emotion head has been added")
        return torch.cat((*emotions, self.subjectness(features)), dim=-1)


class DSCTFTModel(nn.Module):
    """Wrap the immutable upstream DSCT core without vendoring its source."""

    def __init__(
        self,
        core: nn.Module,
        nested_tensor_factory: Callable[[torch.Tensor, torch.Tensor], object],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.core = core
        self.amp_bridge_operator_count = _patch_ms_deform_attention_amp(core)
        self.nested_tensor_factory = nested_tensor_factory
        self.classifier = DSCTTaskClassifier(hidden_dim)
        number_of_layers = len(core.class_embed_dsct)
        core.class_embed_dsct = nn.ModuleList([self.classifier for _ in range(number_of_layers)])

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return self.classifier.head_sizes

    @property
    def num_classes(self) -> int:
        return self.classifier.num_classes

    def add_head(self, classes: int) -> nn.Linear:
        return self.classifier.add_head(classes)

    def restore_heads(self, sizes: Iterable[int]) -> None:
        self.classifier.restore_heads(sizes)

    def forward(self, images: torch.Tensor) -> Mapping[str, torch.Tensor]:
        target_boxes = target_boxes_from_transport(images)
        target_sizes = target_sizes_from_transport(images)
        valid = images[:, 4] > 0.5
        rgb = images[:, :3]
        if images.is_contiguous(memory_format=torch.channels_last):
            rgb = rgb.contiguous(memory_format=torch.channels_last)
        nested = self.nested_tensor_factory(rgb, ~valid)
        output = dict(self.core(nested))
        output["target_boxes"] = target_boxes
        output["target_sizes"] = target_sizes
        return output

    @staticmethod
    def selected_scores(output: Mapping[str, torch.Tensor], current_only: int = 0) -> torch.Tensor:
        indices = target_query_indices(
            output["pred_boxes"], output["target_boxes"], output.get("target_sizes")
        )
        logits = output["pred_logits"]
        selected = logits[torch.arange(logits.shape[0], device=logits.device), indices, :-1]
        return selected[:, -current_only:] if current_only else selected


def dsct_current_task_loss(
    output: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    current_classes: int,
    *,
    class_cost: float = 2.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
    class_loss_weight: float = 5.0,
    bbox_loss_weight: float = 5.0,
    giou_loss_weight: float = 2.0,
    focal_alpha: float = 0.25,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Source DSCT one-target matching/loss restricted to current labels."""

    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    target_boxes = output["target_boxes"]
    emotion_logits = logits[..., -(current_classes + 1):-1]
    probability = emotion_logits.sigmoid()
    expanded_targets = targets[:, None, :].expand_as(probability)
    negative = (1 - focal_alpha) * probability.pow(2) * (-(1 - probability + 1e-8).log())
    positive = focal_alpha * (1 - probability).pow(2) * (-(probability + 1e-8).log())
    classification_cost = ((positive - negative) * expanded_targets).mean(dim=-1)
    bbox_match_cost = torch.cdist(boxes, target_boxes.unsqueeze(1), p=1).squeeze(-1)
    giou_match_cost = []
    for batch_index in range(boxes.shape[0]):
        repeated = target_boxes[batch_index].expand(boxes.shape[1], -1)
        giou_match_cost.append(-generalized_box_iou_aligned(
            box_cxcywh_to_xyxy(boxes[batch_index]), box_cxcywh_to_xyxy(repeated)
        ))
    giou_match_cost = torch.stack(giou_match_cost)
    matched = (class_cost * classification_cost + bbox_cost * bbox_match_cost + giou_cost * giou_match_cost).argmin(1)
    classification_target = torch.zeros_like(logits[..., -(current_classes + 1):])
    rows = torch.arange(logits.shape[0], device=logits.device)
    classification_target[rows, matched, :current_classes] = targets
    classification_target[rows, matched, current_classes] = 1.0
    classification = sigmoid_focal_loss(
        logits[..., -(current_classes + 1):], classification_target, alpha=focal_alpha
    ) * logits.shape[1]
    matched_boxes = boxes[rows, matched]
    bbox = F.l1_loss(matched_boxes, target_boxes, reduction="none").sum() / logits.shape[0]
    giou = (1.0 - generalized_box_iou_aligned(
        box_cxcywh_to_xyxy(matched_boxes), box_cxcywh_to_xyxy(target_boxes)
    )).sum() / logits.shape[0]
    total = class_loss_weight * classification + bbox_loss_weight * bbox + giou_loss_weight * giou
    return total, {"classification": classification, "bbox": bbox, "giou": giou, "matched": matched}
