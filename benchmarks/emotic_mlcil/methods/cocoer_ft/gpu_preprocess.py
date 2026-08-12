"""Batched CUDA three-view preprocessing for CocoER-FT v0.2."""

from __future__ import annotations

import itertools
from typing import Sequence, Tuple, Union

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


_PERMUTATIONS = tuple(itertools.permutations((0, 1, 2)))


class CocoERGPUPreprocessor:
    """Generate source-shaped context/body/head views on the model device."""

    def __init__(self, device: torch.device, seed: int, image_size: int = 224) -> None:
        if device.type != "cuda":
            raise ValueError("CocoER GPU preprocessing requires CUDA")
        self.device = device
        self.image_size = int(image_size)
        # Crop/flip/jitter parameters stay on CPU: their cost is negligible,
        # it avoids device synchronization for integer crop coordinates, and
        # the complete generator state remains checkpointable.
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.mean = torch.tensor(
            (0.485, 0.456, 0.406), device=device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            (0.229, 0.224, 0.225), device=device
        ).view(1, 3, 1, 1)
        self.permutations = torch.tensor(_PERMUTATIONS, dtype=torch.long)

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator)

    def _randint_inclusive(
        self, low: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        span = high - low + 1
        return low + torch.floor(self._rand(low.shape) * span.float()).long()

    def _augment_rois(
        self,
        boxes: torch.Tensor,
        sizes: torch.Tensor,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if boxes.ndim != 3 or tuple(boxes.shape[1:]) != (2, 4):
            raise ValueError("CocoER boxes must have shape [N,2,4]")
        if sizes.shape != (boxes.shape[0], 2):
            raise ValueError("CocoER image sizes must have shape [N,2]")
        body = boxes[:, 0].cpu().long()
        head = boxes[:, 1].cpu().long()
        height, width = sizes.cpu()[:, 0].long(), sizes.cpu()[:, 1].long()
        if not train:
            context = torch.stack(
                (torch.zeros_like(width), torch.zeros_like(height), width, height),
                dim=1,
            )
            return (
                torch.stack((context, body, head), dim=1).float(),
                boxes.cpu().float(),
                torch.zeros_like(width, dtype=torch.bool),
            )

        ux1 = self._randint_inclusive(torch.minimum(head[:, 0], body[:, 0]), head[:, 0])
        uy1 = self._randint_inclusive(torch.minimum(head[:, 1], body[:, 1]), head[:, 1])
        ux2 = self._randint_inclusive(head[:, 2], torch.maximum(head[:, 2], body[:, 2]))
        uy2 = self._randint_inclusive(head[:, 3], torch.maximum(head[:, 3], body[:, 3]))
        cx1 = self._randint_inclusive(torch.zeros_like(ux1), ux1)
        cy1 = self._randint_inclusive(torch.zeros_like(uy1), uy1)
        cx2 = self._randint_inclusive(ux2, width)
        cy2 = self._randint_inclusive(uy2, height)
        context = torch.stack((cx1, cy1, cx2, cy2), dim=1)
        # The released random_crop2 uses the union box for its body view.
        body_roi = torch.stack((ux1, uy1, ux2, uy2), dim=1)
        rois = torch.stack((context, body_roi, head), dim=1).float()
        relative_body = body_roi - context[:, [0, 1, 0, 1]]
        relative_head = head - context[:, [0, 1, 0, 1]]
        relative = torch.stack((relative_body, relative_head), dim=1).float()
        flip = self._rand((boxes.shape[0],)) < 0.5
        crop_width = (cx2 - cx1).float()
        flipped_x1 = crop_width[:, None] - relative[:, :, 2]
        flipped_x2 = crop_width[:, None] - relative[:, :, 0]
        relative[:, :, 0] = torch.where(flip[:, None], flipped_x1, relative[:, :, 0])
        relative[:, :, 2] = torch.where(flip[:, None], flipped_x2, relative[:, :, 2])
        return rois, relative, flip

    def _sample_rois(
        self,
        images: Sequence[torch.Tensor],
        rois: torch.Tensor,
        flip: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        # The released pipeline applies ColorJitter independently to each
        # variable-size view before Resize. Sample all parameters from the
        # checkpointed CPU generator, then execute the image operators on CUDA.
        count = len(images) * 3
        factors = 0.6 + 0.8 * self._rand((count, 3)) if train else None
        orders = (
            torch.floor(self._rand((count,)) * len(_PERMUTATIONS)).long()
            if train
            else None
        )
        output = []
        for index, cpu_image in enumerate(images):
            image = cpu_image.to(self.device, non_blocking=True).float().div_(255.0)
            sample_views = []
            for view_index, roi in enumerate(rois[index].long().tolist()):
                x1, y1, x2, y2 = roi
                view = TF.crop(
                    image,
                    top=y1,
                    left=x1,
                    height=y2 - y1,
                    width=x2 - x1,
                )
                if bool(flip[index]):
                    view = TF.hflip(view)
                if train:
                    flat_index = index * 3 + view_index
                    for operation in self.permutations[orders[flat_index]].tolist():
                        factor = float(factors[flat_index, operation])
                        if operation == 0:
                            view = TF.adjust_brightness(view, factor)
                        elif operation == 1:
                            view = TF.adjust_contrast(view, factor)
                        else:
                            view = TF.adjust_saturation(view, factor)
                view = TF.resize(
                    view,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                sample_views.append(view)
            output.append(torch.stack(sample_views))
        return torch.stack(output)

    def __call__(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
        boxes: torch.Tensor,
        sizes: torch.Tensor,
        *,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, torch.Tensor):
            if images.dtype != torch.uint8 or images.ndim != 4:
                raise ValueError("CocoER GPU preprocessing requires NCHW uint8")
            image_rows = list(images.unbind(0))
        else:
            image_rows = list(images)
        if not image_rows or any(
            value.dtype != torch.uint8 or value.ndim != 3 or value.shape[0] != 3
            for value in image_rows
        ):
            raise ValueError("CocoER GPU preprocessing requires CHW uint8 rows")
        if len(image_rows) != boxes.shape[0]:
            raise ValueError("CocoER image/box batch sizes differ")
        rois, relative, flip = self._augment_rois(boxes, sizes, train)
        views = self._sample_rois(image_rows, rois, flip, train)
        flat = views.flatten(0, 1)
        flat = (flat - self.mean) / self.std
        context_width = (rois[:, 0, 2] - rois[:, 0, 0]).clamp_min(1.0)
        context_height = (rois[:, 0, 3] - rois[:, 0, 1]).clamp_min(1.0)
        scale = torch.stack(
            (self.image_size / context_width, self.image_size / context_height),
            dim=1,
        )
        geometry = relative * scale[:, None, [0, 1, 0, 1]]
        return flat.view_as(views), geometry.float().to(self.device)
