"""Exact ResNet-152-v1 topology used by the official Places365 Caffe model."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Tuple

import torch
from torch import nn


PLACES365_CAFFE_MODEL_SHA256 = (
    "1ba80ddb96c38c493f923f9e5346cbc5def66e0245e93dbfd5f67df7e706b2f5"
)
PLACES365_PROTOTXT_SHA256 = (
    "f99e87362b245faa4a3514be4af089bba60d1f6e711f396e8a7bed86e0fdc893"
)
PLACES365_REPOSITORY = "https://github.com/CSAILVision/places365"
PLACES365_REPOSITORY_COMMIT = "8a953ed56438726dc98bdef3796d042e7f1f171e"
PLACES365_CAFFE_MODEL_URL = (
    "http://places2.csail.mit.edu/models_places365/"
    "resnet152_places365.caffemodel"
)


def _block_names(stage: int, count: int) -> Tuple[str, ...]:
    if stage in (2, 5):
        return tuple(chr(ord("a") + index) for index in range(count))
    return ("a",) + tuple(f"b{index}" for index in range(1, count))


class CaffeBatchNormScale(nn.Module):
    """Caffe BatchNorm followed by Scale, represented as one inference op."""

    def __init__(self, channels: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("variance", torch.ones(channels))
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1, 1)
        normalized = (values - self.mean.view(shape)) * torch.rsqrt(
            self.variance.view(shape) + self.eps
        )
        return normalized * self.weight.view(shape) + self.bias.view(shape)


class CaffeResNetBottleneck(nn.Module):
    def __init__(self, input_channels: int, width: int, output_channels: int,
                 stride: int, projection: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, width, 1, stride=stride, bias=False)
        self.bn1 = CaffeBatchNormScale(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn2 = CaffeBatchNormScale(width)
        self.conv3 = nn.Conv2d(width, output_channels, 1, bias=False)
        self.bn3 = CaffeBatchNormScale(output_channels)
        self.projection = (
            nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
            if projection else None
        )
        self.projection_bn = CaffeBatchNormScale(output_channels) if projection else None
        self.relu = nn.ReLU(inplace=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        identity = values
        output = self.relu(self.bn1(self.conv1(values)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.projection is not None and self.projection_bn is not None:
            identity = self.projection_bn(self.projection(identity))
        return self.relu(output + identity)


class CaffeResNet152Places365(nn.Module):
    """The official Caffe graph through its 2048-D ``pool5`` output."""

    stage_spec = (
        (2, 64, 256, 3), (3, 128, 512, 8),
        (4, 256, 1024, 36), (5, 512, 2048, 3),
    )

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = CaffeBatchNormScale(64)
        self.relu = nn.ReLU(inplace=False)
        # The released prototxt has no padding on pool1. This differs from
        # torchvision ResNet and rules out a torchvision state-dict shortcut.
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        stages = OrderedDict()
        input_channels = 64
        for stage, width, output_channels, count in self.stage_spec:
            blocks = OrderedDict()
            for index, name in enumerate(_block_names(stage, count)):
                stride = 2 if stage > 2 and index == 0 else 1
                blocks[name] = CaffeResNetBottleneck(
                    input_channels, width, output_channels, stride, index == 0
                )
                input_channels = output_channels
            stages[str(stage)] = nn.Sequential(blocks)
        self.stages = nn.ModuleDict(stages)
        self.pool5 = nn.AvgPool2d(7, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.pool1(self.relu(self.bn1(self.conv1(values))))
        for stage in ("2", "3", "4", "5"):
            output = self.stages[stage](output)
        return torch.flatten(self.pool5(output), 1)


def iter_caffe_blocks(model: CaffeResNet152Places365) -> Iterable[
    Tuple[int, str, CaffeResNetBottleneck]
]:
    for stage, _, _, count in model.stage_spec:
        for name in _block_names(stage, count):
            yield stage, name, model.stages[str(stage)]._modules[name]
