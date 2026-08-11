"""PyTorch port of the native EMOT-Net body/context architecture."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _factorized(
    in_channels: int,
    middle_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    *,
    padding: Optional[int] = None,
    batch_norm_after: bool = True,
) -> Iterable[nn.Module]:
    padding = kernel // 2 if padding is None else int(padding)
    layers = [
        nn.Conv2d(
            in_channels,
            middle_channels,
            kernel_size=(kernel, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            middle_channels,
            out_channels,
            kernel_size=(1, kernel),
            stride=(1, stride),
            padding=(0, padding),
        ),
    ]
    if batch_norm_after:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


class EMOTNetContextEncoder(nn.Sequential):
    """Official Places-context branch, ending in a 640-D descriptor."""

    output_dim = 640

    def __init__(self) -> None:
        layers = []
        layers.extend(_factorized(3, 32, 64, 11, 4, padding=2))
        layers.extend(_factorized(64, 128, 256, 5, 2))
        layers.extend(_factorized(256, 384, 512, 3, 2))
        layers.extend(_factorized(512, 384, 384, 3, 1))
        layers.extend(_factorized(384, 640, 640, 3, 2))
        layers.extend(_factorized(640, 640, 640, 3, 1))
        layers.extend(_factorized(640, 640, 640, 3, 2))
        layers.extend(_factorized(640, 640, 640, 3, 1))
        layers.extend((nn.AvgPool2d(kernel_size=4, stride=1), nn.Flatten(1)))
        super().__init__(*layers)


class EMOTNetBodyEncoder(nn.Sequential):
    """Official DecomposeMe-style body branch, ending in 128 dimensions."""

    output_dim = 128

    def __init__(self) -> None:
        layers = []
        layers.extend(_factorized(3, 32, 64, 3, 2))
        layers.extend(
            (
                nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, (1, 3), (1, 2), (0, 1)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
            )
        )
        layers.extend(_factorized(128, 128, 128, 3, 2))
        layers.extend((nn.AvgPool2d(kernel_size=3, stride=16), nn.Flatten(1)))
        super().__init__(*layers)


class EMOTNetFTModel(nn.Module):
    """Native dual-stream EMOT-Net with protocol-ordered expanding heads."""

    context_dim = 640
    body_dim = 128

    def __init__(
        self,
        context_encoder: Optional[nn.Module] = None,
        body_encoder: Optional[nn.Module] = None,
        fusion_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if fusion_dim <= 0:
            raise ValueError("fusion_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.context_encoder = context_encoder or EMOTNetContextEncoder()
        self.body_encoder = body_encoder or EMOTNetBodyEncoder()
        self.fusion_dim = int(fusion_dim)
        self.fusion = nn.Linear(self.context_dim + self.body_dim, self.fusion_dim)
        self.fusion_norm = nn.BatchNorm1d(self.fusion_dim, eps=1.0e-3)
        self.dropout = nn.Dropout(float(dropout))
        self.heads = nn.ModuleList()
        self._head_sizes = []
        self._reset_native_parameters()

    def _reset_native_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    def add_head(self, classes: int) -> nn.Linear:
        if classes <= 0:
            raise ValueError("A task head must contain at least one class")
        head = nn.Linear(self.fusion_dim, int(classes))
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

    @staticmethod
    def _split_views(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5 or images.shape[1] != 2:
            raise ValueError(
                "EMOT-Net requires body_context images shaped [N, 2, C, H, W]"
            )
        if images.shape[-2:] != (224, 224):
            raise ValueError("EMOT-Net context transport canvas must be 224x224")
        # The body view is directly resized to 128x128 by the loader and then
        # zero-padded solely so both views can cross the tensor-only benchmark
        # boundary. Never feed the padded canvas into the native body tower.
        return images[:, 0], images[:, 1, :, :128, :128]

    @staticmethod
    def _validate_descriptor(
        value: torch.Tensor,
        batch: int,
        width: int,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape != (batch, width):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(f"{name} must return [{batch}, {width}], got {shape}")
        return value.float()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        context, body = self._split_views(images)
        context_features = self._validate_descriptor(
            self.context_encoder(context), images.shape[0], self.context_dim, "context encoder"
        )
        body_features = self._validate_descriptor(
            self.body_encoder(body), images.shape[0], self.body_dim, "body encoder"
        )
        features = self.fusion(torch.cat((context_features, body_features), dim=1))
        # Torch BatchNorm cannot estimate variance for a singleton last batch.
        # Reusing the accumulated statistics preserves the official layer while
        # allowing protocol loaders to keep every sample.
        if self.training and features.shape[0] == 1:
            features = F.batch_norm(
                features,
                self.fusion_norm.running_mean,
                self.fusion_norm.running_var,
                self.fusion_norm.weight,
                self.fusion_norm.bias,
                training=False,
                eps=self.fusion_norm.eps,
            )
        else:
            features = self.fusion_norm(features)
        return self.dropout(F.relu(features, inplace=False))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No EMOT-Net task head has been added")
        features = self.encode(images)
        return torch.cat([head(features) for head in self.heads], dim=1)

    def current_logits(self, images: torch.Tensor) -> torch.Tensor:
        if not self.heads:
            raise RuntimeError("No EMOT-Net task head has been added")
        return self.heads[-1](self.encode(images))

    def load_native_initialization(self, payload: Mapping[str, object]) -> None:
        """Load a separately audited conversion of the two official `.t7` towers."""

        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported EMOT-Net initialization schema")
        if payload.get("upstream_commit") != "69c3a5106aed08121cd12f6a5b359c745136931e":
            raise ValueError("EMOT-Net initialization has different upstream provenance")
        if payload.get("upstream_repository") != "https://github.com/rkosti/emotic":
            raise ValueError("EMOT-Net initialization repository differs")
        assets = payload.get("source_assets")
        required_assets = {
            "model_myVDavg_640_Places.t7",
            "myVD_ImgNet_66_old.t7",
        }
        if not isinstance(assets, Mapping) or set(assets) != required_assets:
            raise ValueError("EMOT-Net initialization source asset manifest differs")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in assets.values()
        ):
            raise ValueError("EMOT-Net initialization source SHA-256 is invalid")
        for name, module in (
            ("context_encoder", self.context_encoder),
            ("body_encoder", self.body_encoder),
        ):
            state = payload.get(name)
            if not isinstance(state, Mapping):
                raise ValueError(f"EMOT-Net initialization is missing {name}")
            module.load_state_dict(state, strict=True)
