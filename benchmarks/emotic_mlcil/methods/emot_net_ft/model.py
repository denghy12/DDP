"""PyTorch port of the native EMOT-Net release body/context architecture."""

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
            kernel_size=(1, kernel),
            stride=(1, stride),
            padding=(0, padding),
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            middle_channels,
            out_channels,
            kernel_size=(kernel, 1),
            stride=(stride, 1),
            padding=(padding, 0),
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
    """AlexNet body branch bundled in the official EMOT-Net release."""

    output_dim = 256

    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            nn.LocalResponseNorm(size=5, alpha=1.0e-4, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            nn.LocalResponseNorm(size=5, alpha=1.0e-4, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=16),
            nn.Flatten(1),
        )


class EMOTNetFTModel(nn.Module):
    """Native dual-stream EMOT-Net with protocol-ordered expanding heads."""

    context_dim = 640
    body_dim = 256

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

        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError("Unsupported EMOT-Net initialization schema")
        if payload.get("upstream_commit") != "69c3a5106aed08121cd12f6a5b359c745136931e":
            raise ValueError("EMOT-Net initialization has different upstream provenance")
        if payload.get("upstream_repository") != "https://github.com/rkosti/emotic":
            raise ValueError("EMOT-Net initialization repository differs")
        assets = payload.get("source_assets")
        expected_assets = {
            "model_myVDavg_640_Places.t7": (
                "bbf8a09edb1a17338f8004b78cf2e83f3cccc3a7f0bf7c3705368b482cab1e7c"
            ),
            "alexnet_features.t7": (
                "0abdbce4910f4c242433d614287448d110a81e7d26562ab291364762cf2dae87"
            ),
        }
        if not isinstance(assets, Mapping) or dict(assets) != expected_assets:
            raise ValueError("EMOT-Net initialization source assets differ")
        if (
            payload.get("release_archive_sha256")
            != "ce6096c1af5a3e91badbc06752e2dbbd04fc63f67f24acc95d76a68e1f7e339b"
        ):
            raise ValueError("EMOT-Net release archive provenance differs")
        if payload.get("native_body_variant") != "official_dropbox_alexnet":
            raise ValueError("EMOT-Net native body variant differs")
        expected_release_sources = {
            "Steps_for_training.md": (
                "0d21ce72cddfba3db1593a0ba58e78754b1d1492c1b906a0f9b69dc203004bce"
            ),
            "codes/OptsEmotionModel.lua": (
                "52be6bf4c07c0f81d3e0917bf039c827eaec46d5bfe5c9ad887c87526cf57526"
            ),
            "codes/CreateEmotionModel.lua": (
                "66355d632210f04058ace7a08a112e5443823bd7a1902bf48ef858d53be28ccb"
            ),
            "codes/trainTest_BI.lua": (
                "a4982bc8826a5c72070ba2223c92245062a5bc44ac4007c56408cd164886ef2c"
            ),
        }
        release_sources = payload.get("release_verified_file_sha256")
        if (
            not isinstance(release_sources, Mapping)
            or dict(release_sources) != expected_release_sources
        ):
            raise ValueError("EMOT-Net release source provenance differs")
        tower_selection = payload.get("tower_selection")
        if not isinstance(tower_selection, Mapping):
            raise ValueError("EMOT-Net tower selection provenance is missing")
        for name in ("context_encoder", "body_encoder"):
            selection = tower_selection.get(name)
            if (
                not isinstance(selection, Mapping)
                or int(selection.get("matching_tower_count", 0)) < 1
                or selection.get("selected_tower_index") != 0
                or selection.get("selection_rule")
                != "first_saved_replica_matching_upstream_features_get_1"
            ):
                raise ValueError(f"EMOT-Net {name} selection provenance differs")
            tower_hashes = selection.get("tower_sha256")
            if (
                not isinstance(tower_hashes, list)
                or len(tower_hashes) != int(selection["matching_tower_count"])
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in tower_hashes
                )
            ):
                raise ValueError(f"EMOT-Net {name} tower hashes differ")
        for name, module in (
            ("context_encoder", self.context_encoder),
            ("body_encoder", self.body_encoder),
        ):
            state = payload.get(name)
            if not isinstance(state, Mapping):
                raise ValueError(f"EMOT-Net initialization is missing {name}")
            module.load_state_dict(state, strict=True)
