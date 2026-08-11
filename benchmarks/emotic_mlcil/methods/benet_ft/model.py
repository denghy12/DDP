"""Runtime adapter for the immutable official BENet HigherHRNet source."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


UPSTREAM_REPOSITORY = "https://github.com/TristanCladiere/BENet"
UPSTREAM_COMMIT = "b86747e0e259b1ec70fc84ca76efd7ea3bb3728e"
VERIFIED_SOURCE_SHA256 = {
    "lib/models/BENet.py": "e17d68ba5a42e46fc6496dbdfd7b8686bb7b24467cf556f4b8a989fd7ed7512c",
    "lib/core/loss_mt.py": "2e38e401d685f7465729f64776940a0a0490049f65629b50d1bf017c748cee3e",
    "experiments/emotic/emotic_acivs_1_no_fusion.yaml": "522e8c76ba1a238e8dccffcf8d67618859053faa54eb03783e2e7338547aca32",
}


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_benet_source(source_root: Path) -> Dict[str, str]:
    source_root = source_root.expanduser().resolve()
    observed = {}
    for relative, expected in VERIFIED_SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixed BENet source file: {path}")
        observed[relative] = _sha256(path)
        if observed[relative] != expected:
            raise ValueError(f"BENet source hash differs for {relative}")
    return observed


def _official_config(classes: int, pretrained: str):
    deconv = _AttrDict(
        NUM_DECONVS=1,
        NUM_CHANNELS=[32],
        KERNEL_SIZE=[4],
        NUM_BASIC_BLOCKS=3,
        CAT_OUTPUT=[True],
    )
    extra = _AttrDict(
        FINAL_CONV_KERNEL=1,
        PRETRAINED_LAYERS=["*"],
        STAGE2=_AttrDict(NUM_MODULES=2, NUM_BRANCHES=2, BLOCK="BASIC", NUM_BLOCKS=[3, 3], NUM_CHANNELS=[32, 64], FUSE_METHOD="SUM"),
        STAGE3=_AttrDict(NUM_MODULES=2, NUM_BRANCHES=3, BLOCK="BASIC", NUM_BLOCKS=[3, 3, 3], NUM_CHANNELS=[32, 64, 128], FUSE_METHOD="SUM"),
        STAGE4=_AttrDict(NUM_MODULES=2, NUM_BRANCHES=4, BLOCK="BASIC", NUM_BLOCKS=[3, 3, 3, 3], NUM_CHANNELS=[32, 64, 128, 256], FUSE_METHOD="SUM"),
        DECONV=deconv,
    )
    model = SimpleNamespace(
        NUM_CAT_EMOTIONS=int(classes),
        NUM_CONT_EMOTIONS=0,
        HEAD_EXPANSION=1,
        INIT_WEIGHTS=True,
        PRETRAINED=pretrained,
        EXTRA=extra,
    )
    cfg = SimpleNamespace(
        DATASET=SimpleNamespace(NUM_CAT_MIXED=0),
        LOSS=SimpleNamespace(),
        VERBOSE=False,
    )
    return cfg, model


def load_official_benet_core(source_root: Path, pretrained_weights: Path):
    source_root = source_root.expanduser().resolve()
    pretrained_weights = pretrained_weights.expanduser().resolve()
    hashes = verify_benet_source(source_root)
    if not pretrained_weights.is_file():
        raise FileNotFoundError(f"Missing BENet HigherHRNet initialization: {pretrained_weights}")
    source_file = source_root / "lib/models/BENet.py"
    spec = importlib.util.spec_from_file_location("_emotic_benet_fixed_source", source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load BENet source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg, model_cfg = _official_config(26, str(pretrained_weights))
    core = module.BENet(cfg, model_cfg)
    core.init_weights(str(pretrained_weights), verbose=False)
    return core, module.BasicBlock, {
        "source_files": hashes,
        "pretrained_weights_sha256": _sha256(pretrained_weights),
    }


class BENetBottomUpTaskHead(nn.Module):
    """One protocol task's native two-resolution BENet BU output path."""

    def __init__(self, classes: int, block_class: Callable[..., nn.Module]) -> None:
        super().__init__()
        if classes <= 0:
            raise ValueError("BENet task heads require at least one class")
        middle = (32 + int(classes)) // 2
        self.low_refine = nn.Sequential(*[block_class(32, 32) for _ in range(3)])
        self.low_classifier = nn.Sequential(
            nn.Conv2d(32, middle, 3, padding=1),
            nn.BatchNorm2d(middle),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle, classes, 1),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32 + classes, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            *[block_class(32, 32) for _ in range(3)],
        )
        self.high_classifier = nn.Sequential(
            nn.Conv2d(32, middle, 3, padding=1),
            nn.BatchNorm2d(middle),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle, classes, 1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low = self.low_classifier(self.low_refine(features))
        high_features = self.upsample(torch.cat((features, low), dim=1))
        return low, self.high_classifier(high_features)


def _tag_head(classes: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, classes))


class BENetFTModel(nn.Module):
    """Official BENet trunk with protocol-ordered expanding BU/PC/BG heads."""

    def __init__(
        self,
        core: nn.Module,
        block_class: Callable[..., nn.Module],
        source_provenance: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.block_class = block_class
        self.source_provenance = dict(source_provenance or {})
        # Remove static 26-class modules. The shared detector and three native
        # feature paths remain; task heads below are the only class-dependent modules.
        for name in ("bu_final_layers", "bu_deconv_layers", "pc_classifier", "classifier", "fusion"):
            if hasattr(self.core, name):
                setattr(self.core, name, nn.Identity())
        self.bu_heads = nn.ModuleList()
        self.pc_heads = nn.ModuleList()
        self.context_heads = nn.ModuleList()
        self._head_sizes = []

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return tuple(self._head_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._head_sizes)

    def add_head(self, classes: int) -> None:
        self.bu_heads.append(BENetBottomUpTaskHead(classes, self.block_class))
        self.pc_heads.append(_tag_head(classes))
        self.context_heads.append(_tag_head(classes))
        self._head_sizes.append(int(classes))

    def restore_heads(self, sizes: Iterable[int]) -> None:
        if self._head_sizes:
            raise RuntimeError("restore_heads requires an unexpanded BENet model")
        for size in sizes:
            self.add_head(int(size))

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 11:
            raise ValueError("BENet requires benet_views [N, 11, H, W] images")

    def _backbone(self, images: torch.Tensor) -> Sequence[torch.Tensor]:
        c = self.core
        x = c.relu(c.bn1(c.conv1(images)))
        x = c.relu(c.bn2(c.conv2(x)))
        x = c.layer1(x)
        x_list = [c.transition1[i](x) if c.transition1[i] is not None else x for i in range(c.stage2_cfg["NUM_BRANCHES"])]
        y_list = c.stage2(x_list)
        x_list = [c.transition2[i](y_list[-1]) if c.transition2[i] is not None else y_list[i] for i in range(c.stage3_cfg["NUM_BRANCHES"])]
        y_list = c.stage3(x_list)
        x_list = [c.transition3[i](y_list[-1]) if c.transition3[i] is not None else y_list[i] for i in range(c.stage4_cfg["NUM_BRANCHES"])]
        return c.stage4(x_list)

    @staticmethod
    def _top_down(features: Sequence[torch.Tensor], incre, downsample, final) -> torch.Tensor:
        value = incre[0](features[0])
        for index in range(len(downsample)):
            value = incre[index + 1](features[index + 1]) + downsample[index](value)
        value = final(value)
        return F.adaptive_avg_pool2d(value, 1).flatten(1)

    @staticmethod
    def _sample_maps(maps: Sequence[torch.Tensor], box_mask: torch.Tensor) -> torch.Tensor:
        coordinates = []
        for mask in box_mask:
            points = torch.nonzero(mask > 0.5, as_tuple=False)
            if points.numel() == 0:
                raise ValueError("BENet target-box transport mask is empty")
            coordinates.append(points.float().mean(dim=0))
        sampled = []
        source_h, source_w = box_mask.shape[-2:]
        for value in maps:
            rows = []
            for index, center in enumerate(coordinates):
                y = min(value.shape[-2] - 1, max(0, int(round(float(center[0]) * value.shape[-2] / source_h))))
                x = min(value.shape[-1] - 1, max(0, int(round(float(center[1]) * value.shape[-1] / source_w))))
                rows.append(value[index, :, y, x])
            sampled.append(torch.stack(rows))
        return torch.stack(sampled).mean(dim=0)

    def branch_logits(self, images: torch.Tensor, branch: str, *, current_only: bool) -> torch.Tensor:
        self._validate_images(images)
        if not self._head_sizes:
            raise RuntimeError("No BENet task head has been added")
        view = {"bu": images[:, 0:3], "pc": images[:, 3:6], "context": images[:, 6:9]}[branch]
        features = self._backbone(view)
        indices = [-1] if current_only else range(len(self._head_sizes))
        outputs = []
        for index in indices:
            if branch == "bu":
                outputs.append(self._sample_maps(self.bu_heads[index](features[0]), images[:, 9]))
            elif branch == "pc":
                descriptor = self._top_down(features, self.core.pc_incre_modules, self.core.pc_downsamp_modules, self.core.pc_final_layer)
                outputs.append(self.pc_heads[index](descriptor))
            else:
                descriptor = self._top_down(features, self.core.incre_modules, self.core.downsamp_modules, self.core.final_layer)
                outputs.append(self.context_heads[index](descriptor))
        return torch.cat(outputs, dim=1)

    def detection_outputs(self, images: torch.Tensor):
        self._validate_images(images)
        features = self._backbone(images[:, 0:3])[0]
        c = self.core
        heatmaps, sizes = [c.hm_final_layers[0](features)], [c.hw_final_layers[0](features)]
        x_hm = torch.cat((features, heatmaps[0]), dim=1)
        x_hw = torch.cat((features, sizes[0]), dim=1)
        x_hm = c.hm_deconv_layers[0](x_hm)
        x_hw = c.hw_deconv_layers[0](x_hw)
        heatmaps.append(c.hm_final_layers[1](x_hm))
        sizes.append(c.hw_final_layers[1](x_hw))
        return heatmaps, sizes

    def current_logits(self, images: torch.Tensor, branch: str) -> torch.Tensor:
        return self.branch_logits(images, branch, current_only=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probabilities = [
            torch.sigmoid(self.branch_logits(images, branch, current_only=False))
            for branch in ("bu", "pc", "context")
        ]
        return torch.stack(probabilities).mean(dim=0)

