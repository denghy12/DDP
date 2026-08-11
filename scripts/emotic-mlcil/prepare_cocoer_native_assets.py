#!/usr/bin/env python3
"""Prepare and hash CocoER's two generic native initialization assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESNET_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
RESNET_SHA256_PREFIX = "0676ba61"
CLIP_RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
CLIP_RN50_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{CLIP_RN50_SHA256}/RN50.pt"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "pretrained" / "cocoer",
    )
    parser.add_argument(
        "--clip-output",
        type=Path,
        default=ROOT / "pretrained" / "clip" / "RN50.pt",
    )
    parser.add_argument(
        "--resnet50-source",
        type=Path,
        help=(
            "Existing official resnet50-0676ba61.pth; omit to use "
            "torchvision's hash-checked downloader"
        ),
    )
    parser.add_argument(
        "--clip-rn50-source",
        type=Path,
        help="Existing official RN50.pt; omit to use OpenAI CLIP's hash-checked downloader",
    )
    args = parser.parse_args()

    import torch
    from torchvision.models import ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V1
    if weights.url != RESNET_URL:
        raise RuntimeError(f"Unexpected torchvision ResNet-50 URL: {weights.url}")
    if args.resnet50_source is None:
        state = weights.get_state_dict(progress=True, check_hash=True)
        raw_resnet_source = (
            Path(torch.hub.get_dir()) / "checkpoints" / Path(RESNET_URL).name
        )
    else:
        raw_resnet_source = args.resnet50_source.expanduser().resolve()
        if not raw_resnet_source.is_file():
            raise FileNotFoundError(raw_resnet_source)
        state = torch.load(raw_resnet_source, map_location="cpu")
    if not raw_resnet_source.is_file():
        raise FileNotFoundError(
            f"Torchvision did not leave its downloaded source at {raw_resnet_source}"
        )
    raw_resnet_sha = _sha(raw_resnet_source)
    if not raw_resnet_sha.startswith(RESNET_SHA256_PREFIX):
        raise ValueError(
            "Official ResNet-50 source SHA mismatch: "
            f"{raw_resnet_sha} does not start with {RESNET_SHA256_PREFIX}"
        )
    if not isinstance(state, dict):
        raise ValueError("Official ResNet-50 source is not a state dict")
    required_shapes = {
        "conv1.weight": (64, 3, 7, 7),
        "layer4.2.conv3.weight": (2048, 512, 1, 1),
        "fc.weight": (1000, 2048),
    }
    for name, shape in required_shapes.items():
        if name not in state or tuple(state[name].shape) != shape:
            raise ValueError(f"Official ResNet-50 tensor differs: {name}")
    resnet_output = args.output_dir / "resnet50_imagenet1k_v1.pth"
    resnet_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resnet_output.with_suffix(".pth.partial")
    torch.save(
        {
            "schema_version": 1,
            "asset_kind": "cocoer_torchvision_resnet50_imagenet1k_v1",
            "weights_enum": "ResNet50_Weights.IMAGENET1K_V1",
            "source_url": RESNET_URL,
            "source_filename_hash_prefix": RESNET_SHA256_PREFIX,
            "source_file_sha256": raw_resnet_sha,
            "state_dict": state,
        },
        temporary,
    )
    os.replace(temporary, resnet_output)

    clip_source = args.clip_rn50_source
    if clip_source is None:
        from clip import clip

        if clip._MODELS.get("RN50") != CLIP_RN50_URL:
            raise RuntimeError("Repository CLIP RN50 URL differs from OpenAI")
        clip_source = Path(clip._download(CLIP_RN50_URL)).resolve()
    else:
        clip_source = clip_source.expanduser().resolve()
    if not clip_source.is_file():
        raise FileNotFoundError(clip_source)
    observed_clip_sha = _sha(clip_source)
    if observed_clip_sha != CLIP_RN50_SHA256:
        raise ValueError(
            f"OpenAI CLIP RN50 SHA mismatch: {observed_clip_sha} != {CLIP_RN50_SHA256}"
        )
    clip_output = args.clip_output.expanduser().resolve()
    if clip_source != clip_output:
        _atomic_copy(clip_source, clip_output)

    manifest = {
        "schema_version": 1,
        "method": "CocoER-FT",
        "generic_initialization_only": True,
        "forbidden_full_emotic_assets": ["GWT checkpoint", "VI_weights/w.pth"],
        "resnet50": {
            "path": str(resnet_output.resolve()),
            "sha256": _sha(resnet_output),
            "weights_enum": "ResNet50_Weights.IMAGENET1K_V1",
            "source_url": RESNET_URL,
            "source_file_sha256": raw_resnet_sha,
        },
        "clip_rn50": {
            "path": str(clip_output),
            "sha256": _sha(clip_output),
            "expected_sha256": CLIP_RN50_SHA256,
            "source_url": CLIP_RN50_URL,
        },
    }
    manifest_path = args.output_dir / "native_assets_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
