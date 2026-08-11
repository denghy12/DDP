#!/usr/bin/env python3
"""Jointly audit all generic/native CocoER-FT runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMMIT = "dac8fc139e61b87f1bf0b27c581798df2a5a9d38"
CLIP_RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
RESNET_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_box(raw):
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    values = [float(value) for value in raw]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--resnet50-init", required=True, type=Path)
    parser.add_argument("--clip-rn50", required=True, type=Path)
    parser.add_argument("--head-cache", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from PIL import Image

    from src.helper_functions.emotic_loader import EMOTIC

    for path in (args.resnet50_init, args.clip_rn50, args.head_cache):
        if not path.is_file():
            raise FileNotFoundError(path)

    resnet = torch.load(args.resnet50_init, map_location="cpu")
    expected_resnet = {
        "schema_version": 1,
        "asset_kind": "cocoer_torchvision_resnet50_imagenet1k_v1",
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V1",
        "source_url": RESNET_URL,
        "source_filename_hash_prefix": "0676ba61",
    }
    if not isinstance(resnet, dict):
        raise ValueError("CocoER ResNet-50 asset is not a mapping")
    for key, expected in expected_resnet.items():
        if resnet.get(key) != expected:
            raise ValueError(f"CocoER ResNet-50 {key} differs")
    source_sha = str(resnet.get("source_file_sha256", "")).lower()
    if len(source_sha) != 64 or not source_sha.startswith("0676ba61"):
        raise ValueError("CocoER ResNet-50 source-file SHA-256 differs")
    state = resnet.get("state_dict")
    expected_shapes = {
        "conv1.weight": (64, 3, 7, 7),
        "layer4.2.conv3.weight": (2048, 512, 1, 1),
        "fc.weight": (1000, 2048),
    }
    if not isinstance(state, dict):
        raise ValueError("CocoER ResNet-50 state_dict is missing")
    for name, shape in expected_shapes.items():
        if name not in state or tuple(state[name].shape) != shape:
            raise ValueError(f"CocoER ResNet-50 tensor differs: {name}")

    clip_sha = _sha(args.clip_rn50)
    if clip_sha != CLIP_RN50_SHA256:
        raise ValueError("CocoER CLIP RN50 SHA-256 differs from OpenAI")

    cache = json.loads(args.head_cache.read_text(encoding="utf-8"))
    if (
        not isinstance(cache, dict)
        or int(cache.get("schema_version", -1)) != 1
        or cache.get("upstream_commit") != COMMIT
    ):
        raise ValueError("CocoER head cache provenance differs")
    detector = cache.get("detector")
    if (
        not isinstance(detector, dict)
        or detector.get("library") != "insightface"
        or detector.get("version") != "0.7.3"
        or detector.get("model") != "buffalo_l"
        or detector.get("det_size") != [640, 640]
    ):
        raise ValueError("CocoER head-cache detector differs")
    model_tree_sha = str(detector.get("model_tree_sha256", "")).lower()
    if (
        len(model_tree_sha) != 64
        or any(value not in "0123456789abcdef" for value in model_tree_sha)
        or cache.get("detector_model_tree_sha256") != model_tree_sha
    ):
        raise ValueError("CocoER detector model-tree SHA-256 differs")
    detection_source_sha = str(cache.get("source_detection_json_sha256", "")).lower()
    if len(detection_source_sha) != 64 or any(
        value not in "0123456789abcdef" for value in detection_source_sha
    ):
        raise ValueError("CocoER source-detection SHA-256 is malformed")
    entries = cache.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("CocoER head cache is empty")

    expected = {}
    split_counts = Counter()
    image_sizes = {}
    for split in ("train", "val", "test"):
        dataset = EMOTIC(
            str(args.data_root),
            train=split == "train",
            eval_splits=(split,),
            transform=lambda image: image,
            input_mode="full",
        )
        for key, path, body in zip(
            dataset.sample_keys, dataset.file_paths, dataset.body_bboxes
        ):
            flattened_body = body.ravel() if hasattr(body, "ravel") else body
            coordinates = list(flattened_body)[:4]
            if len(coordinates) != 4:
                raise ValueError(f"Invalid EMOTIC body box: {key}")
            expected[key] = (
                Path(path).resolve(),
                [float(value) for value in coordinates],
            )
            split_counts[split] += 1
    missing = sorted(set(expected).difference(entries))
    unexpected = sorted(set(entries).difference(expected))
    if missing or unexpected:
        raise ValueError(
            "CocoER head-cache sample IDs differ: "
            f"missing={missing[:5]} ({len(missing)}), "
            f"unexpected={unexpected[:5]} ({len(unexpected)})"
        )

    for key, (image_path, raw_body) in expected.items():
        if image_path not in image_sizes:
            with Image.open(image_path) as image:
                image_sizes[image_path] = image.size
        width, height = image_sizes[image_path]
        body = [
            max(0.0, min(float(width), raw_body[0])),
            max(0.0, min(float(height), raw_body[1])),
            max(0.0, min(float(width), raw_body[2])),
            max(0.0, min(float(height), raw_body[3])),
        ]
        face = _valid_box(entries[key])
        if face is None or not (
            0 <= face[0] < face[2] <= width and 0 <= face[1] < face[3] <= height
        ):
            raise ValueError(f"CocoER head box is outside its image: {key}")
        middle_x = (face[0] + face[2]) // 2
        if not (
            body[0] <= middle_x <= body[2]
            and face[0] >= body[0]
            and face[2] <= body[2]
        ):
            raise ValueError(f"CocoER head box violates released person matching: {key}")

    payload = {
        "schema_version": 1,
        "method": "CocoER-FT",
        "upstream_commit": COMMIT,
        "assets_valid": True,
        "resnet50": {
            "path": str(args.resnet50_init.resolve()),
            "sha256": _sha(args.resnet50_init),
            "source_file_sha256": source_sha,
            "tensor_shapes": {key: list(value) for key, value in expected_shapes.items()},
        },
        "clip_rn50": {
            "path": str(args.clip_rn50.resolve()),
            "sha256": clip_sha,
        },
        "head_cache": {
            "path": str(args.head_cache.resolve()),
            "sha256": _sha(args.head_cache),
            "samples": len(entries),
            "samples_by_split": dict(sorted(split_counts.items())),
            "unique_images": len(image_sizes),
            "detector": detector,
            "source_detection_json_sha256": detection_source_sha,
        },
        "forbidden_full_emotic_assets_loaded": False,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
