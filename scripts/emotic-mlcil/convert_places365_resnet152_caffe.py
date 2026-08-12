#!/usr/bin/env python3
"""Convert and audit the official ResNet-152 Places365 Caffe pool5 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft.places365 import (
    CaffeBatchNormScale, CaffeResNet152Places365,
    PLACES365_CAFFE_MODEL_SHA256, PLACES365_CAFFE_MODEL_URL,
    PLACES365_PROTOTXT_SHA256, PLACES365_REPOSITORY,
    PLACES365_REPOSITORY_COMMIT, iter_caffe_blocks,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer(net: Any, name: str) -> Any:
    identifier = net.getLayerId(name)
    if identifier <= 0:
        raise ValueError(f"Caffe layer is missing: {name}")
    return net.getLayer(identifier)


def _copy_conv(net: Any, name: str, target: torch.nn.Conv2d) -> None:
    blobs = _layer(net, name).blobs
    if len(blobs) != 1:
        raise ValueError(f"Unexpected convolution blobs: {name}")
    value = torch.from_numpy(np.asarray(blobs[0])).to(target.weight)
    if value.shape != target.weight.shape:
        raise ValueError(f"Convolution shape differs: {name}")
    target.weight.data.copy_(value)


def _copy_bn_scale(net: Any, stem: str, target: CaffeBatchNormScale) -> None:
    # The first layer is named bn_conv1/scale_conv1, while all residual
    # branches omit the separator (for example bn2a_branch2a).
    suffix = f"_{stem}" if stem == "conv1" else stem
    batch_norm = _layer(net, f"bn{suffix}").blobs
    scale = _layer(net, f"scale{suffix}").blobs
    if len(batch_norm) != 3 or len(scale) != 2:
        raise ValueError(f"Unexpected BatchNorm/Scale blobs: {stem}")
    factor = float(np.asarray(batch_norm[2]).reshape(-1)[0])
    inverse_factor = 0.0 if factor == 0.0 else 1.0 / factor
    values = (
        np.asarray(batch_norm[0]).reshape(-1) * inverse_factor,
        np.asarray(batch_norm[1]).reshape(-1) * inverse_factor,
        np.asarray(scale[0]).reshape(-1), np.asarray(scale[1]).reshape(-1),
    )
    for value, destination in zip(
        values, (target.mean, target.variance, target.weight, target.bias)
    ):
        tensor = torch.from_numpy(value).to(destination)
        if tensor.shape != destination.shape:
            raise ValueError(f"BatchNorm/Scale shape differs: {stem}")
        destination.data.copy_(tensor)


def convert(net: Any) -> CaffeResNet152Places365:
    model = CaffeResNet152Places365().eval()
    _copy_conv(net, "conv1", model.conv1)
    _copy_bn_scale(net, "conv1", model.bn1)
    for stage, block_name, block in iter_caffe_blocks(model):
        stem = f"{stage}{block_name}"
        if block.projection is not None and block.projection_bn is not None:
            _copy_conv(net, f"res{stem}_branch1", block.projection)
            _copy_bn_scale(net, f"{stem}_branch1", block.projection_bn)
        for branch, conv, batch_norm in (
            ("2a", block.conv1, block.bn1),
            ("2b", block.conv2, block.bn2),
            ("2c", block.conv3, block.bn3),
        ):
            _copy_conv(net, f"res{stem}_branch{branch}", conv)
            _copy_bn_scale(net, f"{stem}_branch{branch}", batch_norm)
    return model


def _audit(net: Any, model: CaffeResNet152Places365) -> Dict[str, float]:
    values = np.random.RandomState(0).normal(
        0.0, 32.0, size=(1, 3, 224, 224)
    ).astype(np.float32)
    net.setInput(values)
    expected = np.asarray(net.forward("pool5")).reshape(1, 2048)
    with torch.no_grad():
        observed = model(torch.from_numpy(values)).cpu().numpy()
    difference = np.abs(expected - observed)
    denominator = np.maximum(np.abs(expected), 1.0e-8)
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_relative_error": float((difference / denominator).max()),
        "mean_relative_error": float((difference / denominator).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototxt", required=True, type=Path)
    parser.add_argument("--caffemodel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    observed = {"prototxt": _sha256(args.prototxt),
                "caffemodel": _sha256(args.caffemodel)}
    expected = {"prototxt": PLACES365_PROTOTXT_SHA256,
                "caffemodel": PLACES365_CAFFE_MODEL_SHA256}
    if observed != expected:
        raise ValueError(f"Official Places365 source hash differs: {observed}")
    import cv2
    net = cv2.dnn.readNetFromCaffe(str(args.prototxt), str(args.caffemodel))
    model = convert(net)
    audit = _audit(net, model)
    if audit["max_abs_error"] > 2.0e-3 or audit["mean_abs_error"] > 1.0e-4:
        raise ValueError(f"Caffe/PyTorch pool5 equivalence failed: {audit}")
    payload = {
        "schema_version": 1,
        "resource": "ResNet-152 Places365 Caffe-to-PyTorch pool5 encoder",
        "architecture": "official Caffe ResNet-152-v1 (pool1 padding=0)",
        "source_repository": PLACES365_REPOSITORY,
        "source_repository_commit": PLACES365_REPOSITORY_COMMIT,
        "source_model_url": PLACES365_CAFFE_MODEL_URL,
        "source_caffemodel_sha256": observed["caffemodel"],
        "source_prototxt_sha256": observed["prototxt"],
        "conversion": "direct OpenCV-DNN blob mapping with pool5 audit",
        "operator_equivalence": audit,
        "state_dict": model.state_dict(),
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    result = {key: value for key, value in payload.items() if key != "state_dict"}
    result.update(output=str(destination), output_sha256=_sha256(destination))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
