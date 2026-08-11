#!/usr/bin/env python3
"""Convert the two official EMOT-Net Torch7 towers into an audited bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.emot_net_ft.model import (
    EMOTNetBodyEncoder,
    EMOTNetContextEncoder,
)


COMMIT = "69c3a5106aed08121cd12f6a5b359c745136931e"
CONTEXT_NAME = "model_myVDavg_640_Places.t7"
BODY_NAME = "myVD_ImgNet_66_old.t7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typename(value: Any) -> str:
    return str(getattr(value, "_typename", ""))


def _children(value: Any) -> List[Any]:
    modules = getattr(value, "modules", None)
    if isinstance(modules, (list, tuple)):
        return list(modules)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _objects(root: Any, seen: Any = None) -> Iterable[Any]:
    visited = set() if seen is None else seen
    identity = id(root)
    if identity in visited:
        return
    visited.add(identity)
    yield root
    for child in _children(root):
        yield from _objects(child, visited)


def _layer_sequence(root: Any) -> Tuple[List[Any], List[Any]]:
    objects = list(_objects(root))
    convolutions = [
        value for value in objects if _typename(value).endswith("SpatialConvolution")
    ]
    batch_norms = [
        value
        for value in objects
        if _typename(value).endswith("SpatialBatchNormalization")
    ]
    return convolutions, batch_norms


def _shape(value: Any) -> Tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(value).shape)


def _find_tower(
    root: Any,
    expected_conv_shapes: Sequence[Tuple[int, ...]],
    expected_bn_count: int,
    role: str,
) -> Tuple[List[Any], List[Any]]:
    matches = {}
    for candidate in _objects(root):
        convolutions, batch_norms = _layer_sequence(candidate)
        shapes = tuple(_shape(layer.weight) for layer in convolutions)
        if shapes == tuple(expected_conv_shapes) and len(batch_norms) == expected_bn_count:
            key = tuple(id(layer) for layer in convolutions + batch_norms)
            matches[key] = (convolutions, batch_norms)
    if len(matches) != 1:
        observed = sorted(
            {
                (len(_layer_sequence(candidate)[0]), len(_layer_sequence(candidate)[1]))
                for candidate in _objects(root)
            }
        )
        raise RuntimeError(
            f"Expected one exact {role} tower, found {len(matches)}; "
            f"observed conv/BN subtree counts={observed}"
        )
    return next(iter(matches.values()))


def _copy_tensor(destination: torch.Tensor, source: Any, name: str) -> None:
    value = torch.from_numpy(np.asarray(source)).to(dtype=destination.dtype)
    if value.shape != destination.shape:
        raise ValueError(
            f"Converted {name} shape {tuple(value.shape)} != {tuple(destination.shape)}"
        )
    destination.copy_(value)


def _convert_tower(
    torch7_root: Any,
    module: nn.Module,
    role: str,
) -> None:
    target_convs = [layer for layer in module.modules() if isinstance(layer, nn.Conv2d)]
    target_bns = [layer for layer in module.modules() if isinstance(layer, nn.BatchNorm2d)]
    expected_shapes = [tuple(layer.weight.shape) for layer in target_convs]
    source_convs, source_bns = _find_tower(
        torch7_root, expected_shapes, len(target_bns), role
    )
    with torch.no_grad():
        for index, (source, target) in enumerate(zip(source_convs, target_convs)):
            _copy_tensor(target.weight, source.weight, f"{role}.conv{index}.weight")
            if target.bias is not None:
                _copy_tensor(target.bias, source.bias, f"{role}.conv{index}.bias")
        for index, (source, target) in enumerate(zip(source_bns, target_bns)):
            _copy_tensor(target.weight, source.weight, f"{role}.bn{index}.weight")
            _copy_tensor(target.bias, source.bias, f"{role}.bn{index}.bias")
            _copy_tensor(
                target.running_mean,
                source.running_mean,
                f"{role}.bn{index}.running_mean",
            )
            _copy_tensor(
                target.running_var,
                source.running_var,
                f"{role}.bn{index}.running_var",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-t7", required=True, type=Path)
    parser.add_argument("--body-t7", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.context_t7.name != CONTEXT_NAME or args.body_t7.name != BODY_NAME:
        raise ValueError("Input filenames do not match the official EMOT-Net defaults")
    for path in (args.context_t7, args.body_t7):
        if not path.is_file():
            raise FileNotFoundError(path)
    try:
        import torchfile
    except ImportError as exc:
        raise RuntimeError(
            "Torch7 conversion requires the pure-Python 'torchfile' package"
        ) from exc

    context_source = torchfile.load(str(args.context_t7), force_8bytes_long=True)
    body_source = torchfile.load(str(args.body_t7), force_8bytes_long=True)
    context = EMOTNetContextEncoder()
    body = EMOTNetBodyEncoder()
    _convert_tower(context_source, context, "context_encoder")
    _convert_tower(body_source, body, "body_encoder")
    payload = {
        "schema_version": 1,
        "upstream_repository": "https://github.com/rkosti/emotic",
        "upstream_commit": COMMIT,
        "source_assets": {
            CONTEXT_NAME: sha256(args.context_t7),
            BODY_NAME: sha256(args.body_t7),
        },
        "context_encoder": context.state_dict(),
        "body_encoder": body.state_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(payload, args.output)
    print(
        {
            "output": str(args.output),
            "sha256": sha256(args.output),
            "context_asset_sha256": payload["source_assets"][CONTEXT_NAME],
            "body_asset_sha256": payload["source_assets"][BODY_NAME],
        }
    )


if __name__ == "__main__":
    main()
