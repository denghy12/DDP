#!/usr/bin/env python3
"""Convert the two official EMOT-Net release Torch7 towers into an audited bundle."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.emot_net_ft.model import (
    EMOTNetBodyEncoder,
    EMOTNetContextEncoder,
)


COMMIT = "69c3a5106aed08121cd12f6a5b359c745136931e"
CONTEXT_NAME = "model_myVDavg_640_Places.t7"
BODY_NAME = "alexnet_features.t7"
RELEASE_ARCHIVE_SHA256 = (
    "ce6096c1af5a3e91badbc06752e2dbbd04fc63f67f24acc95d76a68e1f7e339b"
)
EXPECTED_ASSET_SHA256 = {
    CONTEXT_NAME: "bbf8a09edb1a17338f8004b78cf2e83f3cccc3a7f0bf7c3705368b482cab1e7c",
    BODY_NAME: "0abdbce4910f4c242433d614287448d110a81e7d26562ab291364762cf2dae87",
}
RELEASE_ASSET_MEMBERS = {
    CONTEXT_NAME: f"pretrained_models/{CONTEXT_NAME}",
    BODY_NAME: f"pretrained_models/{BODY_NAME}",
}
EXPECTED_RELEASE_SOURCE_SHA256 = {
    "Steps_for_training.md": "0d21ce72cddfba3db1593a0ba58e78754b1d1492c1b906a0f9b69dc203004bce",
    "codes/OptsEmotionModel.lua": "52be6bf4c07c0f81d3e0917bf039c827eaec46d5bfe5c9ad887c87526cf57526",
    "codes/CreateEmotionModel.lua": "66355d632210f04058ace7a08a112e5443823bd7a1902bf48ef858d53be28ccb",
    "codes/trainTest_BI.lua": "a4982bc8826a5c72070ba2223c92245062a5bc44ac4007c56408cd164886ef2c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typename(value: Any) -> str:
    typename = getattr(value, "_typename", "")
    return typename.decode("utf-8") if isinstance(typename, bytes) else str(typename)


def _children(value: Any) -> List[Any]:
    modules = getattr(value, "modules", None)
    if isinstance(modules, (list, tuple)):
        return list(modules)
    if isinstance(value, Mapping):
        return [
            child
            for key, child in value.items()
            if key not in ("unpack", b"unpack")
        ]
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
) -> Tuple[List[Any], List[Any], Tuple[str, ...]]:
    matches = {}
    for candidate in _objects(root):
        convolutions, batch_norms = _layer_sequence(candidate)
        shapes = tuple(_shape(layer.weight) for layer in convolutions)
        if shapes == tuple(expected_conv_shapes) and len(batch_norms) == expected_bn_count:
            key = tuple(id(layer) for layer in convolutions + batch_norms)
            matches[key] = (convolutions, batch_norms)
    if not matches:
        observed = sorted(
            {
                (len(_layer_sequence(candidate)[0]), len(_layer_sequence(candidate)[1]))
                for candidate in _objects(root)
            }
        )
        raise RuntimeError(
            f"Expected at least one exact {role} tower, found 0; "
            f"observed conv/BN subtree counts={observed}"
        )
    candidates = list(matches.values())
    digests = tuple(_tower_digest(*candidate) for candidate in candidates)
    # The official Torch7 loader unwraps nn.DataParallel with features:get(1).
    # _objects preserves module-list order, so candidate zero is that exact
    # upstream selection. Saved BatchNorm statistics may differ by device;
    # record every replica digest instead of assuming equality.
    return candidates[0][0], candidates[0][1], digests


def _tower_digest(convolutions: Sequence[Any], batch_norms: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for layer in list(convolutions) + list(batch_norms):
        for name in ("weight", "bias", "running_mean", "running_var"):
            value = getattr(layer, name, None)
            if value is None:
                continue
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(name.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(memoryview(array))
    return digest.hexdigest()


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
) -> Mapping[str, Any]:
    target_convs = [layer for layer in module.modules() if isinstance(layer, nn.Conv2d)]
    target_bns = [layer for layer in module.modules() if isinstance(layer, nn.BatchNorm2d)]
    expected_shapes = [tuple(layer.weight.shape) for layer in target_convs]
    source_convs, source_bns, replica_digests = _find_tower(
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
    return {
        "matching_tower_count": len(replica_digests),
        "selected_tower_index": 0,
        "tower_sha256": list(replica_digests),
        "selection_rule": "first_saved_replica_matching_upstream_features_get_1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.release_archive.is_file():
        raise FileNotFoundError(args.release_archive)
    if sha256(args.release_archive) != RELEASE_ARCHIVE_SHA256:
        raise ValueError("EMOT-Net release archive SHA-256 mismatch")
    try:
        import torchfile
    except ImportError as exc:
        raise RuntimeError(
            "Torch7 conversion requires the pure-Python 'torchfile' package"
        ) from exc
    if not hasattr(torchfile, "xrange"):
        torchfile.xrange = range
    with zipfile.ZipFile(args.release_archive) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt EMOT-Net release member: {corrupt_member}")
        names = set(archive.namelist())
        required_members = set(RELEASE_ASSET_MEMBERS.values()) | set(
            EXPECTED_RELEASE_SOURCE_SHA256
        )
        missing = sorted(required_members.difference(names))
        if missing:
            raise ValueError(f"EMOT-Net release archive is missing: {missing}")
        observed_release_sources = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in EXPECTED_RELEASE_SOURCE_SHA256
        }
        if observed_release_sources != EXPECTED_RELEASE_SOURCE_SHA256:
            raise ValueError("EMOT-Net release source member SHA-256 mismatch")
        observed_assets = {
            name: hashlib.sha256(archive.read(member)).hexdigest()
            for name, member in RELEASE_ASSET_MEMBERS.items()
        }
        if observed_assets != EXPECTED_ASSET_SHA256:
            raise ValueError(
                f"EMOT-Net release asset SHA-256 mismatch: {observed_assets}"
            )
        with tempfile.TemporaryDirectory(prefix="emot_net_native_") as temporary:
            temporary_root = Path(temporary)
            extracted = {}
            for name, member in RELEASE_ASSET_MEMBERS.items():
                destination = temporary_root / name
                with destination.open("wb") as stream:
                    stream.write(archive.read(member))
                extracted[name] = destination
            context_source = torchfile.load(
                str(extracted[CONTEXT_NAME]), force_8bytes_long=True
            )
            body_source = torchfile.load(
                str(extracted[BODY_NAME]), force_8bytes_long=True
            )
    context = EMOTNetContextEncoder()
    body = EMOTNetBodyEncoder()
    context_selection = _convert_tower(context_source, context, "context_encoder")
    body_selection = _convert_tower(body_source, body, "body_encoder")
    payload = {
        "schema_version": 2,
        "upstream_repository": "https://github.com/rkosti/emotic",
        "upstream_commit": COMMIT,
        "release_archive_sha256": RELEASE_ARCHIVE_SHA256,
        "native_body_variant": "official_dropbox_alexnet",
        "release_verified_file_sha256": observed_release_sources,
        "tower_selection": {
            "context_encoder": context_selection,
            "body_encoder": body_selection,
        },
        "source_assets": {
            CONTEXT_NAME: observed_assets[CONTEXT_NAME],
            BODY_NAME: observed_assets[BODY_NAME],
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
