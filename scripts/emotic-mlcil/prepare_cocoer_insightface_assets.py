#!/usr/bin/env python3
"""Verify and safely unpack the fixed InsightFace buffalo_l release asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path


ARCHIVE_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    "buffalo_l.zip"
)
ARCHIVE_BYTES = 288_621_354
ARCHIVE_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
MODEL_TREE_SHA256 = "50fa1383e97d137f2902b53de7b7305ffbd35eb4ae32135d95d1e25d5a9d9d3d"
MODEL_FILES = {
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha(path)))
    return digest.hexdigest()


def _verify_model_tree(model_root: Path) -> None:
    observed_names = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
    }
    if observed_names != set(MODEL_FILES):
        raise ValueError(
            "InsightFace buffalo_l file set differs: "
            f"observed={sorted(observed_names)}"
        )
    for name, expected in MODEL_FILES.items():
        observed = _sha(model_root / name)
        if observed != expected:
            raise ValueError(f"InsightFace buffalo_l file SHA differs: {name}")
    if _tree_sha256(model_root) != MODEL_TREE_SHA256:
        raise ValueError("InsightFace buffalo_l model-tree SHA differs")


def _extract(archive_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".cocoer_insightface.", dir=str(destination.parent))
    )
    try:
        model_root = stage / "models" / "buffalo_l"
        model_root.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if (
                len(infos) != len(MODEL_FILES)
                or {item.filename for item in infos} != set(MODEL_FILES)
            ):
                raise ValueError("InsightFace buffalo_l ZIP member set differs")
            for item in infos:
                mode = item.external_attr >> 16
                if item.is_dir() or stat.S_ISLNK(mode):
                    raise ValueError("InsightFace buffalo_l ZIP contains an unsafe member")
                target = model_root / item.filename
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        _verify_model_tree(model_root)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    archive_path = args.archive.expanduser().resolve()
    destination = args.output_root.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if archive_path.stat().st_size != ARCHIVE_BYTES or _sha(archive_path) != ARCHIVE_SHA256:
        raise ValueError("InsightFace buffalo_l archive size or SHA-256 differs")

    model_root = destination / "models" / "buffalo_l"
    if destination.exists():
        _verify_model_tree(model_root)
    else:
        _extract(archive_path, destination)

    manifest = {
        "schema_version": 1,
        "asset_kind": "cocoer_insightface_buffalo_l_v0.7",
        "use": "CocoER head-box preprocessing only",
        "archive_url": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "model_tree_sha256": MODEL_TREE_SHA256,
        "model_file_sha256": dict(MODEL_FILES),
        "license_scope": "InsightFace model weights: non-commercial research only",
        "output_root": str(destination),
    }
    manifest_path = destination / "COCOER_ASSET_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
