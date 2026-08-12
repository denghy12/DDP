#!/usr/bin/env python3
"""Build the protocol-safe Task-0 CCIM context dictionary.

This is an offline model-resource preparation step.  It reads only training
samples accessible at Task 0, masks the annotated target person, extracts the
official 2048-D ResNet152-Places365 descriptor, and runs deterministic
K-Means++ with the paper's EMOTIC dictionary size (256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.data_module import EMOTICMLCILDataModule
from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft.method import (
    CCIM_DICTIONARY_SCOPE,
    CCIM_FEATURE_EXTRACTOR,
)
from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft.model import (
    CCIM_SOURCE_SHA256,
    CCIM_UPSTREAM_COMMIT,
    CCIM_UPSTREAM_REPOSITORY,
)
from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft.places365 import (
    CaffeResNet152Places365,
)
from benchmarks.emotic_mlcil.protocol import load_protocol
from src.helper_functions.emotic_loader import EMOTIC


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Unable to inspect CCIM source")
    return result.stdout.strip()


def _validate_ccim_source(source_root: Path) -> None:
    if _git_output(source_root, "rev-parse", "HEAD") != CCIM_UPSTREAM_COMMIT:
        raise ValueError("CCIM source checkout is not at the registered commit")
    if _git_output(source_root, "status", "--porcelain"):
        raise ValueError("CCIM source checkout must be clean")
    source = source_root / "CCIM.py"
    if not source.is_file() or _sha256(source) != CCIM_SOURCE_SHA256:
        raise ValueError("CCIM.py differs from the registered immutable source")


def _load_places365_encoder(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = CaffeResNet152Places365()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Places365 checkpoint is not the audited Caffe conversion")
    if payload.get("resource") != "ResNet-152 Places365 Caffe-to-PyTorch pool5 encoder":
        raise ValueError("Places365 checkpoint resource identity differs")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().to(device)


def _validate_places365_asset(checkpoint_path: Path) -> Tuple[str, str]:
    expected_sha = os.environ.get("CCIM_PLACES365_EXPECTED_SHA256", "").strip().lower()
    source = os.environ.get("CCIM_PLACES365_SOURCE", "").strip()
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise RuntimeError(
            "Set CCIM_PLACES365_EXPECTED_SHA256 to the independently verified "
            "64-character SHA-256 of the converted ResNet152-Places365 checkpoint"
        )
    actual_sha = _sha256(checkpoint_path)
    if actual_sha != expected_sha:
        raise ValueError("ResNet152-Places365 checkpoint SHA-256 differs")
    if not source:
        raise RuntimeError(
            "Set CCIM_PLACES365_SOURCE to the official asset URL and conversion identity"
        )
    return actual_sha, source


class _MaskedContextDataset(Dataset):
    def __init__(
        self,
        source: EMOTIC,
        indices: Sequence[int],
        sample_ids: Sequence[str],
    ) -> None:
        from torchvision import transforms

        self.source = source
        self.indices = tuple(int(value) for value in indices)
        self.sample_ids = tuple(str(sample_ids[value]) for value in self.indices)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        index = self.indices[item]
        image = Image.open(self.source.file_paths[index]).convert("RGB")
        values = np.asarray(self.source.body_bboxes[index], dtype=np.float32).ravel()
        if values.size >= 4 and np.isfinite(values[:4]).all():
            width, height = image.size
            x1 = max(0, min(width, int(np.floor(values[0]))))
            y1 = max(0, min(height, int(np.floor(values[1]))))
            x2 = max(0, min(width, int(np.ceil(values[2]))))
            y2 = max(0, min(height, int(np.ceil(values[3]))))
            if x2 > x1 and y2 > y1:
                image = image.copy()
                ImageDraw.Draw(image).rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
        values = self.transform(image).mul(255.0)
        values = values[[2, 1, 0], :, :]
        values -= values.new_tensor((104.0, 117.0, 123.0)).view(3, 1, 1)
        return values, self.sample_ids[item]


def _extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    features, sample_ids = [], []
    with torch.no_grad():
        for images, identifiers in loader:
            values = model(images.to(device, non_blocking=True).float())
            if values.ndim != 2 or values.shape[1] != 2048:
                raise ValueError("Places365 encoder did not return 2048-D descriptors")
            if not torch.isfinite(values).all():
                raise ValueError("Places365 descriptors contain non-finite values")
            features.append(values.float().cpu())
            sample_ids.extend(str(value) for value in identifiers)
    if not features:
        raise ValueError("Task-0 dictionary dataset is empty")
    return torch.cat(features).numpy(), tuple(sample_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--places365-checkpoint", required=True)
    parser.add_argument("--ccim-source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Batch size and worker settings are invalid")

    protocol = load_protocol(args.protocol).with_seed(0)
    if protocol.track != "B":
        raise ValueError("EMOT-Net+CCIM dictionary preparation requires Track B")
    source_root = Path(args.ccim_source_root).expanduser().resolve()
    _validate_ccim_source(source_root)
    checkpoint = Path(args.places365_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha, checkpoint_source = _validate_places365_asset(checkpoint)

    # Construct only metadata here; raw labels are used solely for the same
    # current-class membership predicate enforced by EMOTICMLCILDataModule.
    source = EMOTIC(
        args.data_root,
        train=True,
        transform=lambda image: image,
        input_mode="full",
        class_names=protocol.class_order,
    )
    current = protocol.current_class_indices(0)
    indices = [
        index
        for index, target in enumerate(source.targets)
        if EMOTICMLCILDataModule._intersects(target, current)
    ]
    all_ids = EMOTICMLCILDataModule._build_sample_ids(source, protocol.train_split)
    dataset = _MaskedContextDataset(source, indices, all_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=False,
    )
    device = torch.device(args.device)
    model = _load_places365_encoder(checkpoint, device)
    features, sample_ids = _extract_features(model, loader, device)
    if len(features) < 256:
        raise ValueError("Task-0-accessible training data has fewer than 256 samples")

    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for source K-Means++") from exc
    clustering = KMeans(
        n_clusters=256,
        init="k-means++",
        n_init=10,
        random_state=0,
        algorithm="lloyd",
    ).fit(features)
    counts = np.bincount(clustering.labels_, minlength=256).astype(np.float32)
    prior = counts / counts.sum()
    payload = {
        "schema_version": 1,
        "resource": "EMOT-Net+CCIM confounder dictionary",
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "class_order_hash": protocol.class_order_hash,
        "task_id": 0,
        "scope": CCIM_DICTIONARY_SCOPE,
        "strategy": "dp_cause",
        "feature_extractor": CCIM_FEATURE_EXTRACTOR,
        "feature_checkpoint_path": str(checkpoint),
        "feature_checkpoint_source": checkpoint_source,
        "feature_checkpoint_sha256": checkpoint_sha,
        "dictionary_size": 256,
        "confounder_dim": 2048,
        "construction_seed": 0,
        "sample_count": len(sample_ids),
        "sample_ids_sha256": _canonical_hash(sample_ids),
        "mask_fill_rgb": [0, 0, 0],
        "preprocessing": (
            "Resize(256), CenterCrop(224), RGB-to-BGR, 0-255 scale, "
            "subtract original Caffe ResNet BGR means [104, 117, 123]"
        ),
        "kmeans": {
            "implementation": "sklearn.cluster.KMeans",
            "sklearn_version": sklearn_version,
            "init": "k-means++",
            "n_init": 10,
            "algorithm": "lloyd",
            "random_state": 0,
            "inertia": float(clustering.inertia_),
            "iterations": int(clustering.n_iter_),
            "cluster_counts": counts.astype(np.int64).tolist(),
        },
        "ccim_repository": CCIM_UPSTREAM_REPOSITORY,
        "ccim_commit": CCIM_UPSTREAM_COMMIT,
        "ccim_source_sha256": CCIM_SOURCE_SHA256,
        "dictionary": torch.from_numpy(clustering.cluster_centers_).float(),
        "prior": torch.from_numpy(prior).float().unsqueeze(1),
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"dictionary", "prior"}
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"output={destination}")
    print(f"sha256={_sha256(destination)}")


if __name__ == "__main__":
    main()
