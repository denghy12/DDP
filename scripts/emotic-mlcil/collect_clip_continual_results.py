#!/usr/bin/env python
"""Collect all checkpoint-free method bundles into one download directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-bundles", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(args.run_output_root).resolve()
    if args.expected_bundles <= 0:
        raise ValueError("expected-bundles must be positive")
    sources = sorted(
        root.glob(
            f"benchmarks/*/*/*/seed*/results_to_sync/{args.run_id}"
        )
    )
    if len(sources) != args.expected_bundles:
        raise RuntimeError(
            f"Expected {args.expected_bundles} result bundles, found "
            f"{len(sources)}"
        )

    destination = root / "download_ready" / args.run_id
    if destination.exists():
        raise FileExistsError(
            f"Download directory already exists: {destination}"
        )
    records = []
    for source in sources:
        seed_name = source.parents[1].name
        method_name = source.parents[2].name
        target = destination / method_name / seed_name
        shutil.copytree(source, target)
        records.append(
            {
                "method": method_name,
                "seed": seed_name,
                "relative_path": str(target.relative_to(destination)),
            }
        )

    checkpoint_files = list(destination.rglob("*.pth"))
    if checkpoint_files:
        raise RuntimeError("Download directory unexpectedly contains checkpoints")
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    payload = {
        "run_id": args.run_id,
        "bundles": records,
        "bundle_count": len(records),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "contains_pth": False,
        "files": [
            {
                "path": str(path.relative_to(destination)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "download_ready": str(destination),
                "bundle_count": len(records),
                "contains_pth": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
