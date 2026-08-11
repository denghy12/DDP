#!/usr/bin/env python3
"""Package checkpoint-free CocoER head preprocessing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence


PACKAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
FORBIDDEN_SUFFIXES = {".pth", ".onnx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path, role: str) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Missing regular {role}: {source}")
    if source.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"Forbidden {role} payload: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_package(
    *,
    output_base: Path,
    package_name: str,
    detections: Path,
    head_cache: Path,
    asset_audit: Path,
    native_manifest: Path,
    logs: Sequence[Path] = (),
) -> Dict[str, object]:
    if not PACKAGE_PATTERN.fullmatch(package_name):
        raise ValueError("package-name contains unsafe characters")
    output_base = output_base.resolve()
    ready_base = output_base / "download_ready"
    package_base = output_base / "download_packages"
    destination = ready_base / package_name
    archive = package_base / f"{package_name}.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    for output in (destination, archive, checksum):
        if output.exists():
            raise FileExistsError(f"Download output already exists: {output}")
    temporary = ready_base / f".{package_name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    try:
        artifacts = (
            (detections, "cocoer_head_detections.json"),
            (head_cache, "emotic_head_boxes_v0.1.json"),
            (asset_audit, "cocoer_assets_audit.json"),
            (native_manifest, "native_assets_manifest.json"),
        )
        for source, name in artifacts:
            _copy(source.resolve(), temporary / "artifacts" / name, name)
        used_log_names = set()
        for source in sorted({path.resolve() for path in logs}):
            if source.name in used_log_names:
                raise ValueError(f"Duplicate log name: {source.name}")
            used_log_names.add(source.name)
            _copy(source, temporary / "logs" / source.name, "log")

        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        records: List[Dict[str, object]] = []
        for path in files:
            relative = path.relative_to(temporary).as_posix()
            if PurePosixPath(relative).suffix.lower() in FORBIDDEN_SUFFIXES:
                raise RuntimeError(f"Forbidden packaged payload: {relative}")
            records.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        manifest = {
            "download_schema_version": 1,
            "artifact_kind": "cocoer_head_preprocess",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_name": package_name,
            "contains_pth": False,
            "contains_onnx": False,
            "excluded_patterns": ["*.pth", "*.onnx"],
            "paths_are_relative": True,
            "manifest_self_included": False,
            "file_count_excluding_manifest": len(records),
            "total_bytes_excluding_manifest": sum(
                int(record["bytes"]) for record in records
            ),
            "files": records,
        }
        (temporary / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ready_base.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        package_base.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(destination, arcname=package_name, recursive=True)
        with tarfile.open(archive, "r:gz") as stream:
            forbidden = [
                member.name
                for member in stream.getmembers()
                if PurePosixPath(member.name).suffix.lower()
                in FORBIDDEN_SUFFIXES
            ]
        if forbidden:
            raise RuntimeError("Archive contains forbidden payloads")
        archive_sha256 = sha256(archive)
        checksum.write_text(
            f"{archive_sha256}  {archive.name}\n", encoding="utf-8"
        )
        return {
            "package_name": package_name,
            "download_directory": str(destination),
            "archive": str(archive),
            "archive_sha256": archive_sha256,
            "checksum_file": str(checksum),
            "file_count_excluding_manifest": len(records),
            "contains_pth": False,
            "contains_onnx": False,
        }
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        if destination.is_dir():
            shutil.rmtree(destination)
        for output in (archive, checksum):
            if output.is_file():
                output.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", required=True, type=Path)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--head-cache", required=True, type=Path)
    parser.add_argument("--asset-audit", required=True, type=Path)
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--log", action="append", type=Path, default=[])
    args = parser.parse_args()
    result = build_package(
        output_base=args.output_base,
        package_name=args.package_name,
        detections=args.detections,
        head_cache=args.head_cache,
        asset_audit=args.asset_audit,
        native_manifest=args.native_manifest,
        logs=args.log,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
