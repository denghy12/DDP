#!/usr/bin/env python3
"""Build the universal checkpoint-free benchmark download package."""

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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
REQUIRED_BUNDLE_FILES = (
    "config_resolved.json",
    "run_manifest.json",
    "report.html",
    "train.log",
    "metrics/task_metrics.json",
    "metrics/summary.json",
    "sync_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _reject_unsafe_tree(root: Path, role: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {role} directory: {root}")
    symlinks = sorted(path for path in root.rglob("*") if path.is_symlink())
    if symlinks:
        raise RuntimeError(
            f"{role} must not contain symlinks: "
            + ", ".join(str(path) for path in symlinks[:3])
        )
    checkpoints = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pth"
    )
    if checkpoints:
        raise RuntimeError(
            f"{role} contains forbidden .pth checkpoints: "
            + ", ".join(str(path) for path in checkpoints[:3])
        )


def _validate_sync_manifest(source: Path) -> Mapping[str, Any]:
    for relative in REQUIRED_BUNDLE_FILES:
        required = source / relative
        if not required.is_file():
            raise FileNotFoundError(
                f"Incomplete results_to_sync bundle; missing {required}"
            )
    _reject_unsafe_tree(source, "results_to_sync bundle")
    sync_manifest = _read_json(source / "sync_manifest.json")
    if sync_manifest.get("contains_pth") is not False:
        raise ValueError("sync_manifest must declare contains_pth=false")
    records = sync_manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("sync_manifest files must be a non-empty list")
    listed_paths = set()
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("sync_manifest file records must be objects")
        relative = PurePosixPath(str(raw_record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe sync_manifest path: {relative}")
        path = source.joinpath(*relative.parts)
        if relative.as_posix() in listed_paths:
            raise ValueError(f"Duplicate sync_manifest path: {relative}")
        listed_paths.add(relative.as_posix())
        if not path.is_file():
            raise FileNotFoundError(f"sync_manifest file is missing: {path}")
        if path.suffix.lower() == ".pth":
            raise RuntimeError(f"sync_manifest lists forbidden checkpoint: {path}")
        if int(raw_record.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"sync_manifest byte count differs: {path}")
        if str(raw_record.get("sha256", "")) != sha256(path):
            raise ValueError(f"sync_manifest SHA-256 differs: {path}")
    actual_paths = {
        path.relative_to(source).as_posix()
        for path in _files(source)
        if path.name != "sync_manifest.json"
    }
    if listed_paths != actual_paths:
        missing_from_manifest = sorted(actual_paths - listed_paths)
        missing_from_bundle = sorted(listed_paths - actual_paths)
        raise ValueError(
            "sync_manifest file set differs from bundle: "
            f"unlisted={missing_from_manifest[:3]}, "
            f"missing={missing_from_bundle[:3]}"
        )

    config = _read_json(source / "config_resolved.json")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("config_resolved protocol must be an object")
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("config_resolved protocol tasks must be non-empty")
    scores = sorted((source / "scores").glob("task*_scores.pt"))
    if len(scores) != len(tasks):
        raise RuntimeError(
            f"Expected {len(tasks)} canonical score files, found {len(scores)}"
        )
    return sync_manifest


def _copy_extra(source: Path, destination: Path, role: str) -> None:
    if source.is_symlink():
        raise RuntimeError(f"{role} must not be a symlink: {source}")
    if not source.exists():
        raise FileNotFoundError(f"Missing {role}: {source}")
    if source.is_file():
        if source.suffix.lower() == ".pth":
            raise RuntimeError(f"{role} is a forbidden .pth checkpoint: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    _reject_unsafe_tree(source, role)
    shutil.copytree(source, destination)


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    unique: Dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        unique[str(resolved)] = resolved
    return [unique[key] for key in sorted(unique)]


def build_download_package(
    *,
    run_root: Path,
    run_id: str,
    expected_bundles: int,
    launcher_logs: Sequence[Path] = (),
    extras: Sequence[Path] = (),
) -> Dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run-id contains unsafe characters")
    if expected_bundles <= 0:
        raise ValueError("expected-bundles must be positive")
    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing run root: {root}")
    sources = sorted(
        root.glob(f"benchmarks/*/*/*/seed*/results_to_sync/{run_id}")
    )
    if len(sources) != expected_bundles:
        raise RuntimeError(
            f"Expected {expected_bundles} result bundles, found {len(sources)}"
        )

    download_base = root / "download_ready"
    package_base = root / "download_packages"
    destination = download_base / run_id
    archive = package_base / f"{run_id}.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    for path in (destination, archive, checksum):
        if path.exists():
            raise FileExistsError(f"Download output already exists: {path}")
    temporary = download_base / f".{run_id}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Temporary download path exists: {temporary}")
    temporary.mkdir(parents=True)

    bundle_records: List[Dict[str, Any]] = []
    try:
        for source in sources:
            sync_manifest = _validate_sync_manifest(source)
            seed_name = source.parents[1].name
            method_name = source.parents[2].name
            target = temporary / "bundles" / method_name / seed_name
            shutil.copytree(source, target)
            run_manifest = _read_json(source / "run_manifest.json")
            bundle_records.append(
                {
                    "method": method_name,
                    "seed": seed_name,
                    "relative_path": target.relative_to(temporary).as_posix(),
                    "git_commit": run_manifest.get("git_commit"),
                    "git_dirty": run_manifest.get("git_dirty"),
                    "reporting_split": run_manifest.get("reporting_split"),
                    "configuration_locked": run_manifest.get(
                        "configuration_locked"
                    ),
                    "sync_file_count": len(sync_manifest["files"]),
                }
            )

        automatic_logs = [
            root.parent / "_launcher_logs" / f"{run_id}.log",
            root.parent / "_launcher_logs" / f"{run_id}_preflight.log",
        ]
        all_logs = _unique_paths(
            list(launcher_logs)
            + [path for path in automatic_logs if path.is_file()]
        )
        used_log_names = set()
        for source in all_logs:
            if source.name in used_log_names:
                raise ValueError(f"Duplicate launcher log name: {source.name}")
            used_log_names.add(source.name)
            _copy_extra(source, temporary / "logs" / source.name, "launcher log")

        used_extra_names = set()
        for source in _unique_paths(extras):
            if source.name in used_extra_names:
                raise ValueError(f"Duplicate extra artifact name: {source.name}")
            used_extra_names.add(source.name)
            _copy_extra(
                source,
                temporary / "extras" / source.name,
                "extra artifact",
            )

        readme = temporary / "DOWNLOAD_INFO.txt"
        readme.write_text(
            "EMOTIC MLCIL checkpoint-free download package\n"
            f"run_id: {run_id}\n"
            "Canonical .pt score tensors are included for metric verification.\n"
            "All .pth checkpoint files are explicitly excluded.\n"
            "Keep the adjacent .tar.gz.sha256 file when transferring this archive.\n",
            encoding="utf-8",
        )
        _reject_unsafe_tree(temporary, "assembled download package")
        files_before_manifest = _files(temporary)
        manifest = {
            "download_schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "bundle_count": len(bundle_records),
            "bundles": bundle_records,
            "launcher_logs": [f"logs/{path.name}" for path in all_logs],
            "extras": [f"extras/{path.name}" for path in _unique_paths(extras)],
            "contains_pth": False,
            "excluded_patterns": ["*.pth"],
            "canonical_scores_included": True,
            "file_count_excluding_manifest": len(files_before_manifest),
            "total_bytes_excluding_manifest": sum(
                path.stat().st_size for path in files_before_manifest
            ),
            "files": [
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files_before_manifest
            ],
        }
        (temporary / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        download_base.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        package_base.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(destination, arcname=run_id, recursive=True)
        with tarfile.open(archive, "r:gz") as stream:
            forbidden_members = [
                member.name
                for member in stream.getmembers()
                if PurePosixPath(member.name).suffix.lower() == ".pth"
            ]
        if forbidden_members:
            raise RuntimeError(
                "Archive unexpectedly contains .pth checkpoints: "
                + ", ".join(forbidden_members[:3])
            )
        archive_sha = sha256(archive)
        checksum.write_text(
            f"{archive_sha}  {archive.name}\n",
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "download_directory": str(destination),
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
            "checksum_file": str(checksum),
            "bundle_count": len(bundle_records),
            "launcher_log_count": len(all_logs),
            "contains_pth": False,
        }
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        # All three paths were required not to exist before packaging. Remove
        # only outputs created by this failed invocation so a packaging-only
        # retry never requires retraining or manual cleanup.
        if destination.is_dir():
            shutil.rmtree(destination)
        for partial in (archive, checksum):
            if partial.is_file():
                partial.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-bundles", type=int, required=True)
    parser.add_argument("--launcher-log", type=Path, action="append", default=[])
    parser.add_argument("--extra", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_download_package(
        run_root=args.run_root,
        run_id=args.run_id,
        expected_bundles=args.expected_bundles,
        launcher_logs=args.launcher_log,
        extras=args.extra,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
