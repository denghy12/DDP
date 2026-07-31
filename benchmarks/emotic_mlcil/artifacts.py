"""Stable artifact layout and schema validation."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
from html import escape
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import torch

from .protocol import BenchmarkProtocol
from .types import BenchmarkSummary, PredictionOutput, TaskMetrics


RUN_MANIFEST_REQUIRED_FIELDS = (
    "method",
    "method_family",
    "protocol_id",
    "track",
    "seed",
    "git_commit",
    "base_commit",
    "backbone",
    "class_order_hash",
    "protocol_hash",
    "data_split_hash",
    "threshold_policy",
    "replay_memory_samples",
    "replay_memory_bytes",
    "total_parameters",
    "trainable_parameters",
    "incremental_parameters",
    "upstream_repository",
    "upstream_commit",
    "test_labels_used_for_selection",
)

MAIN_TABLE_FIELDS = (
    "method",
    "type",
    "replay_memory",
    "backbone",
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_mAP",
    "forgetting",
    "parameter_growth",
)

SHARD_SCHEMA_VERSION = 1
_SHARD_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [key for key in RUN_MANIFEST_REQUIRED_FIELDS if key not in manifest]
    if missing:
        raise ValueError(
            "run_manifest is missing required fields: " + ", ".join(missing)
        )
    null_fields = [
        key for key in RUN_MANIFEST_REQUIRED_FIELDS if manifest.get(key) is None
    ]
    if null_fields:
        raise ValueError(
            "run_manifest required fields cannot be null: " + ", ".join(null_fields)
        )
    if manifest["test_labels_used_for_selection"] is not False:
        raise ValueError("test_labels_used_for_selection must be false")
    if str(manifest["track"]) not in {"A", "B"}:
        raise ValueError("run_manifest track must be A or B")
    if not isinstance(manifest["seed"], int) or manifest["seed"] < 0:
        raise ValueError("run_manifest seed must be a non-negative integer")
    threshold_policy = manifest["threshold_policy"]
    if not isinstance(threshold_policy, dict):
        raise ValueError("threshold_policy must be a mapping")
    if (
        threshold_policy.get("kind") != "fixed_global"
        or float(threshold_policy.get("value", -1)) != 0.5
    ):
        raise ValueError("run_manifest must record fixed_global threshold 0.5")
    for key in (
        "replay_memory_samples",
        "replay_memory_bytes",
        "total_parameters",
        "trainable_parameters",
        "incremental_parameters",
    ):
        value = manifest[key]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"run_manifest {key} must be a non-negative integer")


class ArtifactStore:
    def __init__(
        self,
        output_root: Union[str, Path],
        protocol: BenchmarkProtocol,
        track: str,
        method_name: str,
        seed: int,
    ) -> None:
        if track not in {"A", "B"}:
            raise ValueError("track must be A or B")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.protocol = protocol
        self.track = track
        self.method_name = method_name
        self.seed = int(seed)
        self.root = (
            Path(output_root)
            / "benchmarks"
            / protocol.protocol_id
            / track
            / method_name
            / f"seed{seed}"
        )
        self.checkpoint_dir = self.root / "checkpoints"
        self.score_dir = self.root / "scores"
        self.metrics_dir = self.root / "metrics"
        self.shard_dir = self.root / "shards"

    def initialize(
        self,
        resolved_config: Mapping[str, Any],
        run_manifest: Mapping[str, Any],
    ) -> None:
        validate_run_manifest(run_manifest)
        if run_manifest["protocol_id"] != self.protocol.protocol_id:
            raise ValueError("run_manifest protocol_id does not match ArtifactStore")
        if run_manifest["class_order_hash"] != self.protocol.class_order_hash:
            raise ValueError("run_manifest class_order_hash does not match protocol")
        if run_manifest["protocol_hash"] != self.protocol.protocol_hash:
            raise ValueError("run_manifest protocol_hash does not match protocol")
        if run_manifest["track"] != self.track:
            raise ValueError("run_manifest track does not match ArtifactStore")
        if run_manifest["method"] != self.method_name:
            raise ValueError("run_manifest method does not match ArtifactStore")
        if int(run_manifest["seed"]) != self.seed:
            raise ValueError("run_manifest seed does not match ArtifactStore")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.score_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.root / "config_resolved.json", resolved_config)
        self._write_json(self.root / "run_manifest.json", run_manifest)
        (self.root / "train.log").touch(exist_ok=True)

    def append_log(self, message: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(message.rstrip() + "\n")

    def save_scores(
        self, task_id: int, prediction: PredictionOutput
    ) -> Path:
        self.score_dir.mkdir(parents=True, exist_ok=True)
        path = self.score_dir / f"task{task_id}_scores.pt"
        torch.save(self._score_payload(task_id, prediction), path)
        return path

    @staticmethod
    def _score_payload(
        task_id: int, prediction: PredictionOutput
    ) -> Dict[str, Any]:
        return {
            "scores": prediction.scores.detach().cpu(),
            "targets": prediction.targets.detach().cpu(),
            "sample_ids": list(prediction.sample_ids),
            "class_order_hash": prediction.class_order_hash,
            "split_hash": prediction.split_hash,
            "task_id": int(task_id),
        }

    @staticmethod
    def validate_shard_run_id(shard_run_id: str) -> str:
        value = str(shard_run_id)
        if not _SHARD_RUN_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "shard_run_id must contain only letters, numbers, '.', '_', "
                "or '-', start with a letter/number, and be at most 128 chars"
            )
        return value

    def task_shard_dir(self, shard_run_id: str, task_id: int) -> Path:
        run_id = self.validate_shard_run_id(shard_run_id)
        if not isinstance(task_id, int) or task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        return self.shard_dir / run_id / f"task{task_id}"

    def shard_checkpoint_path(
        self, shard_run_id: str, task_id: int
    ) -> Path:
        path = self.task_shard_dir(shard_run_id, task_id) / "checkpoint.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_shard_scores(
        self,
        shard_run_id: str,
        task_id: int,
        prediction: PredictionOutput,
    ) -> Path:
        path = self.task_shard_dir(shard_run_id, task_id) / "scores.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._score_payload(task_id, prediction), path)
        return path

    def load_shard_scores(
        self, shard_run_id: str, task_id: int
    ) -> PredictionOutput:
        path = self.task_shard_dir(shard_run_id, task_id) / "scores.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"Shard score payload must be a mapping: {path}")
        required = {
            "scores",
            "targets",
            "sample_ids",
            "class_order_hash",
            "split_hash",
            "task_id",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                f"Shard score payload is missing fields: {', '.join(missing)}"
            )
        if int(payload["task_id"]) != task_id:
            raise ValueError(
                f"Shard score task_id {payload['task_id']} != {task_id}"
            )
        return PredictionOutput(
            scores=payload["scores"],
            targets=payload["targets"],
            sample_ids=list(payload["sample_ids"]),
            class_order_hash=str(payload["class_order_hash"]),
            split_hash=str(payload["split_hash"]),
        )

    def append_shard_log(
        self, shard_run_id: str, task_id: int, message: str
    ) -> None:
        path = self.task_shard_dir(shard_run_id, task_id) / "train.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(message.rstrip() + "\n")

    def write_task_shard_metadata(
        self,
        shard_run_id: str,
        task_id: int,
        payload: Mapping[str, Any],
    ) -> Path:
        path = self.task_shard_dir(shard_run_id, task_id) / "metadata.json"
        resolved = dict(payload)
        resolved["shard_schema_version"] = SHARD_SCHEMA_VERSION
        resolved["shard_run_id"] = self.validate_shard_run_id(shard_run_id)
        resolved["task_id"] = int(task_id)
        self._write_json_atomic(path, resolved)
        return path

    def load_task_shard_metadata(
        self, shard_run_id: str, task_id: int
    ) -> Dict[str, Any]:
        path = self.task_shard_dir(shard_run_id, task_id) / "metadata.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed task shard: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"Task shard metadata must be a mapping: {path}")
        return payload

    def publish_task_shard(self, shard_run_id: str, task_id: int) -> None:
        source_dir = self.task_shard_dir(shard_run_id, task_id)
        score_source = source_dir / "scores.pt"
        checkpoint_source = source_dir / "checkpoint.pth"
        if not score_source.is_file() or not checkpoint_source.is_file():
            raise FileNotFoundError(
                f"Task {task_id} shard is missing scores or checkpoint"
            )
        self.score_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            score_source,
            self.score_dir / f"task{task_id}_scores.pt",
        )
        shutil.copy2(
            checkpoint_source,
            self.checkpoint_dir / f"task{task_id}.pth",
        )

    def merge_shard_logs(self, shard_run_id: str, task_ids: Sequence[int]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / "train.log"
        with destination.open("w", encoding="utf-8") as output:
            for task_id in task_ids:
                source = (
                    self.task_shard_dir(shard_run_id, task_id) / "train.log"
                )
                if source.is_file():
                    output.write(source.read_text(encoding="utf-8"))

    def export_sync_results(self, shard_run_id: str) -> Path:
        """Create one download-ready folder containing no checkpoint weights."""

        run_id = self.validate_shard_run_id(shard_run_id)
        destination = self.root / "results_to_sync" / run_id
        required_files = (
            self.root / "config_resolved.json",
            self.root / "run_manifest.json",
            self.root / "report.html",
            self.root / "train.log",
            self.metrics_dir / "task_metrics.json",
            self.metrics_dir / "summary.json",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot export incomplete benchmark artifacts: "
                + ", ".join(missing)
            )
        state_dir = self.shard_dir / run_id / "_state"
        failed = sorted(state_dir.glob("task*.failed"))
        if failed:
            raise RuntimeError(
                "Cannot export a run with failed task markers: "
                + ", ".join(path.name for path in failed)
            )

        def copy_file(source: Path, relative: Union[str, Path]) -> None:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        for source in required_files[:4]:
            copy_file(source, source.name)
        for source in required_files[4:]:
            copy_file(source, Path("metrics") / source.name)
        for task_id in range(self.protocol.num_tasks):
            score = self.score_dir / f"task{task_id}_scores.pt"
            metadata = (
                self.task_shard_dir(run_id, task_id) / "metadata.json"
            )
            shard_log = self.task_shard_dir(run_id, task_id) / "train.log"
            console_log = state_dir / f"task{task_id}.log"
            for required in (score, metadata, shard_log, console_log):
                if not required.is_file():
                    raise FileNotFoundError(
                        f"Cannot export missing task artifact: {required}"
                    )
            copy_file(score, Path("scores") / score.name)
            copy_file(
                metadata,
                Path("shard_metadata") / f"task{task_id}.json",
            )
            copy_file(
                shard_log,
                Path("logs") / f"task{task_id}_metrics.log",
            )
            copy_file(
                console_log,
                Path("logs") / f"task{task_id}_console.log",
            )
        merge_log = state_dir / "merge.log"
        if merge_log.is_file():
            copy_file(merge_log, Path("logs") / "merge.log")

        weight_files = sorted(destination.rglob("*.pth"))
        if weight_files:
            raise RuntimeError(
                "results_to_sync must not contain .pth files"
            )
        exported_files = []
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or path.name == "sync_manifest.json":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            exported_files.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        self._write_json_atomic(
            destination / "sync_manifest.json",
            {
                "bundle_schema_version": 1,
                "protocol_id": self.protocol.protocol_id,
                "track": self.track,
                "method": self.method_name,
                "seed": self.seed,
                "shard_run_id": run_id,
                "contains_pth": False,
                "download_this_directory": str(destination),
                "excluded_heavy_artifacts": [
                    str(self.checkpoint_dir),
                    str(self.shard_dir / run_id / "task*/checkpoint.pth"),
                ],
                "files": exported_files,
            },
        )
        return destination

    def write_task_metrics(self, rows: Sequence[TaskMetrics]) -> Path:
        path = self.metrics_dir / "task_metrics.json"
        self._write_json(path, {"tasks": [row.as_dict() for row in rows]})
        return path

    def write_summary(
        self,
        summary: BenchmarkSummary,
        method_family: str,
        backbone: str,
        replay_memory_samples: int,
        replay_memory_bytes: int,
        parameter_growth: int,
    ) -> Path:
        main_table = {
            "method": self.method_name,
            "type": method_family,
            "replay_memory": {
                "samples": int(replay_memory_samples),
                "bytes": int(replay_memory_bytes),
            },
            "backbone": backbone,
            "final_mAP": float(summary.final_mAP),
            "final_cF1": float(summary.final_cF1),
            "final_oF1": float(summary.final_oF1),
            "average_mAP": float(summary.average_mAP),
            "forgetting": float(summary.forgetting),
            "parameter_growth": int(parameter_growth),
        }
        if tuple(main_table) != MAIN_TABLE_FIELDS:
            raise RuntimeError("Internal main-table schema ordering changed")
        payload = {
            "output_schema_version": self.protocol.output_schema_version,
            "protocol_id": self.protocol.protocol_id,
            "track": self.track,
            "seed": self.seed,
            "main_table": main_table,
            "summary": summary.as_dict(),
        }
        path = self.metrics_dir / "summary.json"
        self._write_json(path, payload)
        self._write_report(payload)
        return path

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    @classmethod
    def _write_json_atomic(
        cls, path: Path, payload: Mapping[str, Any]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.tmp"
        )
        cls._write_json(temporary, payload)
        os.replace(temporary, path)

    def _write_report(self, payload: Mapping[str, Any]) -> None:
        row = payload["main_table"]
        headers = [
            "Method",
            "Type",
            "Replay Memory",
            "Backbone",
            "Final mAP ↑",
            "Final cF1 ↑",
            "Final oF1 ↑",
            "Avg. mAP ↑",
            "Forgetting ↓",
            "Parameter Growth",
        ]
        values = [
            row["method"],
            row["type"],
            f"{row['replay_memory']['samples']} samples / "
            f"{row['replay_memory']['bytes']} bytes",
            row["backbone"],
            f"{row['final_mAP']:.6f}",
            f"{row['final_cF1']:.6f}",
            f"{row['final_oF1']:.6f}",
            f"{row['average_mAP']:.6f}",
            f"{row['forgetting']:.6f}",
            str(row["parameter_growth"]),
        ]
        header_cells = "".join(f"<th>{escape(value)}</th>" for value in headers)
        value_cells = "".join(f"<td>{escape(value)}</td>" for value in values)
        document = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>EMOTIC MLCIL Benchmark Report</title>"
            "<style>body{font-family:Arial,sans-serif;margin:28px;color:#172033}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
            "th{background:#334155;color:white}</style></head><body>"
            f"<h1>{escape(self.protocol.protocol_id)} / "
            f"{escape(self.method_name)}</h1>"
            f"<table><thead><tr>{header_cells}</tr></thead>"
            f"<tbody><tr>{value_cells}</tr></tbody></table>"
            "</body></html>"
        )
        (self.root / "report.html").write_text(document, encoding="utf-8")
