"""Shared data contracts for the EMOTIC MLCIL benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


_EVALUATOR_ACCESS_SENTINEL = object()


class EvaluatorAccessToken:
    """Capability required to materialize evaluator-only seen targets."""

    __slots__ = ("_sentinel",)

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _EVALUATOR_ACCESS_SENTINEL:
            raise RuntimeError("EvaluatorAccessToken can only be issued by the evaluator")
        self._sentinel = sentinel

    def is_valid(self) -> bool:
        return self._sentinel is _EVALUATOR_ACCESS_SENTINEL


def _issue_evaluator_access() -> EvaluatorAccessToken:
    return EvaluatorAccessToken(_EVALUATOR_ACCESS_SENTINEL)


@dataclass(frozen=True)
class TaskContext:
    task_id: int
    protocol_id: str
    class_order: Tuple[str, ...]
    class_order_hash: str
    protocol_hash: str
    current_class_indices: Tuple[int, ...]
    seen_class_indices: Tuple[int, ...]
    future_class_indices: Tuple[int, ...]
    current_class_names: Tuple[str, ...]
    seen_class_names: Tuple[str, ...]
    future_class_names: Tuple[str, ...]
    seed: int
    track: str


@dataclass
class TrainBatch:
    """Method-facing batch; deliberately contains no old/future target tensor."""

    images: torch.Tensor
    sample_ids: List[str]
    targets_current: torch.Tensor
    visible_mask: torch.Tensor


@dataclass
class EvaluationBatch:
    """Evaluator-only batch containing exactly the currently seen columns."""

    images: torch.Tensor
    sample_ids: List[str]
    targets_seen: torch.Tensor
    class_order_hash: str
    split_hash: str


@dataclass
class PredictionOutput:
    scores: torch.Tensor
    targets: torch.Tensor
    sample_ids: List[str]
    class_order_hash: str
    split_hash: str


@dataclass(frozen=True)
class ParameterStatistics:
    total_parameters: int
    trainable_parameters: int
    incremental_parameters: int
    per_task_incremental_parameters: Mapping[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_parameters": int(self.total_parameters),
            "trainable_parameters": int(self.trainable_parameters),
            "incremental_parameters": int(self.incremental_parameters),
            "per_task_incremental_parameters": {
                str(key): int(value)
                for key, value in self.per_task_incremental_parameters.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterStatistics":
        per_task = payload.get("per_task_incremental_parameters", {})
        if not isinstance(per_task, Mapping):
            raise ValueError(
                "per_task_incremental_parameters must be a mapping"
            )
        return cls(
            total_parameters=int(payload["total_parameters"]),
            trainable_parameters=int(payload["trainable_parameters"]),
            incremental_parameters=int(payload["incremental_parameters"]),
            per_task_incremental_parameters={
                int(key): int(value) for key, value in per_task.items()
            },
        )


@dataclass(frozen=True)
class MemoryStatistics:
    replay_memory_samples: int
    replay_memory_bytes: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "replay_memory_samples": int(self.replay_memory_samples),
            "replay_memory_bytes": int(self.replay_memory_bytes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryStatistics":
        return cls(
            replay_memory_samples=int(payload["replay_memory_samples"]),
            replay_memory_bytes=int(payload["replay_memory_bytes"]),
        )


@dataclass
class TaskMetrics:
    task_id: int
    seen_classes: int
    samples: int
    threshold: float
    per_class_ap: List[float]
    mAP: float
    cPrecision: float
    cRecall: float
    cF1: float
    oPrecision: float
    oRecall: float
    oF1: float
    class_order_hash: str
    split_hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "seen_classes": int(self.seen_classes),
            "samples": int(self.samples),
            "threshold": float(self.threshold),
            "per_class_ap": [float(value) for value in self.per_class_ap],
            "mAP": float(self.mAP),
            "cPrecision": float(self.cPrecision),
            "cRecall": float(self.cRecall),
            "cF1": float(self.cF1),
            "oPrecision": float(self.oPrecision),
            "oRecall": float(self.oRecall),
            "oF1": float(self.oF1),
            "class_order_hash": self.class_order_hash,
            "split_hash": self.split_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskMetrics":
        return cls(
            task_id=int(payload["task_id"]),
            seen_classes=int(payload["seen_classes"]),
            samples=int(payload["samples"]),
            threshold=float(payload["threshold"]),
            per_class_ap=[
                float(value) for value in payload["per_class_ap"]
            ],
            mAP=float(payload["mAP"]),
            cPrecision=float(payload["cPrecision"]),
            cRecall=float(payload["cRecall"]),
            cF1=float(payload["cF1"]),
            oPrecision=float(payload["oPrecision"]),
            oRecall=float(payload["oRecall"]),
            oF1=float(payload["oF1"]),
            class_order_hash=str(payload["class_order_hash"]),
            split_hash=str(payload["split_hash"]),
        )


@dataclass
class BenchmarkSummary:
    task_metrics: Sequence[TaskMetrics]
    final_mAP: float
    average_mAP: float
    final_cF1: float
    final_oF1: float
    forgetting: float
    per_class_forgetting: Mapping[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_metrics": [row.as_dict() for row in self.task_metrics],
            "final_mAP": float(self.final_mAP),
            "average_mAP": float(self.average_mAP),
            "final_cF1": float(self.final_cF1),
            "final_oF1": float(self.final_oF1),
            "forgetting": float(self.forgetting),
            "per_class_forgetting": {
                key: float(value) for key, value in self.per_class_forgetting.items()
            },
        }
