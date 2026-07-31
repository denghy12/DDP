"""Uniform method interface for all benchmark families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

from .types import (
    MemoryStatistics,
    ParameterStatistics,
    PredictionOutput,
    TaskContext,
)


class BenchmarkMethod(ABC):
    method_name = "unregistered"
    method_family = "unregistered"
    backbone = "TBD"
    supported_tracks = ("A", "B")

    def training_log_records(self) -> Sequence[Mapping[str, Any]]:
        """Return task-local selection/loss records for the standard log."""

        return ()

    def resolved_method_config(self) -> Mapping[str, Any]:
        """Return method settings that are not part of the dataset protocol."""

        return {}

    @abstractmethod
    def begin_task(self, task_context: TaskContext) -> None:
        """Prepare the method for a protocol-defined task."""

    @abstractmethod
    def train_task(self, train_loader: Iterable[Any], val_loader: Iterable[Any]) -> None:
        """Train/select using only method-facing train/validation batches."""

    @abstractmethod
    def predict_scores(self, data_loader: Iterable[Any]) -> PredictionOutput:
        """Return aligned [N, C_seen] scores, targets, and ordered IDs."""

    @abstractmethod
    def end_task(self) -> None:
        """Finalize task-local state."""

    @abstractmethod
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Write a method checkpoint."""

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load a method checkpoint without changing protocol behavior."""

    @abstractmethod
    def parameter_statistics(self) -> ParameterStatistics:
        """Return total, optimizer-updated, and incremental parameter counts."""

    @abstractmethod
    def memory_statistics(self) -> MemoryStatistics:
        """Return replay memory in both samples and bytes."""
