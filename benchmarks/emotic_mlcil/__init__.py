"""EMOTIC multi-label class-incremental benchmark Core v0.1."""

from .method_base import BenchmarkMethod
from .protocol import BenchmarkProtocol, load_protocol
from .registry import method_class, method_names, register_method
from .types import (
    BenchmarkSummary,
    EvaluationBatch,
    MemoryStatistics,
    ParameterStatistics,
    PredictionOutput,
    TaskContext,
    TaskMetrics,
    TrainBatch,
)
from . import methods as _methods

__all__ = [
    "BenchmarkMethod",
    "BenchmarkProtocol",
    "BenchmarkSummary",
    "EvaluationBatch",
    "MemoryStatistics",
    "ParameterStatistics",
    "PredictionOutput",
    "TaskContext",
    "TaskMetrics",
    "TrainBatch",
    "load_protocol",
    "method_class",
    "method_names",
    "register_method",
]
