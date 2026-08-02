"""Built-in benchmark method adapters."""

from .ddp import DDPBenchmarkMethod
from .clip_classifier import (
    ElasticWeightConsolidationMethod,
    LearningWithoutForgettingMethod,
    SequentialFineTuningMethod,
)
from .krt import KRTBenchmarkMethod
from .csc import CSCBenchmarkMethod

__all__ = [
    "DDPBenchmarkMethod",
    "ElasticWeightConsolidationMethod",
    "LearningWithoutForgettingMethod",
    "KRTBenchmarkMethod",
    "CSCBenchmarkMethod",
    "SequentialFineTuningMethod",
]
