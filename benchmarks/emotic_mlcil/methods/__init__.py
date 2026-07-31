"""Built-in benchmark method adapters."""

from .ddp import DDPBenchmarkMethod
from .clip_classifier import (
    ElasticWeightConsolidationMethod,
    LearningWithoutForgettingMethod,
    SequentialFineTuningMethod,
)

__all__ = [
    "DDPBenchmarkMethod",
    "ElasticWeightConsolidationMethod",
    "LearningWithoutForgettingMethod",
    "SequentialFineTuningMethod",
]
