"""Built-in benchmark method adapters."""

from .ddp import DDPBenchmarkMethod
from .frozen_clip import (
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
