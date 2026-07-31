"""Repository-native frozen-CLIP continual classifier baselines."""

from .method import (
    ElasticWeightConsolidationMethod,
    FrozenCLIPContinualMethod,
    LearningWithoutForgettingMethod,
    SequentialFineTuningMethod,
)
from .model import GrowingMultiLabelClassifier, ResidualFeatureAdapter

__all__ = [
    "ElasticWeightConsolidationMethod",
    "FrozenCLIPContinualMethod",
    "GrowingMultiLabelClassifier",
    "LearningWithoutForgettingMethod",
    "ResidualFeatureAdapter",
    "SequentialFineTuningMethod",
]
