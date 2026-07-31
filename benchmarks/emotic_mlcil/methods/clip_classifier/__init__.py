"""Repository-native CLIP-visual continual classifier baselines."""

from .method import (
    CLIPContinualMethod,
    ElasticWeightConsolidationMethod,
    LearningWithoutForgettingMethod,
    SequentialFineTuningMethod,
)
from .model import GrowingMultiLabelClassifier

__all__ = [
    "CLIPContinualMethod",
    "ElasticWeightConsolidationMethod",
    "GrowingMultiLabelClassifier",
    "LearningWithoutForgettingMethod",
    "SequentialFineTuningMethod",
]
