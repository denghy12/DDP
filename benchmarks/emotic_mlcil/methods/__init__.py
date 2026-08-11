"""Built-in benchmark method adapters."""

from .ddp import DDPBenchmarkMethod
from .clip_classifier import (
    ElasticWeightConsolidationMethod,
    LearningWithoutForgettingMethod,
    SequentialFineTuningMethod,
)
from .krt import KRTBenchmarkMethod
from .csc import CSCBenchmarkMethod
from .multi_lane import MultiLaneBenchmarkMethod
from .l3a import L3ABenchmarkMethod
from .original_ddp import OriginalDDPBenchmarkMethod
from .agcn import AGCNBenchmarkMethod
from .replay import DERPPBenchmarkMethod, ERBenchmarkMethod, PRSBenchmarkMethod
from .emot_net_ft import EMOTNetFTBenchmarkMethod
from .cocoer_ft import CocoERFTBenchmarkMethod

__all__ = [
    "DDPBenchmarkMethod",
    "ElasticWeightConsolidationMethod",
    "LearningWithoutForgettingMethod",
    "KRTBenchmarkMethod",
    "L3ABenchmarkMethod",
    "CSCBenchmarkMethod",
    "MultiLaneBenchmarkMethod",
    "OriginalDDPBenchmarkMethod",
    "AGCNBenchmarkMethod",
    "SequentialFineTuningMethod",
    "ERBenchmarkMethod",
    "PRSBenchmarkMethod",
    "DERPPBenchmarkMethod",
    "EMOTNetFTBenchmarkMethod",
    "CocoERFTBenchmarkMethod",
]
