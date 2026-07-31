"""Legacy DDP benchmark adapter."""

from .method import DDPBenchmarkMethod, legacy_ddp_predict_scores

__all__ = ["DDPBenchmarkMethod", "legacy_ddp_predict_scores"]
