"""Single strict evaluator for every benchmark method."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch

from evaluation_metrics import mAP as legacy_map
from src.helper_functions.detail_report import binary_counts

from .protocol import BenchmarkProtocol
from .types import (
    BenchmarkSummary,
    EvaluatorAccessToken,
    PredictionOutput,
    TaskMetrics,
    _issue_evaluator_access,
)


class BenchmarkEvaluator:
    def __init__(self, protocol: BenchmarkProtocol) -> None:
        self.protocol = protocol
        self._access_token = _issue_evaluator_access()

    @property
    def access_token(self) -> EvaluatorAccessToken:
        return self._access_token

    def validate_prediction(
        self,
        task_id: int,
        prediction: PredictionOutput,
        expected_sample_ids: Sequence[str],
        expected_split_hash: str,
    ) -> None:
        expected_classes = len(self.protocol.seen_class_indices(task_id))
        scores = prediction.scores
        targets = prediction.targets
        if not isinstance(scores, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            raise TypeError("scores and targets must be torch.Tensor values")
        if scores.ndim != 2 or targets.ndim != 2:
            raise ValueError("scores and targets must both have shape [N, C_seen]")
        if scores.shape != targets.shape:
            raise ValueError(
                f"scores shape {tuple(scores.shape)} != targets shape "
                f"{tuple(targets.shape)}"
            )
        if scores.shape[1] != expected_classes:
            raise ValueError(
                f"Task {task_id} requires {expected_classes} seen classes, "
                f"received {scores.shape[1]}"
            )
        if scores.shape[0] == 0:
            raise ValueError("Evaluation predictions must contain at least one sample")
        if scores.shape[0] != len(prediction.sample_ids):
            raise ValueError("sample_ids length must equal score/target row count")
        if not all(
            isinstance(sample_id, str) and sample_id
            for sample_id in prediction.sample_ids
        ):
            raise ValueError("sample_ids must be non-empty strings")
        if len(prediction.sample_ids) != len(set(prediction.sample_ids)):
            raise ValueError("sample_ids must be unique")
        if list(prediction.sample_ids) != list(expected_sample_ids):
            raise ValueError("sample ID order does not match the evaluator dataset")
        if prediction.class_order_hash != self.protocol.class_order_hash:
            raise ValueError("Prediction class_order_hash does not match protocol")
        if prediction.split_hash != expected_split_hash:
            raise ValueError("Prediction split_hash does not match evaluator dataset")
        if not torch.isfinite(scores).all():
            raise ValueError("scores contain NaN or Inf")
        if not bool(((scores >= 0) & (scores <= 1)).all()):
            raise ValueError("scores must be probabilities in [0, 1]")
        if not torch.isfinite(targets).all():
            raise ValueError("targets contain NaN or Inf")
        binary_targets = (targets == 0) | (targets == 1)
        if not bool(binary_targets.all()):
            raise ValueError("targets must contain only binary 0/1 values")

    def evaluate_task(
        self,
        task_id: int,
        prediction: PredictionOutput,
        expected_sample_ids: Sequence[str],
        expected_split_hash: str,
    ) -> TaskMetrics:
        self.validate_prediction(
            task_id,
            prediction,
            expected_sample_ids=expected_sample_ids,
            expected_split_hash=expected_split_hash,
        )
        scores = prediction.scores.detach().float().cpu()
        targets = prediction.targets.detach().float().cpu()
        predictions = scores.gt(self.protocol.threshold)
        target_flags = targets.bool()

        map_value, per_class_fraction = legacy_map(
            targets.numpy(),
            scores.numpy(),
        )
        per_class_ap = [
            100.0 * float(value) for value in per_class_fraction.tolist()
        ]
        class_counts: List[Dict[str, float]] = []
        for class_id in range(scores.shape[1]):
            class_counts.append(
                binary_counts(
                    predictions[:, class_id],
                    target_flags[:, class_id],
                )
            )
        overall = binary_counts(predictions, target_flags)

        def class_mean(key: str) -> float:
            return 100.0 * sum(row[key] for row in class_counts) / len(class_counts)

        return TaskMetrics(
            task_id=task_id,
            seen_classes=scores.shape[1],
            samples=scores.shape[0],
            threshold=self.protocol.threshold,
            per_class_ap=per_class_ap,
            mAP=float(map_value),
            cPrecision=class_mean("precision"),
            cRecall=class_mean("recall"),
            cF1=class_mean("f1"),
            oPrecision=100.0 * overall["precision"],
            oRecall=100.0 * overall["recall"],
            oF1=100.0 * overall["f1"],
            class_order_hash=prediction.class_order_hash,
            split_hash=prediction.split_hash,
        )

    def summarize(self, task_metrics: Sequence[TaskMetrics]) -> BenchmarkSummary:
        if not task_metrics:
            raise ValueError("At least one task metric row is required")
        ordered = sorted(task_metrics, key=lambda row: row.task_id)
        expected_ids = list(range(len(ordered)))
        actual_ids = [row.task_id for row in ordered]
        if actual_ids != expected_ids:
            raise ValueError(
                f"Task metrics must be contiguous from task 0, got {actual_ids}"
            )
        for row in ordered:
            expected_seen = len(self.protocol.seen_class_indices(row.task_id))
            if row.seen_classes != expected_seen:
                raise ValueError(
                    f"Task {row.task_id} metrics contain {row.seen_classes} "
                    f"classes; expected {expected_seen}"
                )
            if row.class_order_hash != self.protocol.class_order_hash:
                raise ValueError("Metric class_order_hash does not match protocol")
            if len(row.per_class_ap) != expected_seen:
                raise ValueError("per_class_ap length does not match seen classes")
            if row.samples <= 0:
                raise ValueError("Task metrics must contain at least one sample")
            if row.threshold != self.protocol.threshold:
                raise ValueError("Task metric threshold does not match protocol")
            if not row.split_hash:
                raise ValueError("Task metric split_hash must not be empty")
            values = [
                row.mAP,
                row.cPrecision,
                row.cRecall,
                row.cF1,
                row.oPrecision,
                row.oRecall,
                row.oF1,
                *row.per_class_ap,
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Task metrics contain NaN or Inf")

        final_row = ordered[-1]
        final_task_id = final_row.task_id
        per_class_forgetting: Dict[str, float] = {}
        for class_id, class_name in enumerate(self.protocol.class_order):
            introduction = self.protocol.introduction_task(class_id)
            if introduction >= final_task_id:
                continue
            history = [
                row.per_class_ap[class_id]
                for row in ordered
                if row.task_id >= introduction and class_id < row.seen_classes
            ]
            if not history:
                raise RuntimeError(
                    f"No AP history found for old class {class_name}"
                )
            per_class_forgetting[class_name] = max(history) - history[-1]
        forgetting = (
            sum(per_class_forgetting.values()) / len(per_class_forgetting)
            if per_class_forgetting
            else 0.0
        )

        return BenchmarkSummary(
            task_metrics=ordered,
            final_mAP=final_row.mAP,
            average_mAP=sum(row.mAP for row in ordered) / len(ordered),
            final_cF1=final_row.cF1,
            final_oF1=final_row.oF1,
            forgetting=forgetting,
            per_class_forgetting=per_class_forgetting,
        )


def assert_prediction_equivalence(
    legacy: PredictionOutput,
    benchmark: PredictionOutput,
    score_atol: float = 1e-7,
) -> float:
    """Strict identity/alignment checks used by DDP integration validation."""

    if legacy.sample_ids != benchmark.sample_ids:
        raise AssertionError("Legacy and benchmark sample IDs differ")
    if legacy.class_order_hash != benchmark.class_order_hash:
        raise AssertionError("Legacy and benchmark class order hashes differ")
    if legacy.split_hash != benchmark.split_hash:
        raise AssertionError("Legacy and benchmark split hashes differ")
    if not torch.equal(legacy.targets, benchmark.targets):
        raise AssertionError("Legacy and benchmark targets differ")
    if legacy.scores.shape != benchmark.scores.shape:
        raise AssertionError("Legacy and benchmark score shapes differ")
    max_error = float(
        (legacy.scores.float() - benchmark.scores.float()).abs().max().item()
    )
    if max_error >= score_atol:
        raise AssertionError(
            f"Score max_abs_error {max_error:.12g} is not < {score_atol}"
        )
    return max_error
