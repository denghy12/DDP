"""Fixtures shared by EMOTIC MLCIL Core v0.1 tests."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.types import TaskContext


EMOTIC_CLASSES = [
    "Affection",
    "Anger",
    "Annoyance",
    "Anticipation",
    "Aversion",
    "Confidence",
    "Disapproval",
    "Disconnection",
    "Disquietment",
    "Doubt/Confusion",
    "Embarrassment",
    "Engagement",
    "Esteem",
    "Excitement",
    "Fatigue",
    "Fear",
    "Happiness",
    "Pain",
    "Peace",
    "Pleasure",
    "Sadness",
    "Sensitivity",
    "Suffering",
    "Surprise",
    "Sympathy",
    "Yearning",
]


def protocol_config(
    class_order: Sequence[str] = EMOTIC_CLASSES,
    tasks: Sequence[Sequence[str]] = (
        EMOTIC_CLASSES[:5],
        EMOTIC_CLASSES[5:8],
        EMOTIC_CLASSES[8:11],
        EMOTIC_CLASSES[11:14],
        EMOTIC_CLASSES[14:17],
        EMOTIC_CLASSES[17:20],
        EMOTIC_CLASSES[20:23],
        EMOTIC_CLASSES[23:26],
    ),
) -> Dict[str, Any]:
    return {
        "protocol_id": "test_protocol",
        "dataset": "EMOTIC",
        "class_order": list(class_order),
        "tasks": [list(task) for task in tasks],
        "train_split": "train",
        "validation_split": "val",
        "test_split": "test",
        "label_visibility": {
            "train": {"current": True, "old": False, "future": False},
            "validation_selection": {
                "current": True,
                "old": False,
                "future": False,
            },
            "evaluator": {"seen": True, "future": False},
        },
        "evaluation_scope": {
            "sample_filter": "intersects_seen_classes",
            "target_columns": "seen_classes",
            "class_order": "protocol",
        },
        "threshold_policy": {
            "kind": "fixed_global",
            "value": 0.5,
            "per_task_selection": False,
            "per_class_selection": False,
        },
        "primary_metric": "final_mAP",
        "auxiliary_metrics": [
            "average_mAP",
            "cF1",
            "oF1",
            "forgetting",
        ],
        "checkpoint_selection": {
            "split": "val",
            "metric": "mAP",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "test_allowed": False,
        },
        "track": "A",
        "seed": 0,
        "output_schema_version": 1,
        "method_options": {
            "ddp": {
                "temperature": {
                    "minimum": 1.0,
                    "maximum": 2.0,
                    "gamma": 0.7,
                }
            }
        },
    }


def make_protocol() -> BenchmarkProtocol:
    return BenchmarkProtocol.from_dict(protocol_config())


def task_context(protocol: BenchmarkProtocol, task_id: int) -> TaskContext:
    current = protocol.current_class_indices(task_id)
    seen = protocol.seen_class_indices(task_id)
    future = protocol.future_class_indices(task_id)
    return TaskContext(
        task_id=task_id,
        protocol_id=protocol.protocol_id,
        class_order=protocol.class_order,
        class_order_hash=protocol.class_order_hash,
        protocol_hash=protocol.protocol_hash,
        current_class_indices=current,
        seen_class_indices=seen,
        future_class_indices=future,
        current_class_names=tuple(protocol.class_order[index] for index in current),
        seen_class_names=tuple(protocol.class_order[index] for index in seen),
        future_class_names=tuple(protocol.class_order[index] for index in future),
        seed=protocol.seed,
        track=protocol.track,
    )
