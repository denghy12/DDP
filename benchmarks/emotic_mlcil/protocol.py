"""Machine-readable protocol loading and validation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union


REQUIRED_KEYS = {
    "protocol_id",
    "dataset",
    "class_order",
    "tasks",
    "train_split",
    "validation_split",
    "test_split",
    "label_visibility",
    "evaluation_scope",
    "threshold_policy",
    "primary_metric",
    "auxiliary_metrics",
    "checkpoint_selection",
    "track",
    "seed",
    "output_schema_version",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Validated protocol whose behavior is derived only from config content."""

    _resolved: Mapping[str, Any]
    class_order: Tuple[str, ...]
    tasks: Tuple[Tuple[str, ...], ...]
    task_indices: Tuple[Tuple[int, ...], ...]
    class_order_hash: str
    protocol_hash: str

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "BenchmarkProtocol":
        resolved = copy.deepcopy(dict(config))
        missing = sorted(REQUIRED_KEYS.difference(resolved))
        if missing:
            raise ValueError(
                "Protocol is missing required keys: " + ", ".join(missing)
            )

        protocol_id = resolved["protocol_id"]
        if not isinstance(protocol_id, str) or not protocol_id.strip():
            raise ValueError("protocol_id must be a non-empty string")
        dataset = resolved["dataset"]
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must be a non-empty string")

        class_order_raw = resolved["class_order"]
        if not isinstance(class_order_raw, list) or not class_order_raw:
            raise ValueError("class_order must be a non-empty list")
        if not all(isinstance(name, str) and name for name in class_order_raw):
            raise ValueError("Every class_order entry must be a non-empty string")
        if len(set(class_order_raw)) != len(class_order_raw):
            raise ValueError("class_order must not contain duplicates")
        class_order = tuple(class_order_raw)

        tasks_raw = resolved["tasks"]
        if not isinstance(tasks_raw, list) or not tasks_raw:
            raise ValueError("tasks must be a non-empty list")
        tasks = []
        for task_id, task in enumerate(tasks_raw):
            if not isinstance(task, list) or not task:
                raise ValueError(f"tasks[{task_id}] must be a non-empty list")
            if not all(isinstance(name, str) and name for name in task):
                raise ValueError(
                    f"Every entry in tasks[{task_id}] must be a class name"
                )
            tasks.append(tuple(task))
        flattened = tuple(name for task in tasks for name in task)
        if flattened != class_order:
            raise ValueError(
                "Flattened tasks must equal class_order exactly; every class "
                "must occur once and in protocol order"
            )

        split_values = [
            resolved["train_split"],
            resolved["validation_split"],
            resolved["test_split"],
        ]
        if not all(isinstance(value, str) and value for value in split_values):
            raise ValueError("train/validation/test splits must be non-empty strings")
        if len(set(split_values)) != 3:
            raise ValueError("train, validation, and test splits must be distinct")

        cls._validate_visibility(resolved["label_visibility"])
        cls._validate_evaluation_scope(resolved["evaluation_scope"])
        cls._validate_threshold_policy(resolved["threshold_policy"])
        cls._validate_checkpoint_selection(resolved["checkpoint_selection"])

        if resolved["primary_metric"] != "final_mAP":
            raise ValueError("Core v0.1 primary_metric must be final_mAP")
        if not isinstance(resolved["auxiliary_metrics"], list):
            raise ValueError("auxiliary_metrics must be a list")
        if str(resolved["track"]) not in {"A", "B"}:
            raise ValueError("track must be A or B")
        if not isinstance(resolved["seed"], int) or resolved["seed"] < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(resolved["output_schema_version"], int)
            or resolved["output_schema_version"] <= 0
        ):
            raise ValueError("output_schema_version must be a positive integer")

        class_to_index = {name: index for index, name in enumerate(class_order)}
        task_indices = tuple(
            tuple(class_to_index[name] for name in task) for task in tasks
        )
        return cls(
            _resolved=resolved,
            class_order=class_order,
            tasks=tuple(tasks),
            task_indices=task_indices,
            class_order_hash=_stable_hash(list(class_order)),
            protocol_hash=_stable_hash(resolved),
        )

    @staticmethod
    def _validate_visibility(value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("label_visibility must be a mapping")
        expected = {
            "train": {"current": True, "old": False, "future": False},
            "validation_selection": {
                "current": True,
                "old": False,
                "future": False,
            },
            "evaluator": {"seen": True, "future": False},
        }
        for scope, fields in expected.items():
            actual = value.get(scope)
            if not isinstance(actual, dict):
                raise ValueError(f"label_visibility.{scope} must be a mapping")
            for key, required in fields.items():
                if actual.get(key) is not required:
                    raise ValueError(
                        f"label_visibility.{scope}.{key} must be {required}"
                    )

    @staticmethod
    def _validate_evaluation_scope(value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("evaluation_scope must be a mapping")
        required = {
            "sample_filter": "intersects_seen_classes",
            "target_columns": "seen_classes",
            "class_order": "protocol",
        }
        for key, expected in required.items():
            if value.get(key) != expected:
                raise ValueError(f"evaluation_scope.{key} must be {expected}")

    @staticmethod
    def _validate_threshold_policy(value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("threshold_policy must be a mapping")
        if value.get("kind") != "fixed_global":
            raise ValueError("threshold_policy.kind must be fixed_global")
        threshold = value.get("value")
        if not isinstance(threshold, (int, float)) or float(threshold) != 0.5:
            raise ValueError("Core v0.1 requires one global threshold of 0.5")
        if value.get("per_task_selection") is not False:
            raise ValueError("Per-task threshold selection is forbidden")
        if value.get("per_class_selection") is not False:
            raise ValueError("Per-class threshold selection is forbidden")

    @staticmethod
    def _validate_checkpoint_selection(value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("checkpoint_selection must be a mapping")
        if value.get("split") != "val":
            raise ValueError("Checkpoint selection must use val")
        if value.get("metric") != "mAP" or value.get("mode") != "max":
            raise ValueError("Checkpoint selection must maximize validation mAP")
        if value.get("tie_break") != "earliest_epoch":
            raise ValueError("Checkpoint ties must select the earliest epoch")
        if value.get("test_allowed") is not False:
            raise ValueError("Test cannot be used for checkpoint selection")

    @property
    def protocol_id(self) -> str:
        return str(self._resolved["protocol_id"])

    @property
    def dataset(self) -> str:
        return str(self._resolved["dataset"])

    @property
    def train_split(self) -> str:
        return str(self._resolved["train_split"])

    @property
    def validation_split(self) -> str:
        return str(self._resolved["validation_split"])

    @property
    def test_split(self) -> str:
        return str(self._resolved["test_split"])

    @property
    def track(self) -> str:
        return str(self._resolved["track"])

    @property
    def seed(self) -> int:
        return int(self._resolved["seed"])

    @property
    def threshold(self) -> float:
        return float(self._resolved["threshold_policy"]["value"])

    @property
    def output_schema_version(self) -> int:
        return int(self._resolved["output_schema_version"])

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def num_classes(self) -> int:
        return len(self.class_order)

    def _validate_task_id(self, task_id: int) -> None:
        if not isinstance(task_id, int) or not 0 <= task_id < self.num_tasks:
            raise ValueError(
                f"task_id must be in [0, {self.num_tasks}), got {task_id}"
            )

    def current_class_indices(self, task_id: int) -> Tuple[int, ...]:
        self._validate_task_id(task_id)
        return self.task_indices[task_id]

    def seen_class_indices(self, task_id: int) -> Tuple[int, ...]:
        self._validate_task_id(task_id)
        return tuple(
            class_id
            for indices in self.task_indices[: task_id + 1]
            for class_id in indices
        )

    def future_class_indices(self, task_id: int) -> Tuple[int, ...]:
        self._validate_task_id(task_id)
        return tuple(
            class_id
            for indices in self.task_indices[task_id + 1 :]
            for class_id in indices
        )

    def task_class_range(self, task_id: int) -> Tuple[int, int]:
        current = self.current_class_indices(task_id)
        return current[0], current[-1] + 1

    @property
    def task_class_ranges(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self.task_class_range(task_id) for task_id in range(self.num_tasks))

    def introduction_task(self, class_id: int) -> int:
        if not isinstance(class_id, int) or not 0 <= class_id < self.num_classes:
            raise ValueError(
                f"class_id must be in [0, {self.num_classes}), got {class_id}"
            )
        for task_id, indices in enumerate(self.task_indices):
            if class_id in indices:
                return task_id
        raise RuntimeError(f"Class {class_id} is not assigned to a task")

    def method_options(self, method_name: str) -> Mapping[str, Any]:
        options = self._resolved.get("method_options", {})
        if not isinstance(options, dict):
            raise ValueError("method_options must be a mapping")
        value = options.get(method_name, {})
        if not isinstance(value, dict):
            raise ValueError(f"method_options.{method_name} must be a mapping")
        return copy.deepcopy(value)

    def as_dict(self) -> Dict[str, Any]:
        resolved = copy.deepcopy(dict(self._resolved))
        resolved["class_order_hash"] = self.class_order_hash
        resolved["protocol_hash"] = self.protocol_hash
        return resolved


def load_protocol(path: Union[str, Path]) -> BenchmarkProtocol:
    """Load YAML based on its content; filenames carry no protocol semantics."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise RuntimeError(
            "PyYAML is required to load benchmark protocol files"
        ) from exc

    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Protocol YAML must contain a top-level mapping")
    return BenchmarkProtocol.from_dict(payload)
