"""Protocol-safe replay memory shared by controlled Track-A baselines."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

from .protocol import BenchmarkProtocol
from .types import MemoryStatistics, TaskContext


@dataclass(frozen=True)
class ReplayMemoryContract:
    """Machine-readable storage and replay-exposure budget."""

    contract_id: str
    protocol_id: str
    samples_per_seen_class: int
    task_capacities: Tuple[int, ...]
    replay_to_current_ratio: float
    update_timing: str
    sample_unit: str
    image_payload: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        protocol: BenchmarkProtocol,
    ) -> "ReplayMemoryContract":
        capacity = value.get("capacity", {})
        payload = value.get("payload", {})
        training = value.get("training", {})
        if not isinstance(capacity, Mapping):
            raise ValueError("Replay capacity configuration must be a mapping")
        if not isinstance(payload, Mapping) or not isinstance(training, Mapping):
            raise ValueError("Replay payload/training configuration must be mappings")
        if capacity.get("kind") != "per_seen_class":
            raise ValueError("Only per_seen_class replay capacity is registered")
        samples_per_seen_class = int(capacity.get("samples_per_seen_class", 0))
        if samples_per_seen_class <= 0:
            raise ValueError("samples_per_seen_class must be positive")
        expected_capacities = tuple(
            samples_per_seen_class * len(protocol.seen_class_indices(task_id))
            for task_id in range(protocol.num_tasks)
        )
        configured_capacities = tuple(
            int(item) for item in capacity.get("task_capacities", ())
        )
        if configured_capacities != expected_capacities:
            raise ValueError(
                "Replay task capacities do not match the protocol class schedule"
            )
        ratio = float(training.get("replay_to_current_ratio", 0.0))
        if not math.isfinite(ratio) or ratio != 1.0:
            raise ValueError("Registered replay-to-current ratio must be 1.0")
        update_timing = str(training.get("update_timing", ""))
        if update_timing != "task_end_single_pass":
            raise ValueError("Unexpected replay memory update timing")
        sample_unit = str(value.get("sample_unit", ""))
        if sample_unit != "unique_emotic_person_sample":
            raise ValueError("Unexpected replay sample unit")
        image_payload = str(payload.get("image", ""))
        if image_payload != "post_transform_float32_tensor":
            raise ValueError("Unexpected replay image payload")
        if payload.get("targets") != "visible_binary_columns_only":
            raise ValueError("Replay may store only visible binary target columns")
        if payload.get("visible_mask") is not True:
            raise ValueError("Replay payload must carry a visible-label mask")
        if payload.get("stable_sample_id") is not True:
            raise ValueError("Replay payload must carry a stable sample ID")
        if training.get("sample_without_replacement") is not True:
            raise ValueError("Replay batches must sample without replacement")
        if training.get("current_and_replay_equal_sample_weight") is not True:
            raise ValueError("Current and replay samples must have equal weight")
        contract = cls(
            contract_id=str(value.get("contract_id", "")),
            protocol_id=str(value.get("protocol_id", "")),
            samples_per_seen_class=samples_per_seen_class,
            task_capacities=configured_capacities,
            replay_to_current_ratio=ratio,
            update_timing=update_timing,
            sample_unit=sample_unit,
            image_payload=image_payload,
        )
        if not contract.contract_id:
            raise ValueError("Replay contract_id must not be empty")
        if contract.protocol_id != protocol.protocol_id:
            raise ValueError("Replay contract protocol_id differs from protocol")
        return contract

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        protocol: BenchmarkProtocol,
    ) -> "ReplayMemoryContract":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("Replay contract YAML root must be a mapping")
        return cls.from_mapping(payload, protocol)

    def capacity_for_task(self, task_id: int) -> int:
        try:
            return self.task_capacities[task_id]
        except IndexError as exc:
            raise ValueError(f"Invalid replay task id: {task_id}") from exc

    def capacity_for_context(self, context: TaskContext) -> int:
        expected = self.samples_per_seen_class * len(context.seen_class_indices)
        capacity = self.capacity_for_task(context.task_id)
        if capacity != expected:
            raise ValueError("Replay capacity differs from task context")
        return capacity

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "protocol_id": self.protocol_id,
            "sample_unit": self.sample_unit,
            "capacity_kind": "per_seen_class",
            "samples_per_seen_class": self.samples_per_seen_class,
            "task_capacities": list(self.task_capacities),
            "replay_to_current_ratio": self.replay_to_current_ratio,
            "update_timing": self.update_timing,
            "image_payload": self.image_payload,
            "deduplicate_by_sample_id": True,
            "stored_targets": "visible_columns_only",
            "future_truth_stored": False,
        }


@dataclass
class ReplayRecord:
    """One retained person-sample with only capture-time-visible labels."""

    image: torch.Tensor
    sample_id: str
    targets: torch.Tensor
    visible_mask: torch.Tensor

    def __post_init__(self) -> None:
        self.image = self.image.detach().cpu().float().contiguous().clone()
        self.targets = self.targets.detach().cpu().float().contiguous().clone()
        self.visible_mask = (
            self.visible_mask.detach().cpu().bool().contiguous().clone()
        )
        self.sample_id = str(self.sample_id)
        if not self.sample_id:
            raise ValueError("Replay sample_id must not be empty")
        if self.targets.ndim != 1 or self.visible_mask.ndim != 1:
            raise ValueError("Replay targets and visible_mask must be vectors")
        if self.targets.shape != self.visible_mask.shape:
            raise ValueError("Replay targets and visible_mask shapes differ")
        visible_targets = self.targets[self.visible_mask]
        if not bool(((visible_targets == 0) | (visible_targets == 1)).all()):
            raise ValueError("Visible replay targets must be binary")
        if bool((self.targets[~self.visible_mask] != 0).any()):
            raise ValueError("Hidden replay target columns must be zero")

    @property
    def positive_indices(self) -> Tuple[int, ...]:
        positive = self.visible_mask & (self.targets > 0.5)
        return tuple(int(index) for index in positive.nonzero().flatten().tolist())

    def merged_with(self, other: "ReplayRecord") -> "ReplayRecord":
        if self.sample_id != other.sample_id:
            raise ValueError("Cannot merge replay records with different IDs")
        if self.image.shape != other.image.shape:
            raise ValueError("Repeated replay image shape changed")
        if self.targets.shape != other.targets.shape:
            raise ValueError("Repeated replay target width changed")
        overlap = self.visible_mask & other.visible_mask
        if bool((self.targets[overlap] != other.targets[overlap]).any()):
            raise ValueError("Repeated replay sample has conflicting visible labels")
        visible_mask = self.visible_mask | other.visible_mask
        targets = torch.where(other.visible_mask, other.targets, self.targets)
        targets = torch.where(visible_mask, targets, torch.zeros_like(targets))
        return ReplayRecord(
            image=other.image,
            sample_id=self.sample_id,
            targets=targets,
            visible_mask=visible_mask,
        )

    def byte_count(self) -> int:
        tensors = (self.image, self.targets, self.visible_mask)
        return sum(value.numel() * value.element_size() for value in tensors) + len(
            self.sample_id.encode("utf-8")
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "sample_id": self.sample_id,
            "targets": self.targets,
            "visible_mask": self.visible_mask,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "ReplayRecord":
        return cls(
            image=payload["image"],
            sample_id=str(payload["sample_id"]),
            targets=payload["targets"],
            visible_mask=payload["visible_mask"],
        )


class ReplayBuffer:
    """Base storage with stable-ID merging and deterministic replay sampling."""

    policy_name = "unregistered"

    def __init__(self, num_classes: int, seed: int) -> None:
        if num_classes <= 0:
            raise ValueError("Replay class count must be positive")
        self.num_classes = int(num_classes)
        self.capacity = 0
        self.records: List[ReplayRecord] = []
        self.stream_observations = 0
        self._rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.records)

    def set_capacity(self, capacity: int) -> None:
        capacity = int(capacity)
        if capacity < self.capacity:
            raise ValueError("Registered replay capacity must not shrink")
        if capacity < len(self.records):
            raise ValueError("Replay capacity cannot be smaller than stored memory")
        self.capacity = capacity

    def _record_index(self, sample_id: str) -> Optional[int]:
        for index, record in enumerate(self.records):
            if record.sample_id == sample_id:
                return index
        return None

    def observe(self, record: ReplayRecord) -> None:
        if record.targets.numel() != self.num_classes:
            raise ValueError("Replay record target width differs from protocol")
        existing = self._record_index(record.sample_id)
        if existing is not None:
            self.records[existing] = self.records[existing].merged_with(record)
            self._observe_statistics(record)
            return
        self.stream_observations += 1
        self._observe_new(record)
        self._observe_statistics(record)
        if len(self.records) > self.capacity:
            raise RuntimeError("Replay policy exceeded its registered capacity")

    def _observe_statistics(self, record: ReplayRecord) -> None:
        del record

    def _observe_new(self, record: ReplayRecord) -> None:
        raise NotImplementedError

    def sample(self, count: int) -> Tuple[ReplayRecord, ...]:
        if count <= 0 or not self.records:
            return ()
        count = min(int(count), len(self.records))
        indices = self._rng.sample(range(len(self.records)), count)
        return tuple(self.records[index] for index in indices)

    def memory_statistics(self) -> MemoryStatistics:
        return MemoryStatistics(
            replay_memory_samples=len(self.records),
            replay_memory_bytes=sum(record.byte_count() for record in self.records),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy_name,
            "num_classes": self.num_classes,
            "capacity": self.capacity,
            "stream_observations": self.stream_observations,
            "records": [record.state_dict() for record in self.records],
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("policy") != self.policy_name:
            raise ValueError("Replay checkpoint policy differs from method")
        if int(payload.get("num_classes", -1)) != self.num_classes:
            raise ValueError("Replay checkpoint class count differs")
        records = [
            ReplayRecord.from_state_dict(item)
            for item in list(payload.get("records", []))
        ]
        if len({record.sample_id for record in records}) != len(records):
            raise ValueError("Replay checkpoint contains duplicate sample IDs")
        capacity = int(payload.get("capacity", -1))
        if capacity < len(records):
            raise ValueError("Replay checkpoint exceeds its saved capacity")
        self.capacity = capacity
        self.stream_observations = int(payload.get("stream_observations", 0))
        self.records = records
        self._rng.setstate(payload["rng_state"])


class ReservoirReplayBuffer(ReplayBuffer):
    """Conventional reservoir sampling used by the ER control."""

    policy_name = "reservoir"

    def _observe_new(self, record: ReplayRecord) -> None:
        if self.capacity <= 0:
            return
        if len(self.records) < self.capacity:
            self.records.append(record)
            return
        index = self._rng.randrange(self.stream_observations)
        if index < self.capacity:
            self.records[index] = record


class PartitionedReservoirReplayBuffer(ReplayBuffer):
    """Independent implementation of the PRS multi-label buffer policy."""

    policy_name = "partitioned_reservoir"

    def __init__(self, num_classes: int, seed: int, allocation_power: float) -> None:
        super().__init__(num_classes=num_classes, seed=seed)
        if not math.isfinite(allocation_power):
            raise ValueError("PRS allocation power must be finite")
        self.allocation_power = float(allocation_power)
        self.observed_positive_counts = [0.0] * self.num_classes
        self.target_proportions = [0.0] * self.num_classes
        # Upstream uses an OrderedDict of substreams.  A class keeps its
        # first-observation position even while it has no retained member.
        self.substream_order: List[int] = []

    def _register_substreams(self, record: ReplayRecord) -> None:
        for index in record.positive_indices:
            if index not in self.substream_order:
                self.substream_order.append(index)

    def _active_proportions(self) -> Dict[int, float]:
        counts = self._buffer_label_counts()
        return {
            index: self.target_proportions[index]
            for index, count in counts.items()
            if count > 0
        }

    def _update_proportions(self) -> None:
        active = tuple(self._buffer_label_counts())
        weights = {
            index: self.observed_positive_counts[index] ** self.allocation_power
            for index in active
            if self.observed_positive_counts[index] > 0
        }
        total = sum(weights.values())
        if total <= 0:
            return
        for index, value in weights.items():
            self.target_proportions[index] = value / total

    def _buffer_label_counts(
        self,
        records: Optional[Sequence[ReplayRecord]] = None,
    ) -> Dict[int, int]:
        counts: Dict[int, int] = {
            index: 0 for index in self.substream_order
        }
        for record in self.records if records is None else records:
            for index in record.positive_indices:
                counts[index] = counts.get(index, 0) + 1
        return {index: count for index, count in counts.items() if count > 0}

    def _deltas(
        self,
        records: Optional[Sequence[ReplayRecord]] = None,
    ) -> Dict[int, float]:
        counts = self._buffer_label_counts(records)
        proportions = {
            index: self.target_proportions[index]
            for index, count in counts.items()
            if count > 0
        }
        membership_total = float(sum(counts.values()))
        return {
            index: counts.get(index, 0) - proportion * membership_total
            for index, proportion in proportions.items()
            if counts.get(index, 0) > 0
        }

    def _weighted_choice(self, values: Sequence[int], weights: Sequence[float]) -> int:
        if len(values) != len(weights) or not values:
            raise ValueError("Invalid PRS weighted choice")
        total = float(sum(weights))
        if total <= 0:
            return values[0]
        point = self._rng.random() * total
        cumulative = 0.0
        for value, weight in zip(values, weights):
            cumulative += float(weight)
            if point <= cumulative:
                return value
        return values[-1]

    def _admit(self, record: ReplayRecord) -> bool:
        keys = record.positive_indices
        if not keys:
            return False
        probabilities: List[float] = []
        negative_counts: List[float] = []
        for key in keys:
            count = self.observed_positive_counts[key]
            target = self.capacity * self.target_proportions[key]
            probabilities.append(target / count if count > target else 1.0)
            negative_counts.append(-count)
        maximum = max(negative_counts)
        weights = [math.exp(value - maximum) for value in negative_counts]
        probability = sum(
            item * weight for item, weight in zip(probabilities, weights)
        ) / sum(weights)
        return self._rng.random() < min(max(probability, 0.0), 1.0)

    def _eviction_index(self) -> int:
        deltas = self._deltas()
        if not deltas:
            return 0
        keys = list(deltas)
        maximum = max(deltas.values())
        weights = [math.exp(deltas[key] - maximum) for key in keys]
        selected_key = self._weighted_choice(keys, weights)
        underrepresented = {key for key, delta in deltas.items() if delta <= 0}
        candidates = [
            index
            for index, record in enumerate(self.records)
            if selected_key in record.positive_indices
        ]
        if not candidates:
            raise RuntimeError("PRS selected an empty class partition")
        scores = {
            index: sum(
                key not in self.records[index].positive_indices
                for key in underrepresented
            )
            for index in candidates
        }
        best_score = max(scores.values())
        finalists = [index for index in candidates if scores[index] == best_score]
        # The fixed upstream implementation initializes the comparison with
        # ``(first_candidate, reservoir_size)`` and accepts only strict
        # improvements.  Preserve that tie/fallback behavior exactly.
        best_index = finalists[0]
        best_diff = float(self.capacity)
        for index in finalists:
            remaining = self.records[:index] + self.records[index + 1 :]
            diff = sum(abs(value) for value in self._deltas(remaining).values())
            if diff < best_diff:
                best_index = index
                best_diff = diff
        return best_index

    def _observe_new(self, record: ReplayRecord) -> None:
        self._register_substreams(record)
        if self.capacity <= 0:
            return
        if len(self.records) < self.capacity:
            self.records.append(record)
            return
        if self._admit(record):
            self.records[self._eviction_index()] = record

    def _observe_statistics(self, record: ReplayRecord) -> None:
        self._register_substreams(record)
        for index in record.positive_indices:
            self.observed_positive_counts[index] += 1.0
        self._update_proportions()

    def state_dict(self) -> Dict[str, Any]:
        payload = super().state_dict()
        payload.update(
            {
                "allocation_power": self.allocation_power,
                "observed_positive_counts": list(self.observed_positive_counts),
                "target_proportions": list(self.target_proportions),
                "substream_order": list(self.substream_order),
            }
        )
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if float(payload.get("allocation_power", math.nan)) != self.allocation_power:
            raise ValueError("Replay checkpoint PRS allocation power differs")
        counts = [float(value) for value in payload["observed_positive_counts"]]
        proportions = [float(value) for value in payload["target_proportions"]]
        substream_order = [int(value) for value in payload["substream_order"]]
        if len(counts) != self.num_classes or any(value < 0 for value in counts):
            raise ValueError("Replay checkpoint PRS statistics are invalid")
        if (
            len(proportions) != self.num_classes
            or any(not math.isfinite(value) or value < 0 for value in proportions)
        ):
            raise ValueError("Replay checkpoint PRS proportions are invalid")
        if (
            len(set(substream_order)) != len(substream_order)
            or any(not 0 <= value < self.num_classes for value in substream_order)
        ):
            raise ValueError("Replay checkpoint PRS substream order is invalid")
        super().load_state_dict(payload)
        self.observed_positive_counts = counts
        self.target_proportions = proportions
        self.substream_order = substream_order


def records_to_tensors(
    records: Iterable[ReplayRecord],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = tuple(records)
    if not rows:
        raise ValueError("Cannot tensorize an empty replay batch")
    return (
        torch.stack([row.image for row in rows]),
        torch.stack([row.targets for row in rows]),
        torch.stack([row.visible_mask for row in rows]),
    )
