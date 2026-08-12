"""Protocol-safe views over the existing EMOTIC dataset implementation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.helper_functions.emotic_loader import EMOTIC

from .protocol import BenchmarkProtocol
from .types import EvaluationBatch, EvaluatorAccessToken, TrainBatch


def ordered_split_hash(split: str, sample_ids: Sequence[str]) -> str:
    payload = {"split": split, "sample_ids": list(sample_ids)}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _ProtocolDatasetView(Dataset):
    def __init__(
        self,
        source: Dataset,
        indices: Sequence[int],
        sample_ids: Sequence[str],
        class_indices: Sequence[int],
        total_classes: int,
        class_order_hash: str,
        split: str,
        evaluator_only: bool,
    ) -> None:
        self._source = source
        self.indices = tuple(int(index) for index in indices)
        self._all_sample_ids = tuple(sample_ids)
        self.sample_ids = tuple(self._all_sample_ids[index] for index in self.indices)
        self.class_indices = tuple(int(index) for index in class_indices)
        self.total_classes = int(total_classes)
        self.class_order_hash = class_order_hash
        self.split = split
        self.evaluator_only = bool(evaluator_only)
        self.split_hash = ordered_split_hash(split, self.sample_ids)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        source_index = self.indices[item]
        image, full_target = self._source[source_index]
        target = full_target[list(self.class_indices)].float()
        row = {
            "image": image,
            "sample_id": self._all_sample_ids[source_index],
            "class_order_hash": self.class_order_hash,
            "split_hash": self.split_hash,
        }
        if self.evaluator_only:
            row["targets_seen"] = target
        else:
            visible_mask = torch.zeros(self.total_classes, dtype=torch.bool)
            visible_mask[list(self.class_indices)] = True
            row["targets_current"] = target
            row["visible_mask"] = visible_mask
        return row


class MethodDataLoader:
    """Iterable facade that does not expose the target-bearing source dataset."""

    __slots__ = ("_loader",)

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


def _collate_train(rows: Sequence[Dict[str, Any]]) -> TrainBatch:
    if not rows:
        raise ValueError("Cannot collate an empty training batch")
    return TrainBatch(
        images=_stack_or_pad_images([row["image"] for row in rows]),
        sample_ids=[str(row["sample_id"]) for row in rows],
        targets_current=torch.stack([row["targets_current"] for row in rows]),
        visible_mask=torch.stack([row["visible_mask"] for row in rows]),
    )


def _collate_evaluation(rows: Sequence[Dict[str, Any]]) -> EvaluationBatch:
    if not rows:
        raise ValueError("Cannot collate an empty evaluation batch")
    class_hashes = {row["class_order_hash"] for row in rows}
    split_hashes = {row["split_hash"] for row in rows}
    if len(class_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Evaluation batch metadata is internally inconsistent")
    return EvaluationBatch(
        images=_stack_or_pad_images([row["image"] for row in rows]),
        sample_ids=[str(row["sample_id"]) for row in rows],
        targets_seen=torch.stack([row["targets_seen"] for row in rows]),
        class_order_hash=next(iter(class_hashes)),
        split_hash=next(iter(split_hashes)),
    )


def _stack_or_pad_images(images: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack fixed tensors or bottom/right-pad variable DSCT scene tensors."""

    if not images:
        raise ValueError("Cannot collate an empty image sequence")
    shapes = {tuple(image.shape) for image in images}
    if len(shapes) == 1:
        return torch.stack(list(images))
    if any(image.ndim != 3 for image in images):
        raise ValueError("Variable-size collation supports CHW images only")
    channels = {int(image.shape[0]) for image in images}
    if len(channels) != 1:
        raise ValueError("Variable-size images must have equal channels")
    height = max(int(image.shape[-2]) for image in images)
    width = max(int(image.shape[-1]) for image in images)
    output = images[0].new_zeros((len(images), next(iter(channels)), height, width))
    for index, image in enumerate(images):
        output[index, :, : image.shape[-2], : image.shape[-1]] = image
    return output


class EMOTICMLCILDataModule:
    """Reuses EMOTIC and applies membership/visibility at one boundary."""

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        data_root: str,
        train_transform: Any,
        eval_transform: Any,
        input_mode: str = "full",
    ) -> None:
        if protocol.dataset.lower() != "emotic":
            raise ValueError(
                f"EMOTICMLCILDataModule cannot load dataset {protocol.dataset}"
            )
        self.protocol = protocol
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.input_mode = input_mode
        self._sources: Dict[str, EMOTIC] = {}
        self._sample_ids: Dict[str, Tuple[str, ...]] = {}
        self._dataset_classes_validated = False

    @staticmethod
    def _intersects(target: Sequence[int], class_indices: Sequence[int]) -> bool:
        return bool(set(int(value) for value in target).intersection(class_indices))

    def _source(self, split: str) -> EMOTIC:
        allowed = {
            self.protocol.train_split,
            self.protocol.validation_split,
            self.protocol.test_split,
        }
        if split not in allowed:
            raise ValueError(f"Split '{split}' is not declared by the protocol")
        if split not in self._sources:
            is_train = split == self.protocol.train_split
            source = EMOTIC(
                self.data_root,
                train=is_train,
                transform=self.train_transform if is_train else self.eval_transform,
                eval_splits=(split,),
                input_mode=self.input_mode,
                class_names=(
                    self.protocol.class_order
                    if self._dataset_classes_validated
                    else None
                ),
            )
            if not self._dataset_classes_validated:
                discovered = tuple(source.classes)
                if set(discovered) != set(self.protocol.class_order):
                    missing = sorted(set(self.protocol.class_order) - set(discovered))
                    unexpected = sorted(set(discovered) - set(self.protocol.class_order))
                    raise RuntimeError(
                        "EMOTIC annotations and protocol classes differ: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
                self._dataset_classes_validated = True
                if discovered != self.protocol.class_order:
                    # Reuse the existing loader implementation with an explicit
                    # protocol order for future custom-order configurations.
                    source = EMOTIC(
                        self.data_root,
                        train=is_train,
                        transform=(
                            self.train_transform
                            if is_train
                            else self.eval_transform
                        ),
                        eval_splits=(split,),
                        input_mode=self.input_mode,
                        class_names=self.protocol.class_order,
                    )
            if tuple(source.classes) != self.protocol.class_order:
                raise RuntimeError("EMOTIC loader and protocol class orders differ")
            self._sources[split] = source
            self._sample_ids[split] = self._build_sample_ids(source, split)
        return self._sources[split]

    @staticmethod
    def _build_sample_ids(source: EMOTIC, split: str) -> Tuple[str, ...]:
        if len(source.file_paths) != len(source):
            raise RuntimeError("EMOTIC file path metadata is incomplete")
        occurrences: Dict[str, int] = defaultdict(int)
        sample_ids: List[str] = []
        dataset_root = Path(source.path).resolve()
        for file_path in source.file_paths:
            resolved_path = Path(file_path).resolve()
            try:
                relative_path = resolved_path.relative_to(dataset_root).as_posix()
            except ValueError as exc:
                raise RuntimeError(
                    f"EMOTIC sample is outside dataset root: {resolved_path}"
                ) from exc
            ordinal = occurrences[relative_path]
            occurrences[relative_path] += 1
            sample_ids.append(
                f"emotic:{split}:{relative_path}:person={ordinal}"
            )
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError("Derived EMOTIC sample IDs are not unique")
        return tuple(sample_ids)

    def method_dataset(
        self, task_id: int, split: Optional[str] = None
    ) -> _ProtocolDatasetView:
        """Return current-label-only train or validation-selection data."""

        selected_split = split or self.protocol.train_split
        if selected_split not in {
            self.protocol.train_split,
            self.protocol.validation_split,
        }:
            raise ValueError("Method-facing data is limited to train or validation")
        source = self._source(selected_split)
        current = self.protocol.current_class_indices(task_id)
        indices = [
            index
            for index, target in enumerate(source.targets)
            if self._intersects(target, current)
        ]
        return _ProtocolDatasetView(
            source=source,
            indices=indices,
            sample_ids=self._sample_ids[selected_split],
            class_indices=current,
            total_classes=self.protocol.num_classes,
            class_order_hash=self.protocol.class_order_hash,
            split=selected_split,
            evaluator_only=False,
        )

    def evaluator_dataset(
        self,
        task_id: int,
        split: str,
        access: EvaluatorAccessToken,
    ) -> _ProtocolDatasetView:
        """Return seen targets only when called with an evaluator capability."""

        if not isinstance(access, EvaluatorAccessToken) or not access.is_valid():
            raise PermissionError(
                "Full targets_seen require a BenchmarkEvaluator access token"
            )
        if split not in {
            self.protocol.validation_split,
            self.protocol.test_split,
        }:
            raise ValueError("Evaluator split must be validation or test")
        source = self._source(split)
        seen = self.protocol.seen_class_indices(task_id)
        indices = [
            index
            for index, target in enumerate(source.targets)
            if self._intersects(target, seen)
        ]
        return _ProtocolDatasetView(
            source=source,
            indices=indices,
            sample_ids=self._sample_ids[split],
            class_indices=seen,
            total_classes=self.protocol.num_classes,
            class_order_hash=self.protocol.class_order_hash,
            split=split,
            evaluator_only=True,
        )

    def method_loader(
        self,
        task_id: int,
        batch_size: int,
        num_workers: int,
        split: Optional[str] = None,
        shuffle: Optional[bool] = None,
    ) -> MethodDataLoader:
        selected_split = split or self.protocol.train_split
        dataset = self.method_dataset(task_id, selected_split)
        if shuffle is None:
            shuffle = selected_split == self.protocol.train_split
        loader_options: Dict[str, Any] = {}
        if num_workers > 0:
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=2,
            )
        return MethodDataLoader(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=bool(shuffle),
                num_workers=num_workers,
                pin_memory=(
                    selected_split == self.protocol.train_split
                    or self.input_mode == "dsct_scene"
                ),
                drop_last=False,
                collate_fn=_collate_train,
                **loader_options,
            )
        )

    def evaluator_loader(
        self,
        task_id: int,
        split: str,
        access: EvaluatorAccessToken,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        dataset = self.evaluator_dataset(task_id, split, access)
        loader_options: Dict[str, Any] = {}
        if num_workers > 0:
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=2,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.input_mode == "dsct_scene",
            drop_last=False,
            collate_fn=_collate_evaluation,
            **loader_options,
        )
