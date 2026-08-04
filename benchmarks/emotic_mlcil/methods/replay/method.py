"""Budget-controlled ER and PRS methods for EMOTIC Track A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F

from ...protocol import BenchmarkProtocol
from ...registry import register_method
from ...replay_memory import (
    PartitionedReservoirReplayBuffer,
    ReplayBuffer,
    ReplayMemoryContract,
    ReplayRecord,
    ReservoirReplayBuffer,
    records_to_tensors,
)
from ...types import MemoryStatistics, TaskContext, TrainBatch
from ..clip_classifier.method import (
    CLIPClassifierOptions,
    CLIPContinualMethod,
    _validate_train_batch,
)
from ..clip_classifier.model import GrowingMultiLabelClassifier


_DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "emotic_mlcil"
    / "replay_20c_v0.1.yaml"
)


@dataclass(frozen=True)
class ReplayOptions:
    prs_allocation_power: float = -0.03

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayOptions":
        options = cls(
            prs_allocation_power=float(value.get("prs_allocation_power", -0.03))
        )
        if not math.isfinite(options.prs_allocation_power):
            raise ValueError("PRS allocation power must be finite")
        return options


class ReplayContinualMethod(CLIPContinualMethod):
    """Common training path; subclasses change only memory update policy."""

    method_name = "Replay"
    method_family = "Replay"
    strategy = "replay"
    buffer_class: Type[ReplayBuffer] = ReservoirReplayBuffer
    upstream_repository = "Repository-native controlled replay baseline"
    upstream_commit = "N/A"
    source_original_capacity: Optional[int] = None
    source_original_update_timing = "N/A"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        clip_model_path: Union[str, Path] = "./pretrained/clip/ViT-B-16.pt",
        device: Optional[Union[str, torch.device]] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        model: Optional[GrowingMultiLabelClassifier] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
        memory_contract_path: Optional[Union[str, Path]] = None,
        memory_contract: Optional[ReplayMemoryContract] = None,
    ) -> None:
        overrides = dict(option_overrides or {})
        classifier_fields = set(CLIPClassifierOptions.__dataclass_fields__)
        replay_fields = set(ReplayOptions.__dataclass_fields__)
        unknown = sorted(set(overrides).difference(classifier_fields | replay_fields))
        if unknown:
            raise ValueError(
                "Unknown replay option override(s): " + ", ".join(unknown)
            )
        classifier_overrides = {
            key: value for key, value in overrides.items() if key in classifier_fields
        }
        replay_overrides = {
            key: value for key, value in overrides.items() if key in replay_fields
        }
        super().__init__(
            protocol=protocol,
            clip_model_path=clip_model_path,
            device=device,
            feature_extractor=feature_extractor,
            model=model,
            option_overrides=classifier_overrides,
        )
        self.replay_options = ReplayOptions.from_mapping(replay_overrides)
        if memory_contract is not None and memory_contract_path is not None:
            raise ValueError("Provide replay memory_contract or path, not both")
        if memory_contract is None:
            contract_path = Path(memory_contract_path or _DEFAULT_CONTRACT)
            self.memory_contract_path = str(contract_path.resolve())
            memory_contract = ReplayMemoryContract.from_yaml(
                contract_path,
                protocol,
            )
        else:
            self.memory_contract_path = "injected"
            if memory_contract.protocol_id != protocol.protocol_id:
                raise ValueError("Injected replay contract protocol_id differs")
            if len(memory_contract.task_capacities) != protocol.num_tasks:
                raise ValueError("Injected replay contract task count differs")
        self.memory_contract = memory_contract
        self.replay_memory = self._make_replay_memory()
        self.optimizer_attempts = 0
        self.amp_overflow_skips = 0

    def _make_replay_memory(self) -> ReplayBuffer:
        if self.buffer_class is PartitionedReservoirReplayBuffer:
            return PartitionedReservoirReplayBuffer(
                num_classes=len(self.protocol.class_order),
                seed=self.protocol.seed,
                allocation_power=self.replay_options.prs_allocation_power,
            )
        return self.buffer_class(
            num_classes=len(self.protocol.class_order),
            seed=self.protocol.seed,
        )

    def begin_task(self, task_context: TaskContext) -> None:
        super().begin_task(task_context)
        self.replay_memory.set_capacity(
            self.memory_contract.capacity_for_context(task_context)
        )
        self.optimizer_attempts = 0
        self.amp_overflow_skips = 0

    @staticmethod
    def _per_sample_current_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        ).mean(dim=1)

    @staticmethod
    def _per_sample_replay_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(visible_mask.any(dim=1).all()):
            raise ValueError("Every replay sample must expose at least one label")
        losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        mask = visible_mask.to(losses.dtype)
        return (losses * mask).sum(dim=1) / mask.sum(dim=1)

    def _sample_replay(
        self,
        current_batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requested = max(
            1,
            int(round(current_batch_size * self.memory_contract.replay_to_current_ratio)),
        )
        records = self.replay_memory.sample(requested)
        if not records:
            return (
                torch.empty(0),
                torch.empty(0),
                torch.empty(0, dtype=torch.bool),
            )
        return records_to_tensors(records)

    def _update_memory(self, train_loader: Iterable[TrainBatch]) -> int:
        if self.task_context is None:
            raise RuntimeError("Replay update requires an active task")
        observed = 0
        total_classes = len(self.protocol.class_order)
        current_indices = list(self.task_context.current_class_indices)
        for raw_batch in train_loader:
            batch = _validate_train_batch(raw_batch, self.task_context)
            if len(batch.sample_ids) != batch.images.shape[0]:
                raise ValueError("Replay batch sample IDs do not align with images")
            for row, sample_id in enumerate(batch.sample_ids):
                targets = torch.zeros(total_classes, dtype=torch.float32)
                targets[current_indices] = batch.targets_current[row].detach().cpu().float()
                visible_mask = batch.visible_mask[row].detach().cpu().bool()
                record = ReplayRecord(
                    image=batch.images[row],
                    sample_id=sample_id,
                    targets=targets,
                    visible_mask=visible_mask,
                )
                self.replay_memory.observe(record)
                observed += 1
        if observed == 0:
            raise ValueError("Replay update loader produced no samples")
        return observed

    def train_task(
        self,
        train_loader: Iterable[TrainBatch],
        val_loader: Iterable[TrainBatch],
    ) -> None:
        if self.task_context is None:
            raise RuntimeError("begin_task must be called before train_task")
        optimizer = self._optimizer()
        scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        best_map = -math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0
        replay_samples_before = len(self.replay_memory)

        for epoch in range(self.options.epochs):
            self.model.train()
            current_total = 0.0
            replay_total = 0.0
            replay_examples = 0
            batches = 0
            for raw_batch in train_loader:
                batch = _validate_train_batch(raw_batch, self.task_context)
                images = self._prepare_images(batch.images)
                targets = batch.targets_current.to(
                    self.device,
                    non_blocking=True,
                ).float()
                replay_images, replay_targets, replay_mask = self._sample_replay(
                    images.shape[0]
                )
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    current_logits = self.model.current_logits(images)
                    current_losses = self._per_sample_current_loss(
                        current_logits,
                        targets,
                    )
                    if replay_images.numel() > 0:
                        replay_images = self._prepare_images(replay_images)
                        seen_indices = list(self.task_context.seen_class_indices)
                        replay_targets = replay_targets[:, seen_indices].to(
                            self.device,
                            non_blocking=True,
                        )
                        replay_mask = replay_mask[:, seen_indices].to(
                            self.device,
                            non_blocking=True,
                        )
                        replay_logits = self.model(replay_images)
                        replay_losses = self._per_sample_replay_loss(
                            replay_logits,
                            replay_targets,
                            replay_mask,
                        )
                        loss = torch.cat([current_losses, replay_losses]).mean()
                    else:
                        replay_losses = torch.empty(0, device=self.device)
                        loss = current_losses.mean()
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.options.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                self.optimizer_attempts += 1
                if float(scaler.get_scale()) < scale_before:
                    self.amp_overflow_skips += 1

                current_total += float(current_losses.mean().detach().cpu())
                if replay_losses.numel() > 0:
                    replay_total += float(replay_losses.mean().detach().cpu())
                    replay_examples += int(replay_losses.numel())
                batches += 1
            if batches == 0:
                raise ValueError("Training loader produced no samples")
            selection_map = self._selection_map(val_loader)
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "current_loss": current_total / batches,
                    "replay_loss": replay_total / batches,
                    "replay_examples": float(replay_examples),
                    "selection_mAP": selection_map,
                    "replay_samples_before": float(replay_samples_before),
                }
            )
            if selection_map > best_map:
                best_map = selection_map
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.options.early_stopping_patience:
                    break

        if best_state is None:
            raise RuntimeError("No replay training epoch produced a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.to(self.device)
        observed = self._update_memory(train_loader)
        self.training_history[-1].update(
            {
                "memory_update_observations": float(observed),
                "replay_samples_after": float(len(self.replay_memory)),
                "replay_bytes_after": float(
                    self.replay_memory.memory_statistics().replay_memory_bytes
                ),
                "optimizer_attempts": float(self.optimizer_attempts),
                "amp_overflow_skips": float(self.amp_overflow_skips),
            }
        )
        self._completed_task_id = self.task_context.task_id

    def resolved_method_config(self) -> Mapping[str, Any]:
        payload = dict(super().resolved_method_config())
        payload.update(
            {
                "strategy": self.strategy,
                "replay_policy": self.replay_memory.policy_name,
                "replay_contract": self.memory_contract.as_dict(),
                "replay_contract_path": self.memory_contract_path,
                "prs_allocation_power": self.replay_options.prs_allocation_power,
                "upstream_repository": self.upstream_repository,
                "upstream_commit": self.upstream_commit,
                "source_original_capacity": self.source_original_capacity,
                "source_original_update_timing": self.source_original_update_timing,
                "training_lifecycle": "task_based_replay_then_task_end_update",
                "current_and_replay_equal_sample_weight": True,
                "replay_loss": "visible_masked_sigmoid_bce",
                "track_a_primary_budget_differs_from_prs_source": (
                    self.strategy == "prs"
                ),
            }
        )
        return payload

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        super().save_checkpoint(path)
        destination = Path(path)
        payload = torch.load(destination, map_location="cpu")
        payload.update(
            {
                "replay_contract": self.memory_contract.as_dict(),
                "replay_options": self.replay_options.__dict__,
                "replay_memory": self.replay_memory.state_dict(),
                "optimizer_attempts": self.optimizer_attempts,
                "amp_overflow_skips": self.amp_overflow_skips,
            }
        )
        torch.save(payload, destination)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        super().load_checkpoint(path)
        payload = torch.load(Path(path), map_location="cpu")
        if dict(payload.get("replay_contract", {})) != self.memory_contract.as_dict():
            raise ValueError("Checkpoint replay memory contract differs")
        if dict(payload.get("replay_options", {})) != self.replay_options.__dict__:
            raise ValueError("Checkpoint replay options differ")
        self.replay_memory.load_state_dict(payload["replay_memory"])
        self.optimizer_attempts = int(payload.get("optimizer_attempts", 0))
        self.amp_overflow_skips = int(payload.get("amp_overflow_skips", 0))

    def memory_statistics(self) -> MemoryStatistics:
        return self.replay_memory.memory_statistics()


class ERBenchmarkMethod(ReplayContinualMethod):
    method_name = "ER"
    method_family = "Replay Control"
    strategy = "er"
    buffer_class = ReservoirReplayBuffer


class PRSBenchmarkMethod(ReplayContinualMethod):
    method_name = "PRS"
    method_family = "Native Multi-Label Replay"
    strategy = "prs"
    buffer_class = PartitionedReservoirReplayBuffer
    upstream_repository = "https://github.com/cdjkim/PRS"
    upstream_commit = "136cee1863af03cc914dc05dfd41bda8b7bc0bf2"
    source_original_capacity = 2000
    source_original_update_timing = "online_after_each_optimizer_step"


register_method("er", ERBenchmarkMethod)
register_method("prs", PRSBenchmarkMethod)
