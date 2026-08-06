"""Task-routed Transformer Adapter Bank for EMOTIC B5-C3.

The Adapter layout follows the Continual-Adapter block used by P2L-CA: a
down projection, ReLU, and up projection operating in parallel with the
frozen ViT MLP.  This project extends that layout to a deterministic
class-introduction-task bank.  Both positive and negative DDP paths of class
``c`` are routed to the Adapter learned when ``c`` was introduced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from emotic_task_adapter_bank import (
    TASK_CLASS_RANGES,
    class_task_id,
    class_to_task_map,
    file_sha256,
    task_class_range,
)


# Paper notation is one based (layers 4--12).  PyTorch ModuleList indices are
# zero based, hence 3--11 here.
P2L_CA_LAYER_NUMBERS: Tuple[int, ...] = tuple(range(4, 13))
P2L_CA_LAYER_INDICES: Tuple[int, ...] = tuple(number - 1 for number in P2L_CA_LAYER_NUMBERS)
TRANSFORMER_ADAPTER_SCHEMA_VERSION = 1
TRANSFORMER_BANK_SCHEMA_VERSION = 1


class TransformerBlockAdapter(nn.Module):
    """P2L-CA-style bottleneck branch returning a feature delta only."""

    def __init__(self, hidden_dim: int = 768, bottleneck_dim: int = 128):
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim and bottleneck_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.down = nn.Linear(self.hidden_dim, self.bottleneck_dim, bias=True)
        self.activation = nn.ReLU()
        self.up = nn.Linear(self.bottleneck_dim, self.hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        # Exact identity at initialization: the optional Adapter branch emits
        # zero while the original frozen DDP path remains untouched.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, normalized_tokens: torch.Tensor) -> torch.Tensor:
        if normalized_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected token width {self.hidden_dim}, got "
                f"{normalized_tokens.shape[-1]}"
            )
        return self.up(self.activation(self.down(normalized_tokens)))


class TransformerTaskAdapter(nn.Module):
    """The nine block Adapters owned by one incremental task."""

    def __init__(
        self,
        hidden_dim: int = 768,
        bottleneck_dim: int = 128,
        layer_indices: Sequence[int] = P2L_CA_LAYER_INDICES,
    ):
        super().__init__()
        normalized_layers = tuple(int(index) for index in layer_indices)
        if not normalized_layers or len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError("layer_indices must be non-empty and unique")
        if tuple(sorted(normalized_layers)) != normalized_layers:
            raise ValueError("layer_indices must be sorted")
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.layer_indices = normalized_layers
        self.blocks = nn.ModuleDict(
            {
                str(layer_index): TransformerBlockAdapter(
                    hidden_dim=self.hidden_dim,
                    bottleneck_dim=self.bottleneck_dim,
                )
                for layer_index in self.layer_indices
            }
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def has_layer(self, layer_index: int) -> bool:
        return str(int(layer_index)) in self.blocks

    def forward_layer(
        self, layer_index: int, normalized_tokens: torch.Tensor
    ) -> torch.Tensor:
        key = str(int(layer_index))
        if key not in self.blocks:
            return torch.zeros_like(normalized_tokens)
        return self.blocks[key](normalized_tokens)


class TaskRoutedTransformerAdapterBank(nn.Module):
    """Route ViT token paths by class-introduction task at every block."""

    routing_mode = "class_introduction_task"
    adapter_location = "parallel_to_vit_mlp"

    def __init__(
        self,
        adapters: Mapping[int, TransformerTaskAdapter],
        classification_loss: str = "asl",
        loss_config_sha256: Optional[str] = None,
        require_contiguous: bool = True,
    ):
        super().__init__()
        if not adapters:
            raise ValueError("Transformer Adapter Bank cannot be empty")
        task_ids = sorted(int(task_id) for task_id in adapters)
        if require_contiguous and task_ids != list(range(task_ids[-1] + 1)):
            raise ValueError(
                "Inference bank tasks must be contiguous from task 0; got "
                f"{task_ids}"
            )
        reference = adapters[task_ids[0]]
        for task_id in task_ids[1:]:
            candidate = adapters[task_id]
            if candidate.hidden_dim != reference.hidden_dim:
                raise ValueError("All task Adapters must use one hidden dimension")
            if candidate.bottleneck_dim != reference.bottleneck_dim:
                raise ValueError("All task Adapters must use one bottleneck")
            if candidate.layer_indices != reference.layer_indices:
                raise ValueError("All task Adapters must use the same layers")
        self.adapters = nn.ModuleDict(
            {str(task_id): adapters[task_id] for task_id in task_ids}
        )
        self.hidden_dim = reference.hidden_dim
        self.bottleneck_dim = reference.bottleneck_dim
        self.layer_indices = reference.layer_indices
        self.classification_loss = str(classification_loss)
        self.loss_config_sha256 = loss_config_sha256

    @property
    def max_task(self) -> int:
        return max(int(key) for key in self.adapters)

    @property
    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def freeze(self) -> "TaskRoutedTransformerAdapterBank":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def delta_for_layer(
        self,
        layer_index: int,
        normalized_tokens: torch.Tensor,
        path_task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return routed Adapter deltas for ``[tokens, paths, width]``."""

        if normalized_tokens.ndim != 3:
            raise ValueError("normalized_tokens must be [tokens, paths, width]")
        if normalized_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden width {self.hidden_dim}, got "
                f"{normalized_tokens.shape[-1]}"
            )
        if path_task_ids.ndim != 1 or path_task_ids.numel() != normalized_tokens.shape[1]:
            raise ValueError(
                "path_task_ids must contain one task id per DDP path; got "
                f"{tuple(path_task_ids.shape)} for {normalized_tokens.shape[1]} paths"
            )
        if int(layer_index) not in self.layer_indices:
            return torch.zeros_like(normalized_tokens)

        output = torch.zeros_like(normalized_tokens)
        unique_tasks = torch.unique(path_task_ids.detach()).tolist()
        for raw_task_id in unique_tasks:
            task_id = int(raw_task_id)
            key = str(task_id)
            if key not in self.adapters:
                raise ValueError(
                    f"No Transformer Adapter is loaded for routed task {task_id}"
                )
            path_indices = torch.nonzero(
                path_task_ids.eq(task_id), as_tuple=False
            ).flatten()
            selected = normalized_tokens.index_select(1, path_indices)
            delta = self.adapters[key].forward_layer(layer_index, selected)
            output = output.index_copy(1, path_indices, delta)
        return output

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        device: Union[str, torch.device],
        max_task: Optional[int] = None,
        classnames: Optional[Sequence[str]] = None,
    ) -> "TaskRoutedTransformerAdapterBank":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != TRANSFORMER_BANK_SCHEMA_VERSION:
            raise ValueError("Unsupported Transformer Adapter Bank schema")
        if manifest.get("adapter_location") != cls.adapter_location:
            raise ValueError("Manifest is not a Transformer-block Adapter Bank")
        if manifest.get("routing_mode") != cls.routing_mode:
            raise ValueError("Manifest uses an unsupported routing mode")
        if classnames is not None and list(manifest.get("classnames", [])) != list(classnames):
            raise ValueError("Transformer Bank and DDP class orders differ")

        available = sorted(int(key) for key in manifest.get("adapters", {}))
        if not available:
            raise ValueError("Manifest contains no Transformer Adapters")
        resolved_max = available[-1] if max_task is None else int(max_task)
        expected = list(range(resolved_max + 1))
        if not all(task_id in available for task_id in expected):
            raise ValueError(
                f"Manifest does not contain every task in {expected}; got {available}"
            )

        architecture = manifest["architecture"]
        adapters: Dict[int, TransformerTaskAdapter] = {}
        for task_id in expected:
            entry = manifest["adapters"][str(task_id)]
            checkpoint_path = manifest_path.parent / entry["checkpoint"]
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if file_sha256(checkpoint_path) != entry.get("sha256"):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            validate_transformer_adapter_checkpoint(
                checkpoint,
                task_id=task_id,
                training_mode=manifest["training_mode"],
                seed=manifest["seed"],
                classnames=manifest["classnames"],
                classification_loss=manifest["classification_loss"],
                loss_config_sha256=manifest["loss_config"]["sha256"],
            )
            adapter = TransformerTaskAdapter(
                hidden_dim=int(architecture["hidden_dim"]),
                bottleneck_dim=int(architecture["bottleneck_dim"]),
                layer_indices=architecture["layer_indices"],
            )
            adapter.load_state_dict(checkpoint["model"], strict=True)
            adapters[task_id] = adapter.to(device)
        return cls(
            adapters,
            classification_loss=manifest["classification_loss"],
            loss_config_sha256=manifest["loss_config"]["sha256"],
        ).to(device).freeze()


def transformer_adapter_metadata(checkpoint: Mapping) -> Mapping:
    metadata = checkpoint.get("transformer_task_adapter")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint lacks transformer_task_adapter metadata")
    return metadata


def validate_transformer_adapter_checkpoint(
    checkpoint: Mapping,
    task_id: Optional[int] = None,
    training_mode: Optional[str] = None,
    seed: Optional[int] = None,
    classnames: Optional[Sequence[str]] = None,
    classification_loss: Optional[str] = None,
    loss_config_sha256: Optional[str] = None,
) -> Mapping:
    metadata = transformer_adapter_metadata(checkpoint)
    checkpoint_task = int(metadata["task_id"])
    if int(metadata.get("schema_version", -1)) != TRANSFORMER_ADAPTER_SCHEMA_VERSION:
        raise ValueError("Unsupported Transformer Adapter checkpoint schema")
    if list(metadata.get("class_range", [])) != list(task_class_range(checkpoint_task)):
        raise ValueError("Transformer Adapter checkpoint has an invalid class range")
    checks = (
        (task_id, checkpoint_task, "task"),
        (training_mode, metadata.get("training_mode"), "training mode"),
        (seed, int(metadata.get("seed", -1)), "seed"),
        (classification_loss, metadata.get("classification_loss"), "loss"),
        (loss_config_sha256, metadata.get("loss_config", {}).get("sha256"), "loss hash"),
    )
    for expected, actual, label in checks:
        if expected is not None and expected != actual:
            raise ValueError(f"Expected {label} {expected}, found {actual}")
    if classnames is not None and list(checkpoint.get("classnames", [])) != list(classnames):
        raise ValueError("Transformer Adapter and DDP class orders differ")
    architecture = metadata.get("architecture", {})
    if tuple(architecture.get("layer_indices", ())) != P2L_CA_LAYER_INDICES:
        raise ValueError("Transformer Adapter must use P2L-CA layers 4--12")
    if int(architecture.get("hidden_dim", -1)) != 768:
        raise ValueError("Transformer Adapter hidden_dim must be 768")
    if int(architecture.get("bottleneck_dim", -1)) != 128:
        raise ValueError("Transformer Adapter bottleneck_dim must be 128")
    if checkpoint.get("model") is None:
        raise ValueError("Transformer Adapter checkpoint has no model state")
    return metadata


def build_transformer_bank_manifest(
    bank_dir: Union[str, Path],
    training_mode: str,
    seed: int,
    classnames: Sequence[str],
    checkpoint_filename: str = "last_transformer_adapter.pth",
) -> dict:
    bank_dir = Path(bank_dir)
    entries = {}
    shared_loss = None
    shared_loss_config = None
    shared_architecture = None
    for task_id in range(len(TASK_CLASS_RANGES)):
        checkpoint_path = bank_dir / f"task{task_id}" / checkpoint_filename
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = validate_transformer_adapter_checkpoint(
            checkpoint,
            task_id=task_id,
            training_mode=training_mode,
            seed=seed,
            classnames=classnames,
        )
        loss_name = metadata["classification_loss"]
        loss_config = metadata["loss_config"]
        architecture = metadata["architecture"]
        if shared_loss is None:
            shared_loss = loss_name
            shared_loss_config = loss_config
            shared_architecture = architecture
        if loss_name != shared_loss or loss_config != shared_loss_config:
            raise ValueError("Task checkpoints use different loss configurations")
        if architecture != shared_architecture:
            raise ValueError("Task checkpoints use different Adapter architectures")
        entries[str(task_id)] = {
            "task_id": task_id,
            "class_range": list(task_class_range(task_id)),
            "checkpoint": str(checkpoint_path.relative_to(bank_dir)),
            "sha256": file_sha256(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "reporting_val_mAP": checkpoint.get("reporting_val_mAP"),
            "ddp_checkpoint": metadata.get("ddp_checkpoint"),
            "initialization": metadata.get("initialization"),
        }
    return {
        "schema_version": TRANSFORMER_BANK_SCHEMA_VERSION,
        "name": "EMOTIC B5-C3 task-routed Transformer Adapter Bank",
        "training_mode": training_mode,
        "seed": int(seed),
        "classification_loss": shared_loss,
        "loss_config": shared_loss_config,
        "checkpoint_rule": "last_epoch",
        "routing_mode": TaskRoutedTransformerAdapterBank.routing_mode,
        "adapter_location": TaskRoutedTransformerAdapterBank.adapter_location,
        "architecture": shared_architecture,
        "decision_threshold": 0.5,
        "test_used_for_selection": False,
        "validation_used_for_checkpoint_selection": False,
        "classnames": list(classnames),
        "class_to_task": list(class_to_task_map()),
        "adapters": entries,
    }


def path_task_ids_for_classes(
    class_ids: Sequence[int], batch_size: int, device: torch.device
) -> torch.Tensor:
    """Match DDP's per-sample ``[negative classes, positive classes]`` order."""

    routed = torch.tensor(
        [class_task_id(class_id) for class_id in class_ids],
        dtype=torch.long,
        device=device,
    )
    per_sample = torch.cat([routed, routed], dim=0)
    return per_sample.unsqueeze(0).expand(int(batch_size), -1).reshape(-1)
