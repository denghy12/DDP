"""EMOT-Net+CCIM converted to protocol-safe sequential fine-tuning."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch

from ...protocol import BenchmarkProtocol
from ...registry import register_method
from ...types import MemoryStatistics, ParameterStatistics
from ..emot_net_ft.method import EMOTNetFTBenchmarkMethod, EMOTNetFTOptions
from .model import (
    CCIM_SOURCE_SHA256,
    CCIM_UPSTREAM_COMMIT,
    CCIM_UPSTREAM_REPOSITORY,
    EMOTNetCCIMFTModel,
)


CCIM_DICTIONARY_SCHEMA_VERSION = 1
CCIM_DICTIONARY_SCOPE = "task0_train_current_accessible_frozen"
CCIM_FEATURE_EXTRACTOR = "ResNet152-Places365-last-pool"


@dataclass(frozen=True)
class EMOTNetCCIMFTOptions(EMOTNetFTOptions):
    ccim_hidden_dim: int = 128
    ccim_attention_dim: int = 256
    ccim_confounder_dim: int = 2048
    ccim_dictionary_size: int = 256
    ccim_strategy: str = "dp_cause"
    ccim_dictionary_scope: str = CCIM_DICTIONARY_SCOPE

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EMOTNetCCIMFTOptions":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError(
                "Unknown EMOT-Net+CCIM-FT option(s): " + ", ".join(unknown)
            )
        options = cls(**{key: value[key] for key in value})
        if min(
            options.fusion_dim,
            options.epochs,
            options.early_stopping_patience,
            options.lr_drop_epoch,
        ) <= 0:
            raise ValueError("EMOT-Net dimensions and epoch settings must be positive")
        if options.learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("EMOT-Net optimizer settings are invalid")
        if not 0 <= options.momentum < 1:
            raise ValueError("EMOT-Net momentum must lie in [0, 1)")
        if not 0 < options.lr_drop_gamma <= 1:
            raise ValueError("EMOT-Net lr_drop_gamma must lie in (0, 1]")
        if not 0 <= options.dropout < 1 or options.norm_factor <= 1:
            raise ValueError("EMOT-Net dropout or norm_factor is invalid")
        if options.gradient_clip_norm <= 0:
            raise ValueError("EMOT-Net gradient clipping must be positive")
        if options.discrete_loss_weight <= 0:
            raise ValueError("EMOT-Net discrete loss weight must be positive")
        if min(
            options.ccim_hidden_dim,
            options.ccim_attention_dim,
            options.ccim_confounder_dim,
            options.ccim_dictionary_size,
        ) <= 0:
            raise ValueError("CCIM dimensions and dictionary size must be positive")
        if options.ccim_strategy != "dp_cause":
            raise ValueError("The frozen EMOT-Net+CCIM baseline requires dp_cause")
        if options.ccim_dictionary_scope != CCIM_DICTIONARY_SCOPE:
            raise ValueError("CCIM dictionary must be frozen from Task-0-accessible train data")
        return options


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_ccim_dictionary_payload(
    payload: Mapping[str, Any],
    protocol: BenchmarkProtocol,
    options: EMOTNetCCIMFTOptions,
) -> None:
    expected = {
        "schema_version": CCIM_DICTIONARY_SCHEMA_VERSION,
        "resource": "EMOT-Net+CCIM confounder dictionary",
        "protocol_id": protocol.protocol_id,
        # Dictionary construction is intentionally invariant across run seeds.
        # All formal seeds consume the same Task-0 resource rather than gaining
        # an extra K-Means source of variation.
        "protocol_hash": protocol.with_seed(0).protocol_hash,
        "class_order_hash": protocol.class_order_hash,
        "task_id": 0,
        "scope": options.ccim_dictionary_scope,
        "strategy": options.ccim_strategy,
        "feature_extractor": CCIM_FEATURE_EXTRACTOR,
        "dictionary_size": options.ccim_dictionary_size,
        "confounder_dim": options.ccim_confounder_dim,
        "construction_seed": 0,
        "ccim_repository": CCIM_UPSTREAM_REPOSITORY,
        "ccim_commit": CCIM_UPSTREAM_COMMIT,
        "ccim_source_sha256": CCIM_SOURCE_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"CCIM dictionary {key} differs from the frozen contract")
    dictionary = payload.get("dictionary")
    prior = payload.get("prior")
    if not isinstance(dictionary, torch.Tensor) or dictionary.shape != (
        options.ccim_dictionary_size,
        options.ccim_confounder_dim,
    ):
        raise ValueError("CCIM dictionary tensor has the wrong shape")
    if not isinstance(prior, torch.Tensor) or prior.shape not in {
        (options.ccim_dictionary_size,),
        (options.ccim_dictionary_size, 1),
    }:
        raise ValueError("CCIM prior tensor has the wrong shape")
    if not isinstance(payload.get("sample_ids_sha256"), str):
        raise ValueError("CCIM dictionary is missing its Task-0 sample-ID hash")
    feature_checkpoint_sha = payload.get("feature_checkpoint_sha256")
    if not isinstance(feature_checkpoint_sha, str) or len(feature_checkpoint_sha) != 64:
        raise ValueError("CCIM dictionary is missing the Places365 checkpoint hash")


class EMOTNetCCIMFTBenchmarkMethod(EMOTNetFTBenchmarkMethod):
    """Original EMOT-Net+CCIM architecture under no-memory sequential FT."""

    method_name = "EMOT-Net+CCIM-FT"
    method_family = "Static EMOTIC model / Sequential Fine-Tuning"
    backbone = "Native EMOT-Net + CCIM (ResNet152-Places365 confounders)"
    supported_tracks = ("B",)
    upstream_repository = CCIM_UPSTREAM_REPOSITORY
    upstream_commit = CCIM_UPSTREAM_COMMIT
    upstream_license = "MIT"

    def __init__(
        self,
        protocol: BenchmarkProtocol,
        native_initialization_path: Optional[Union[str, Path]] = None,
        ccim_dictionary_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[EMOTNetCCIMFTModel] = None,
        dictionary_payload: Optional[Mapping[str, Any]] = None,
        option_overrides: Optional[Mapping[str, Any]] = None,
        tower_model_parallel: bool = False,
    ) -> None:
        configured = dict(protocol.method_options("emot_net_ccim_ft"))
        if option_overrides:
            configured.update(dict(option_overrides))
        full_options = EMOTNetCCIMFTOptions.from_mapping(configured)
        dictionary_path = (
            Path(ccim_dictionary_path).expanduser().resolve()
            if ccim_dictionary_path is not None
            else None
        )
        if dictionary_payload is None:
            if dictionary_path is None or not dictionary_path.is_file():
                raise FileNotFoundError(
                    "EMOT-Net+CCIM-FT requires a protocol-audited Task-0 confounder dictionary"
                )
            dictionary_payload = torch.load(dictionary_path, map_location="cpu")
        if not isinstance(dictionary_payload, Mapping):
            raise ValueError("CCIM dictionary resource must be a mapping")
        validate_ccim_dictionary_payload(dictionary_payload, protocol, full_options)

        native_path = (
            Path(native_initialization_path).expanduser().resolve()
            if native_initialization_path is not None
            else None
        )
        native_sha: Optional[str] = None
        native_assets: Dict[str, str] = {}
        if model is None:
            if native_path is None or not native_path.is_file():
                raise FileNotFoundError(
                    "EMOT-Net+CCIM-FT requires the audited native EMOT-Net initialization"
                )
            model = EMOTNetCCIMFTModel(
                dictionary_payload["dictionary"],
                dictionary_payload["prior"],
                fusion_dim=full_options.fusion_dim,
                ccim_hidden_dim=full_options.ccim_hidden_dim,
                ccim_attention_dim=full_options.ccim_attention_dim,
                ccim_strategy=full_options.ccim_strategy,
                dropout=full_options.dropout,
            )
            native_payload = torch.load(native_path, map_location="cpu")
            if not isinstance(native_payload, Mapping):
                raise ValueError("EMOT-Net native initialization must be a mapping")
            model.load_native_initialization(native_payload)
            native_sha = _sha256(native_path)
            native_assets = {
                str(name): str(value)
                for name, value in dict(native_payload["source_assets"]).items()
            }
        if model.dictionary_size != full_options.ccim_dictionary_size:
            raise ValueError("CCIM model dictionary size differs from options")
        if model.confounder_dim != full_options.ccim_confounder_dim:
            raise ValueError("CCIM model confounder width differs from options")
        if model.ccim_hidden_dim != full_options.ccim_hidden_dim:
            raise ValueError("CCIM model hidden width differs from options")

        base_fields = set(EMOTNetFTOptions.__dataclass_fields__)
        base_options = {
            key: value
            for key, value in asdict(full_options).items()
            if key in base_fields
        }
        super().__init__(
            protocol=protocol,
            native_initialization_path=native_path,
            device=device,
            model=model,
            option_overrides=base_options,
            tower_model_parallel=tower_model_parallel,
        )
        self.options = full_options
        if native_sha is not None:
            self.native_initialization_sha256 = native_sha
            self.native_source_asset_sha256 = native_assets
        self.ccim_dictionary_path = str(dictionary_path) if dictionary_path else None
        self.ccim_dictionary_sha256 = (
            _sha256(dictionary_path) if dictionary_path is not None else None
        )
        self.ccim_dictionary_provenance = {
            key: dictionary_payload[key]
            for key in (
                "sample_ids_sha256",
                "feature_checkpoint_sha256",
                "feature_checkpoint_source",
                "construction_seed",
                "scope",
            )
            if key in dictionary_payload
        }

    def resolved_method_config(self) -> Mapping[str, Any]:
        base = dict(super().resolved_method_config())
        base.update(
            {
                "strategy": "sequential_finetuning",
                "conversion_interface": "EMOT-Net+CCIM-FT-v0.1",
                "host_repository": "https://github.com/rkosti/emotic",
                "host_commit": "69c3a5106aed08121cd12f6a5b359c745136931e",
                "ccim_repository": CCIM_UPSTREAM_REPOSITORY,
                "ccim_commit": CCIM_UPSTREAM_COMMIT,
                "ccim_source_sha256": CCIM_SOURCE_SHA256,
                "ccim_dictionary_path": self.ccim_dictionary_path,
                "ccim_dictionary_sha256": self.ccim_dictionary_sha256,
                "ccim_dictionary_provenance": dict(self.ccim_dictionary_provenance),
                "ccim_dictionary_frozen_after_task0": True,
                "ccim_future_task_images_used": False,
                "ccim_future_labels_used": False,
                "ccim_static_full_train_dictionary_rejected": True,
                "ccim_static_full_train_dictionary_rejected_reason": (
                    "would inspect future-task training images"
                ),
                "persistent_auxiliary_bytes": (
                    self.model.confounder_dictionary.numel()
                    * self.model.confounder_dictionary.element_size()
                    + self.model.confounder_prior.numel()
                    * self.model.confounder_prior.element_size()
                ),
                **asdict(self.options),
            }
        )
        return base

    def checkpoint_extra_metadata(self) -> Mapping[str, Any]:
        return {
            "ccim_options": {
                key: value
                for key, value in asdict(self.options).items()
                if key.startswith("ccim_")
            },
            "ccim_dictionary_sha256": self.ccim_dictionary_sha256,
            "ccim_dictionary_provenance": dict(self.ccim_dictionary_provenance),
        }

    def validate_checkpoint_extra_metadata(self, value: Mapping[str, Any]) -> None:
        if dict(value) != dict(self.checkpoint_extra_metadata()):
            raise ValueError("Checkpoint CCIM dictionary provenance differs")

    def parameter_statistics(self) -> ParameterStatistics:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        parameters = dict(self.model.named_parameters())
        trainable = sum(
            parameters[name].numel()
            for name in self._optimizer_parameter_names
            if name in parameters
        )
        per_task = {
            task_id: (
                0
                if task_id == 0
                else len(self.protocol.current_class_indices(task_id))
                * (self.options.ccim_hidden_dim + 1)
            )
            for task_id in range(self.protocol.num_tasks)
        }
        completed = max(self._completed_task_id, 0)
        incremental = sum(
            per_task[task_id]
            for task_id in range(1, min(completed + 1, self.protocol.num_tasks))
        )
        return ParameterStatistics(
            total_parameters=total,
            trainable_parameters=trainable,
            incremental_parameters=incremental,
            per_task_incremental_parameters=per_task,
        )

    def memory_statistics(self) -> MemoryStatistics:
        # The fixed causal dictionary is an architectural resource, separately
        # reported in bytes above; it contains no stored replay examples.
        return MemoryStatistics(replay_memory_samples=0, replay_memory_bytes=0)


register_method("emot_net_ccim_ft", EMOTNetCCIMFTBenchmarkMethod)
