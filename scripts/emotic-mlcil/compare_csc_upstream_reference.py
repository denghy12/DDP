#!/usr/bin/env python3
"""Compare the independent CSC port with the fixed external upstream source.

The script never vendors upstream code. It verifies the immutable external
source tree, imports ``cigcn.py`` as an execution oracle, and extracts the exact
``network_expansion`` function from ``CSC.py`` at runtime for a controlled
expansion audit.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.csc.model import CSCModel


UPSTREAM_COMMIT = "0bab38a00d6e0555f2df855ae2fe8db1fea68b12"
UPSTREAM_ARCHIVE_SHA256 = (
    "588a098a1c7f3d813dee7df777c283fd27768a08a125d4e60b11b2d6ebdb7faa"
)
UPSTREAM_TREE_SHA256 = (
    "87abbdd45cdef71cd271db1a1cbac14c89a2aec5b34fdfeb8ec8ac79802e9087"
)
UPSTREAM_CIGCN_SHA256 = (
    "78ac46707b1c9479f963fd085258f16c51ef5d3c70488b7cc788380ec110996e"
)
UPSTREAM_DRIVER_SHA256 = (
    "a144eb9999bcfb054593ea7fc3c7c68ba66fe910290c4101b9c9ff6f29093a78"
)
TOKEN_DIM = 2048
GRAPH_DIM = 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_upstream(root: Path, archive: Optional[Path]) -> Dict[str, Any]:
    cigcn_path = root / "CSC" / "cigcn.py"
    driver_path = root / "CSC" / "CSC.py"
    if not cigcn_path.is_file() or not driver_path.is_file():
        raise FileNotFoundError(
            "Expected fixed CSC extraction containing CSC/cigcn.py and CSC/CSC.py"
        )
    observed = {
        "tree_sha256": _tree_sha256(root),
        "cigcn_sha256": _sha256(cigcn_path),
        "driver_sha256": _sha256(driver_path),
    }
    expected = {
        "tree_sha256": UPSTREAM_TREE_SHA256,
        "cigcn_sha256": UPSTREAM_CIGCN_SHA256,
        "driver_sha256": UPSTREAM_DRIVER_SHA256,
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(
                f"CSC upstream {key} differs: {observed[key]} != {value}"
            )
    archive_sha = None
    if archive is not None:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archive_sha = _sha256(archive)
        if archive_sha != UPSTREAM_ARCHIVE_SHA256:
            raise ValueError(
                "CSC upstream archive SHA-256 differs: "
                f"{archive_sha} != {UPSTREAM_ARCHIVE_SHA256}"
            )
    return {
        "commit": UPSTREAM_COMMIT,
        "archive_sha256": archive_sha or UPSTREAM_ARCHIVE_SHA256,
        **observed,
        "source_copied_into_repository": False,
        "oracle_mode": "dynamic_import_from_external_fixed_extraction",
    }


def _load_upstream_cigcn(root: Path):
    path = root / "CSC" / "cigcn.py"
    name = "csc_upstream_0bab38a_cigcn"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream CSC module: {path}")
    module = importlib.util.module_from_spec(spec)
    old_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_setting
    return module


def _load_upstream_expansion(root: Path):
    """Compile only the exact upstream method body from the verified file."""

    path = root / "CSC" / "CSC.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CSC_MLCIL":
            method = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef)
                    and child.name == "network_expansion"
                ),
                None,
            )
            break
    if method is None:
        raise ValueError("Fixed upstream CSC.py has no network_expansion method")
    oracle_class = ast.ClassDef(
        name="UpstreamExpansionOracle",
        bases=[],
        keywords=[],
        body=[copy.deepcopy(method)],
        decorator_list=[],
    )
    extracted = ast.Module(body=[oracle_class], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: Dict[str, Any] = {"torch": torch, "nn": nn}
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace["UpstreamExpansionOracle"]


class _DummyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(TOKEN_DIM, 1)


class _IdentitySpatialBackbone(nn.Module):
    """Give the upstream CI_GCN an already-extracted spatial feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.space_to_depth = nn.Identity()
        self.conv1 = nn.Identity()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()
        self.head = _DummyHead()


class _SpatialToTokenEncoder(nn.Module):
    """Expose the identical spatial map as CLS + flattened patch tokens."""

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        patches = feature_map.flatten(2).transpose(1, 2)
        cls = torch.zeros(
            patches.shape[0],
            1,
            patches.shape[2],
            device=patches.device,
            dtype=patches.dtype,
        )
        return torch.cat([cls, patches], dim=1)


def _build_models(upstream_module, classes: int, dtype: torch.dtype):
    upstream = upstream_module.CI_GCN(
        _IdentitySpatialBackbone(), 0, classes
    ).to(dtype=dtype)
    port = CSCModel(
        token_encoder=_SpatialToTokenEncoder(),
        token_dim=TOKEN_DIM,
        graph_dim=GRAPH_DIM,
    ).to(dtype=dtype)
    port.add_task(classes)
    upstream.eval()
    port.eval()
    return upstream, port


def _copy_upstream_to_port(
    upstream: nn.Module,
    port: CSCModel,
    *,
    preserve_port_old_biases: int = 0,
) -> None:
    if (
        port.class_activation is None
        or port.general_relation is None
        or port.specific_relation is None
        or port.graph_classifier is None
        or port.identity_mask is None
    ):
        raise RuntimeError("CSC port has not been expanded")
    old_specific_bias = port.specific_relation.bias[
        :preserve_port_old_biases
    ].detach().clone()
    old_graph_bias = port.graph_classifier.bias[
        :preserve_port_old_biases
    ].detach().clone()
    with torch.no_grad():
        port.class_activation.weight.copy_(upstream.fc.weight[:, :, 0, 0])
        port.feature_projection.weight.copy_(
            upstream.conv_transform.weight[:, :, 0, 0]
        )
        port.feature_projection.bias.copy_(upstream.conv_transform.bias)
        port.general_relation.weight.copy_(upstream.gcn.general_adj[0].weight)
        port.general_weight.weight.copy_(upstream.gcn.general_weight[0].weight)
        port.general_weight.bias.copy_(upstream.gcn.general_weight[0].bias)
        port.global_projection.weight.copy_(upstream.gcn.conv_global.weight)
        port.global_projection.bias.copy_(upstream.gcn.conv_global.bias)
        port.global_norm.weight.copy_(upstream.gcn.bn_global.weight)
        port.global_norm.bias.copy_(upstream.gcn.bn_global.bias)
        port.global_norm.running_mean.copy_(upstream.gcn.bn_global.running_mean)
        port.global_norm.running_var.copy_(upstream.gcn.bn_global.running_var)
        port.global_norm.num_batches_tracked.copy_(
            upstream.gcn.bn_global.num_batches_tracked
        )
        port.specific_relation.weight.copy_(
            upstream.gcn.conv_create_co_mat.weight
        )
        port.specific_relation.bias.copy_(upstream.gcn.conv_create_co_mat.bias)
        port.specific_weight.weight.copy_(upstream.gcn.specific_weight.weight)
        port.specific_weight.bias.copy_(upstream.gcn.specific_weight.bias)
        port.graph_classifier.weight.copy_(upstream.last_linear.weight)
        port.graph_classifier.bias.copy_(upstream.last_linear.bias)
        port.identity_mask.copy_(upstream.mask_mat)
        if preserve_port_old_biases:
            port.specific_relation.bias[:preserve_port_old_biases].copy_(
                old_specific_bias
            )
            port.graph_classifier.bias[:preserve_port_old_biases].copy_(
                old_graph_bias
            )


def _parameter_pairs(
    upstream: nn.Module, port: CSCModel
) -> Iterable[Tuple[str, nn.Parameter, nn.Parameter]]:
    if (
        port.class_activation is None
        or port.general_relation is None
        or port.specific_relation is None
        or port.graph_classifier is None
    ):
        raise RuntimeError("CSC port has not been expanded")
    pairs = [
        ("class_activation.weight", upstream.fc.weight, port.class_activation.weight),
        (
            "feature_projection.weight",
            upstream.conv_transform.weight,
            port.feature_projection.weight,
        ),
        (
            "feature_projection.bias",
            upstream.conv_transform.bias,
            port.feature_projection.bias,
        ),
        (
            "general_relation.weight",
            upstream.gcn.general_adj[0].weight,
            port.general_relation.weight,
        ),
        (
            "general_weight.weight",
            upstream.gcn.general_weight[0].weight,
            port.general_weight.weight,
        ),
        (
            "general_weight.bias",
            upstream.gcn.general_weight[0].bias,
            port.general_weight.bias,
        ),
        (
            "global_projection.weight",
            upstream.gcn.conv_global.weight,
            port.global_projection.weight,
        ),
        (
            "global_projection.bias",
            upstream.gcn.conv_global.bias,
            port.global_projection.bias,
        ),
        ("global_norm.weight", upstream.gcn.bn_global.weight, port.global_norm.weight),
        ("global_norm.bias", upstream.gcn.bn_global.bias, port.global_norm.bias),
        (
            "specific_relation.weight",
            upstream.gcn.conv_create_co_mat.weight,
            port.specific_relation.weight,
        ),
        (
            "specific_relation.bias",
            upstream.gcn.conv_create_co_mat.bias,
            port.specific_relation.bias,
        ),
        (
            "specific_weight.weight",
            upstream.gcn.specific_weight.weight,
            port.specific_weight.weight,
        ),
        (
            "specific_weight.bias",
            upstream.gcn.specific_weight.bias,
            port.specific_weight.bias,
        ),
        (
            "graph_classifier.weight",
            upstream.last_linear.weight,
            port.graph_classifier.weight,
        ),
        (
            "graph_classifier.bias",
            upstream.last_linear.bias,
            port.graph_classifier.bias,
        ),
    ]
    return pairs


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.detach() - second.detach()).abs().max().cpu())


def _operator_equivalence(
    upstream_module,
    *,
    dtype: torch.dtype,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    classes = 3
    upstream, port = _build_models(upstream_module, classes, dtype)
    _copy_upstream_to_port(upstream, port)
    base = torch.randn(2, TOKEN_DIM, 2, 3, dtype=dtype)
    upstream_input = base.clone().requires_grad_(True)
    port_input = base.clone().requires_grad_(True)

    upstream_cls, upstream_graph, upstream_logits = upstream(upstream_input)
    port_output = port(port_input)
    probe = torch.randn_like(upstream_logits)
    (upstream_logits * probe).sum().backward()
    (port_output["logits"] * probe).sum().backward()

    parameter_errors: Dict[str, float] = {}
    for name, upstream_parameter, port_parameter in _parameter_pairs(upstream, port):
        if upstream_parameter.grad is None or port_parameter.grad is None:
            raise RuntimeError(f"Mapped parameter has no gradient: {name}")
        upstream_gradient = upstream_parameter.grad
        while upstream_gradient.ndim > port_parameter.grad.ndim:
            upstream_gradient = upstream_gradient.squeeze(-1)
        if upstream_gradient.shape != port_parameter.grad.shape:
            raise RuntimeError(
                f"Mapped gradient shape differs for {name}: "
                f"{tuple(upstream_gradient.shape)} != "
                f"{tuple(port_parameter.grad.shape)}"
            )
        parameter_errors[name] = _max_abs(
            upstream_gradient, port_parameter.grad
        )

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "batch": 2,
        "spatial_shape": [2, 3],
        "classes": classes,
        "classification_logits_max_abs_error": _max_abs(
            upstream_cls, port_output["classification_logits"]
        ),
        "graph_logits_max_abs_error": _max_abs(
            upstream_graph, port_output["graph_logits"]
        ),
        "combined_logits_max_abs_error": _max_abs(
            upstream_logits, port_output["logits"]
        ),
        "sample_relation_max_abs_error": _max_abs(
            upstream.gcn.specific_adj_show,
            port_output["sample_relation"],
        ),
        "input_gradient_max_abs_error": _max_abs(
            upstream_input.grad, port_input.grad
        ),
        "mapped_parameter_gradient_max_abs_error": max(
            parameter_errors.values()
        ),
        "mapped_parameter_gradient_errors": parameter_errors,
    }


def _preservation_errors(
    upstream: nn.Module,
    port: CSCModel,
    snapshots: Mapping[str, torch.Tensor],
    old_classes: int,
) -> Dict[str, Dict[str, float]]:
    if (
        port.class_activation is None
        or port.general_relation is None
        or port.specific_relation is None
        or port.graph_classifier is None
        or port.identity_mask is None
    ):
        raise RuntimeError("CSC port has not been expanded")
    return {
        "official_release": {
            "old_class_activation_weight_max_change": _max_abs(
                upstream.fc.weight[:old_classes], snapshots["upstream_fc"]
            ),
            "old_general_relation_block_max_change": _max_abs(
                upstream.gcn.general_adj[0].weight[:old_classes, :old_classes],
                snapshots["upstream_general"],
            ),
            "old_specific_relation_weight_max_change": _max_abs(
                upstream.gcn.conv_create_co_mat.weight[:old_classes],
                snapshots["upstream_specific_weight"],
            ),
            "old_specific_relation_bias_max_change": _max_abs(
                upstream.gcn.conv_create_co_mat.bias[:old_classes],
                snapshots["upstream_specific_bias"],
            ),
            "old_graph_classifier_weight_max_change": _max_abs(
                upstream.last_linear.weight[:old_classes],
                snapshots["upstream_graph_weight"],
            ),
            "old_graph_classifier_bias_max_change": _max_abs(
                upstream.last_linear.bias[:old_classes],
                snapshots["upstream_graph_bias"],
            ),
            "old_identity_block_max_change": _max_abs(
                upstream.mask_mat[:old_classes, :old_classes],
                snapshots["upstream_identity"],
            ),
        },
        "independent_port": {
            "old_class_activation_weight_max_change": _max_abs(
                port.class_activation.weight[:old_classes], snapshots["port_fc"]
            ),
            "old_general_relation_block_max_change": _max_abs(
                port.general_relation.weight[:old_classes, :old_classes],
                snapshots["port_general"],
            ),
            "old_specific_relation_weight_max_change": _max_abs(
                port.specific_relation.weight[:old_classes],
                snapshots["port_specific_weight"],
            ),
            "old_specific_relation_bias_max_change": _max_abs(
                port.specific_relation.bias[:old_classes],
                snapshots["port_specific_bias"],
            ),
            "old_graph_classifier_weight_max_change": _max_abs(
                port.graph_classifier.weight[:old_classes],
                snapshots["port_graph_weight"],
            ),
            "old_graph_classifier_bias_max_change": _max_abs(
                port.graph_classifier.bias[:old_classes],
                snapshots["port_graph_bias"],
            ),
            "old_identity_block_max_change": _max_abs(
                port.identity_mask[:old_classes, :old_classes],
                snapshots["port_identity"],
            ),
        },
    }


def _optimizer_coverage(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, int]:
    live_parameters = list(model.parameters())
    live_ids = {id(parameter) for parameter in live_parameters}
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    missing = [
        parameter
        for parameter in live_parameters
        if parameter.requires_grad and id(parameter) not in optimizer_ids
    ]
    stale = [
        parameter
        for parameter in optimizer_parameters
        if id(parameter) not in live_ids
    ]
    return {
        "live_trainable_parameter_tensors": sum(
            int(parameter.requires_grad) for parameter in live_parameters
        ),
        "missing_live_parameter_tensors": len(missing),
        "missing_live_parameters": sum(parameter.numel() for parameter in missing),
        "stale_parameter_tensors": len(stale),
        "stale_parameters": sum(parameter.numel() for parameter in stale),
    }


def _expansion_audit(upstream_module, expansion_class, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    old_classes = 3
    added_classes = 2
    upstream, port = _build_models(upstream_module, old_classes, torch.float32)
    _copy_upstream_to_port(upstream, port)
    snapshots = {
        "upstream_fc": upstream.fc.weight.detach().clone(),
        "upstream_general": upstream.gcn.general_adj[0].weight.detach().clone(),
        "upstream_specific_weight": (
            upstream.gcn.conv_create_co_mat.weight.detach().clone()
        ),
        "upstream_specific_bias": (
            upstream.gcn.conv_create_co_mat.bias.detach().clone()
        ),
        "upstream_graph_weight": upstream.last_linear.weight.detach().clone(),
        "upstream_graph_bias": upstream.last_linear.bias.detach().clone(),
        "upstream_identity": upstream.mask_mat.detach().clone(),
        "port_fc": port.class_activation.weight.detach().clone(),
        "port_general": port.general_relation.weight.detach().clone(),
        "port_specific_weight": port.specific_relation.weight.detach().clone(),
        "port_specific_bias": port.specific_relation.bias.detach().clone(),
        "port_graph_weight": port.graph_classifier.weight.detach().clone(),
        "port_graph_bias": port.graph_classifier.bias.detach().clone(),
        "port_identity": port.identity_mask.detach().clone(),
    }
    old_upstream_optimizer = torch.optim.Adam(upstream.parameters(), lr=4.0e-5)
    oracle = expansion_class()
    oracle.model = upstream
    oracle.task_size = added_classes
    oracle.device = torch.device("cpu")
    oracle.network_expansion(old_classes, old_classes + added_classes)
    port.add_task(added_classes)
    preservation = _preservation_errors(
        upstream, port, snapshots, old_classes
    )

    # Synchronize every expanded value except the two old bias blocks that the
    # release reinitializes but the port intentionally preserves.
    _copy_upstream_to_port(
        upstream,
        port,
        preserve_port_old_biases=old_classes,
    )
    feature_map = torch.randn(2, TOKEN_DIM, 2, 3)
    with torch.no_grad():
        controlled_upstream = upstream(feature_map)[2]
        controlled_port = port(feature_map)["logits"]
    controlled_delta = _max_abs(controlled_upstream, controlled_port)

    # A full parameter remap must still collapse the operator error after the
    # class axis grows; this separates operator fidelity from lifecycle policy.
    _copy_upstream_to_port(upstream, port)
    with torch.no_grad():
        full_remap_delta = _max_abs(
            upstream(feature_map)[2], port(feature_map)["logits"]
        )
    rebuilt_port_optimizer = torch.optim.Adam(
        (parameter for parameter in port.parameters() if parameter.requires_grad),
        lr=4.0e-5,
    )
    return {
        "seed": seed,
        "old_classes": old_classes,
        "added_classes": added_classes,
        "parameter_preservation": preservation,
        "controlled_logit_delta_from_reinitialized_old_biases": controlled_delta,
        "full_expanded_parameter_remap_max_abs_error": full_remap_delta,
        "optimizer_coverage_after_expansion": {
            "official_optimizer_created_before_expansion": _optimizer_coverage(
                upstream, old_upstream_optimizer
            ),
            "port_optimizer_rebuilt_after_expansion": _optimizer_coverage(
                port, rebuilt_port_optimizer
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-archive", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.upstream_root.resolve()
    provenance = _verify_upstream(root, args.upstream_archive)
    upstream_module = _load_upstream_cigcn(root)
    expansion_class = _load_upstream_expansion(root)
    result = {
        "schema_version": 1,
        "comparison": "independent_csc_track_a_vs_fixed_upstream_execution_oracle",
        "provenance": provenance,
        "base_operator_equivalence": [
            _operator_equivalence(
                upstream_module,
                dtype=torch.float64,
                seed=2026080201,
            ),
            _operator_equivalence(
                upstream_module,
                dtype=torch.float32,
                seed=2026080202,
            ),
        ],
        "incremental_expansion_audit": _expansion_audit(
            upstream_module,
            expansion_class,
            seed=2026080203,
        ),
        "interpretation": {
            "operator_equivalence": (
                "Same input and mapped weights; measures implementation error."
            ),
            "controlled_expansion_delta": (
                "All expanded values synchronized except old biases reinitialized "
                "by the release and preserved by the port."
            ),
            "full_remap": (
                "All expanded values synchronized; must recover numerical equivalence."
            ),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
