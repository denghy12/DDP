#!/usr/bin/env python3
"""Verify the independent AGCN operators against immutable upstream source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

if not hasattr(np, "int"):
    np.int = int

from benchmarks.emotic_mlcil.methods.agcn import (
    AGCNGraphNetwork,
    graph_normalize,
    threshold_scale,
)

COMMIT = "3afe2ecbbef0051c6e841a97c369885011a683f0"
TREE = "6c25689b81d0807523108a782ee59630d25a9b40"
ARCHIVE_SHA = "b5843d3ee0964767b48c49ba4b71fdf93c4bf954460669ed2b1cd05f9504f301"
FILE_SHA = {
    "AGCN-LML/GCN.py": "ad8732008613a80664dfb0767ec3fa3b05f7911a3808dad7324d84bd153f584a",
    "AGCN-LML/GCNRSN.py": "e8badf2948e426b5a02339e1e8519877d8c60df68770aac3df10dca2a4274b18",
    "AGCN-LML/make_cmatrix_online.py": "7256a649da1699e6c4f41c1cc8aec827c37f61b9e73408f3d4fc4ee75898459f",
    "AGCN-LML/init_cmatrix.py": "fc59fa8f0cd077668ca11e06aff88d90cece19a8a64953d1bc28b6942dce0cc9",
    "AGCN-LML/myresnet_fc.py": "365d4ded3794c67b1d8fdb2a40354ba9aa2023224c1b669366293b2325825c3e",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extracted_namespace(path: Path, names, prefix: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "torch": torch, "nn": torch.nn, "Parameter": torch.nn.Parameter,
        "math": __import__("math"), "np": np, "device": torch.device("cpu"),
    }
    exec(compile(module, f"{prefix}:{path}", "exec"), namespace)
    return namespace


def compare(source_root: Path, source_archive: Optional[Path] = None):
    observed = {}
    for relative, expected in FILE_SHA.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha(path)
        if observed[relative] != expected:
            raise ValueError(f"Fixed AGCN source differs: {relative}")
    archive_observed = None
    if source_archive is not None:
        archive_observed = sha(source_archive)
        if archive_observed != ARCHIVE_SHA:
            raise ValueError("Fixed AGCN archive SHA-256 differs")

    gcnrsn = extracted_namespace(
        source_root / "AGCN-LML/GCNRSN.py", {"GraphConvolution", "gnn"}, "agcn-upstream"
    )
    graph_source = gcnrsn["gnn"](in_channel=3).cpu()
    graph_port = AGCNGraphNetwork(3, 1024, 4).cpu()
    with torch.no_grad():
        graph_port.gc1.weight.copy_(graph_source.gc1.weight)
        graph_port.gc2.weight.copy_(graph_source.gc2.weight[:, :4])
    nodes = torch.tensor([[0.2, 0.4, 0.6], [0.1, 0.7, 0.3]], dtype=torch.float32)
    adjacency = torch.tensor([[1.0, 0.2], [0.3, 1.0]], dtype=torch.float32)
    upstream_nodes = graph_source(adjacency, nodes)[:4].t()
    port_nodes = graph_port(adjacency, nodes)

    base = extracted_namespace(
        source_root / "AGCN-LML/GCN.py", {"gen_A2", "gen_P"}, "agcn-base"
    )
    raw = np.array([[0.0, 0.8], [0.4, 0.0]], dtype=np.float32)
    upstream_base = base["gen_A2"](2, 0.0, raw.copy())
    port_base = threshold_scale(
        torch.tensor(raw), threshold=0.0, scale=0.28, add_identity=True
    )
    upstream_norm = base["gen_P"](torch.tensor(upstream_base).float())
    port_norm = graph_normalize(port_base, -0.8)

    online = extracted_namespace(
        source_root / "AGCN-LML/make_cmatrix_online.py",
        {"gen_A2", "gen_A3", "gen_P"},
        "agcn-online",
    )
    cross = np.array([[0.1, 0.8], [0.5, 0.2]], dtype=np.float32)
    upstream_cross = online["gen_A3"](0.3, cross.copy())
    port_cross = threshold_scale(
        torch.tensor(cross), threshold=0.3, scale=0.25, add_identity=False
    )
    current = np.array([[0.0, 0.9], [0.7, 0.0]], dtype=np.float32)
    upstream_current = online["gen_P"](
        torch.tensor(online["gen_A2"](2, 0.4, current.copy())).float()
    )
    port_current = graph_normalize(
        threshold_scale(torch.tensor(current), threshold=0.4, scale=0.25, add_identity=True), -0.5
    )
    errors = {
        "two_layer_gcn_max_abs_error": float((upstream_nodes - port_nodes).abs().max()),
        "task0_threshold_scale_max_abs_error": float(np.max(np.abs(upstream_base - port_base.numpy()))),
        "task0_normalization_max_abs_error": float((upstream_norm - port_norm).abs().max()),
        "cross_block_max_abs_error": float(np.max(np.abs(upstream_cross - port_cross.numpy()))),
        "incremental_current_normalization_max_abs_error": float((upstream_current - port_current).abs().max()),
    }
    if max(errors.values()) > 1.0e-6:
        raise ValueError(f"AGCN operator equivalence failed: {errors}")
    return {
        "schema_version": 1,
        "method": "AGCN",
        "upstream": {
            "repository": "https://github.com/Kaile-Du/AGCN",
            "commit": COMMIT,
            "tree": TREE,
            "archive_sha256": ARCHIVE_SHA,
            "observed_archive_sha256": archive_observed,
            "license": "Apache-2.0",
            "verified_file_sha256": observed,
            "source_copied_into_repository": False,
        },
        "operator_equivalence": errors,
        "documented_source_gaps": {
            "undefined_loss_variables": ["a", "b", "c"],
            "paper_resolved_loss_weights": [0.07, 0.93, 100000.0],
            "acm_old_activation": "softmax_in_release",
            "distillation_old_activation": "sigmoid_in_release",
            "released_adam_epsilon": 1e-8,
            "paper_adam_epsilon": 1e-4,
            "license_note": "root Apache-2.0 license; GCN.py has a bare BSD header comment",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--upstream-archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = compare(args.upstream_root, args.upstream_archive)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
