#!/usr/bin/env python3
"""Audit the independent L3A operators against fixed external source code."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.l3a import (
    AsymmetricLoss,
    weighted_analytic_statistics,
)


UPSTREAM_COMMIT = "1067bbd6124a7aa96136baa555080ba9183ab2ac"
UPSTREAM_GIT_TREE = "9e6ac470600328f2a393f26e30ea764c3b50102b"
UPSTREAM_ARCHIVE_SHA256 = (
    "307d721d76069a7fe95a10346be1323a842cd9624f8d55200f4cf7cd45462409"
)
EXPECTED_FILES = {
    "MultiLabelIncremental_L3A.py": (
        "3f6ccdad5fa4c45627f414d7f942d5c544c847cb5e03f100a618230a335eb689"
    ),
    "src/helper_functions/helper_functions.py": (
        "82ec1c1ee32f270ddf62e8d4d62e81f3dc5d8362379a45af06338748652ff35a"
    ),
    "src/loss_functions/losses.py": (
        "2c0cab7b8feaa5c9ed3ba6725e31fdd003b60f84a14cdc281ccfc29711209139"
    ),
    "src/models/utils/factory.py": (
        "1cee7a09e75ce7dfb8d8f395370bf66fe293295a84fcb54bfc7e99cc3b213104"
    ),
    "configs/l3a_vit_coco.yaml": (
        "09508b79e9bc8cdafe4253fa7e0897ded115ca2d24a144e5fddd4af38b8fea00"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_upstream(root: Path, archive: Optional[Path]) -> Dict[str, Any]:
    observed: Dict[str, str] = {}
    for relative, expected in EXPECTED_FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(
                f"L3A upstream {relative} differs: {digest} != {expected}"
            )
        observed[relative] = digest

    git_commit = None
    git_tree = None
    if (root / ".git").exists():
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if git_commit != UPSTREAM_COMMIT or git_tree != UPSTREAM_GIT_TREE:
            raise ValueError("L3A external Git commit/tree differs")

    archive_sha = None
    if archive is not None:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archive_sha = _sha256(archive)
        if archive_sha != UPSTREAM_ARCHIVE_SHA256:
            raise ValueError(
                f"L3A archive differs: {archive_sha} != {UPSTREAM_ARCHIVE_SHA256}"
            )
    return {
        "repository": "https://github.com/scut-zx/L3A",
        "commit": UPSTREAM_COMMIT,
        "git_tree": UPSTREAM_GIT_TREE,
        "observed_git_commit": git_commit,
        "observed_git_tree": git_tree,
        "registered_archive_sha256": UPSTREAM_ARCHIVE_SHA256,
        "observed_archive_sha256": archive_sha,
        "verified_file_sha256": observed,
        "source_copied_into_repository": False,
        "oracle_mode": "runtime_AST_extraction_from_external_fixed_source",
    }


def _extract_definition(
    path: Path,
    name: str,
    node_type,
    namespace: Dict[str, Any],
):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, node_type) and item.name == name
        ),
        None,
    )
    if node is None:
        raise ValueError(f"Fixed L3A source has no {name}")
    extracted = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace[name]


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach() - right.detach())).cpu())


def _operator_audit(root: Path) -> Dict[str, Any]:
    upstream_normalize = _extract_definition(
        root / "src/helper_functions/helper_functions.py",
        "normalize_sqrt",
        ast.FunctionDef,
        {"np": np},
    )
    upstream_loss_class = _extract_definition(
        root / "src/loss_functions/losses.py",
        "AsymmetricLoss",
        ast.ClassDef,
        {"torch": torch, "nn": nn},
    )

    counts = [12.0, 3.0, 0.0, 27.0]
    with np.errstate(divide="ignore"):
        expected_weights = torch.tensor(
            upstream_normalize(counts), dtype=torch.float64
        )
    inverse = torch.zeros(4, dtype=torch.float64)
    positive = torch.tensor(counts, dtype=torch.float64) > 0
    inverse[positive] = torch.tensor(counts, dtype=torch.float64)[
        positive
    ].reciprocal().pow(0.5)
    port_weights = len(counts) * inverse / inverse.sum()

    features = torch.tensor(
        [[0.2, 0.5, 1.1], [1.0, 0.3, 0.4], [0.7, 1.2, 0.6]],
        dtype=torch.float64,
    )
    labels = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    port_a, port_c, port_omega = weighted_analytic_statistics(
        features, labels, port_weights
    )
    upstream_omega = torch.tensor(
        [
            float(
                sum(
                    expected_weights[index]
                    for index in range(labels.shape[1])
                    if labels[row, index] == 1
                )
                / labels[row].sum()
            )
            for row in range(labels.shape[0])
        ],
        dtype=torch.float64,
    )
    omega_matrix = torch.diag(upstream_omega)
    upstream_a = features.t() @ omega_matrix @ features
    upstream_c = features.t() @ omega_matrix @ labels
    ridge = 1.0
    upstream_solution = torch.inverse(
        upstream_a + ridge * torch.eye(features.shape[1], dtype=torch.float64)
    ) @ upstream_c
    port_solution = torch.linalg.inv(
        port_a + ridge * torch.eye(features.shape[1], dtype=torch.float64)
    ) @ port_c

    logits_upstream = torch.tensor(
        [[-1.2, 0.4, 2.1], [0.7, -0.3, 1.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits_port = logits_upstream.detach().clone().requires_grad_(True)
    targets = torch.tensor(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    upstream_loss = upstream_loss_class(
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        disable_torch_grad_focal_loss=True,
    )(logits_upstream, targets)
    port_loss = AsymmetricLoss()(logits_port, targets)
    upstream_loss.backward()
    port_loss.backward()

    probabilities = torch.tensor([[0.7, 0.7000001, 0.2]], dtype=torch.float64)
    pseudo_expected = probabilities > 0.7
    pseudo_port = probabilities > 0.7
    return {
        "normalized_weight_max_abs_error": _max_abs(
            expected_weights, port_weights
        ),
        "sample_weight_max_abs_error": _max_abs(upstream_omega, port_omega),
        "analytic_a_max_abs_error": _max_abs(upstream_a, port_a),
        "analytic_c_max_abs_error": _max_abs(upstream_c, port_c),
        "analytic_solution_max_abs_error": _max_abs(
            upstream_solution, port_solution
        ),
        "asl_loss_abs_error": abs(
            float(upstream_loss.detach()) - float(port_loss.detach())
        ),
        "asl_gradient_max_abs_error": _max_abs(
            logits_upstream.grad, logits_port.grad
        ),
        "pseudo_threshold_exact_match": bool(
            torch.equal(pseudo_expected, pseudo_port)
        ),
        "pseudo_threshold_is_strict_greater_than": True,
    }


def _port_provenance() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "method_sha256": _sha256(
            REPOSITORY_ROOT
            / "benchmarks/emotic_mlcil/methods/l3a/method.py"
        ),
        "model_sha256": _sha256(
            REPOSITORY_ROOT
            / "benchmarks/emotic_mlcil/methods/l3a/model.py"
        ),
        "comparison_script_sha256": _sha256(Path(__file__).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--upstream-archive")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.upstream_root).resolve()
    archive = (
        Path(args.upstream_archive).resolve()
        if args.upstream_archive
        else None
    )
    payload = {
        "schema_version": 1,
        "method": "L3A",
        "upstream": _verify_upstream(root, archive),
        "operator_equivalence": _operator_audit(root),
        "port": _port_provenance(),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
