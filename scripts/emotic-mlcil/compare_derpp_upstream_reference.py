#!/usr/bin/env python3
"""Verify DER++ objective structure against the immutable NeurIPS source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.replay import (
    derpp_weighted_loss,
    masked_logit_mse,
)


TAG_OBJECT = "1f3978bd58a7f873675f5245d3e6a1ad46bee28a"
COMMIT = "cb9a36d788d6ad051c9eee0da358b25421d909f5"
ARCHIVE_SHA256 = "d7cdffefdb7d77939a1055984cb586ad83af0220439cd76e4a106ea201c1695b"
FILE_SHA256 = {
    "LICENSE": "309ca56cfbbe29aa036d1c53f8f05d5ee0a1dd78dcc37f53f94e840b22b60275",
    "models/derpp.py": "d38736e8d8c888300a8e5ddcac7cab1ac12e2eb1a0aa4c28f2387bdd79fc973a",
    "utils/buffer.py": "3f4c9b416e22241bd6417d1cd2d6ec0625d7309c1d9e308757635debe545d5d3",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source_structure(source: str):
    tree = ast.parse(source)
    observe = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "observe"
    )
    calls = [node for node in ast.walk(observe) if isinstance(node, ast.Call)]
    get_data_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get_data"
    ]
    mse_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "mse_loss"
    ]
    add_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_data"
    ]
    if len(get_data_calls) != 2 or len(mse_calls) != 1 or len(add_calls) != 1:
        raise ValueError("Fixed DER++ source objective structure changed")
    stored_fields = {keyword.arg for keyword in add_calls[0].keywords}
    if stored_fields != {"examples", "labels", "logits"}:
        raise ValueError("Fixed DER++ source buffer payload changed")
    attributes = {
        node.attr for node in ast.walk(observe) if isinstance(node, ast.Attribute)
    }
    if not {"alpha", "beta", "mse_loss", "add_data"}.issubset(attributes):
        raise ValueError("Fixed DER++ source coefficients/operators changed")
    return {
        "independent_get_data_calls": len(get_data_calls),
        "mse_loss_calls": len(mse_calls),
        "stored_payload_fields": sorted(stored_fields),
    }


def _numeric_equivalence():
    current_logits = torch.tensor([[0.2, -0.4], [0.7, 0.1]])
    current_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dark_outputs = torch.tensor([[0.5, -0.3], [0.2, 0.8]])
    stored_logits = torch.tensor([[0.1, -0.9], [-0.2, 0.4]])
    replay_outputs = torch.tensor([[-0.1, 0.3], [0.9, -0.2]])
    replay_targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask = torch.ones_like(dark_outputs, dtype=torch.bool)
    alpha = 0.5
    beta = 0.5

    current_loss = F.binary_cross_entropy_with_logits(
        current_logits,
        current_targets,
    )
    upstream_dark = F.mse_loss(dark_outputs, stored_logits)
    upstream_labels = F.binary_cross_entropy_with_logits(
        replay_outputs,
        replay_targets,
    )
    upstream_total = current_loss + alpha * upstream_dark + beta * upstream_labels
    port_dark = masked_logit_mse(dark_outputs, stored_logits, mask)
    port_total = derpp_weighted_loss(
        current_loss,
        port_dark,
        upstream_labels,
        alpha,
        beta,
    )
    return {
        "dense_logit_mse_abs_error": abs(
            float(upstream_dark) - float(port_dark)
        ),
        "weighted_objective_abs_error": abs(
            float(upstream_total) - float(port_total)
        ),
    }


def compare(source_root: Path):
    observed = {}
    for relative, expected in FILE_SHA256.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = _sha(path)
        if observed[relative] != expected:
            raise ValueError(f"Fixed DER++ source differs: {relative}")

    structure = _verify_source_structure(
        (source_root / "models/derpp.py").read_text(encoding="utf-8")
    )
    errors = _numeric_equivalence()
    if max(errors.values()) > 1.0e-7:
        raise ValueError(f"DER++ objective equivalence failed: {errors}")
    return {
        "schema_version": 1,
        "method": "DER++",
        "upstream": {
            "repository": "https://github.com/aimagelab/mammoth",
            "tag": "neurips2020",
            "tag_object": TAG_OBJECT,
            "commit": COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "license": "MIT",
            "verified_file_sha256": observed,
            "source_copied_into_repository": False,
        },
        "source_structure": structure,
        "operator_equivalence": errors,
        "track_a_mapping": {
            "source_loss": "CE + alpha*MSE(logits) + beta*CE(replay)",
            "registered_loss": (
                "current sigmoid BCE + alpha*masked logit MSE + "
                "beta*visible-label sigmoid BCE"
            ),
            "alpha": 0.5,
            "beta": 0.5,
            "registered_capacity": "20 * seen classes",
            "registered_update_timing": "online after optimizer attempt",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.upstream_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
