#!/usr/bin/env python3
"""Verify fixed BENet source identity and source-owned loss operators."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

from benchmarks.emotic_mlcil.methods.benet_ft import focal_tag_loss, verify_benet_source
from benchmarks.emotic_mlcil.methods.benet_ft.model import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY


REGISTERED_SOURCE_COMMIT = "b86747e0e259b1ec70fc84ca76efd7ea3bb3728e"
if UPSTREAM_COMMIT != REGISTERED_SOURCE_COMMIT:
    raise RuntimeError("BENet adapter and oracle source commits differ")


def _git_value(root: Path, *arguments: str):
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def compare(source_root: Path):
    source_root = source_root.expanduser().resolve()
    hashes = verify_benet_source(source_root)
    loss_file = source_root / "lib/core/loss_mt.py"
    spec = importlib.util.spec_from_file_location("_benet_fixed_loss", loss_file)
    if spec is None or spec.loader is None:
        raise ImportError(loss_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logits = torch.tensor([[0.2, -0.8, 1.1], [-0.4, 0.6, 0.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    source_logits = logits.detach().clone().requires_grad_(True)
    port_loss = focal_tag_loss(logits, targets)
    source_loss = module.FocalTagLoss()(torch.sigmoid(source_logits), targets)
    port_grad = torch.autograd.grad(port_loss, logits)[0]
    source_grad = torch.autograd.grad(source_loss, source_logits)[0]
    loss_error = abs(float(port_loss.detach()) - float(source_loss.detach()))
    gradient_error = float((port_grad - source_grad).abs().max())
    if loss_error > 1.0e-12 or gradient_error > 1.0e-12:
        raise RuntimeError("BENet FocalTagLoss differs from fixed source")
    observed_commit = _git_value(source_root, "rev-parse", "HEAD")
    if observed_commit is not None and observed_commit != UPSTREAM_COMMIT:
        raise ValueError("BENet checkout commit differs from registered source")
    return {
        "schema_version": 1,
        "method": "BENet-FT",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "observed_git_commit": observed_commit,
            "license": "NOASSERTION",
            "source_copied_into_repository": False,
            "verified_file_sha256": hashes,
        },
        "operator_equivalence": {
            "focal_tag_loss_abs_error": loss_error,
            "focal_tag_gradient_max_abs_error": gradient_error,
        },
        "protocol_mapping": {
            "strategy": "current-label-only sequential fine-tuning",
            "extra_heco_data_used": False,
            "test_selection_used": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = compare(Path(args.source_root))
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
