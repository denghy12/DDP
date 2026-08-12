#!/usr/bin/env python3
"""Compare the immutable official CCIM operators with the benchmark port."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft.model import (
    CCIMBackdoorIntervention,
    CCIM_SOURCE_SHA256,
    CCIM_UPSTREAM_COMMIT,
    CCIM_UPSTREAM_REPOSITORY,
)


EXPECTED_SHA256 = {
    "CCIM.py": CCIM_SOURCE_SHA256,
    "README.md": "274b3d9af132ce302aff4426a598f3258b3389f6c3929df85720eaa497d6d65f",
    "license": "d1ad4ba3c7b21062cc77013d84348e9ea8eda2b3770eff36d8f93ce290711a17",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Unable to inspect CCIM Git source")
    return result.stdout.strip()


def _load_source_namespace(path: Path) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected_names = {
        "gelu",
        "classifier",
        "dot_product_intervention",
        "additive_intervention",
        "CCIM",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in selected_names
    ]
    if {node.name for node in selected} != selected_names:
        raise ValueError("Official CCIM source operators are incomplete")
    module = ast.Module(body=selected, type_ignores=[])
    namespace: Dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Parameter": Parameter,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _copy_shared_state(source: nn.Module, port: CCIMBackdoorIntervention) -> None:
    port.w_h.data.copy_(source.w_h.data)
    port.w_g.data.copy_(source.w_g.data)
    port.query.load_state_dict(source.causal_intervention.query.state_dict())
    port.key.load_state_dict(source.causal_intervention.key.state_dict())
    if port.w_t is not None:
        port.w_t.load_state_dict(source.causal_intervention.w_t.state_dict())
    port.classifier.norm.load_state_dict(source.classifier.norm.state_dict())
    port.classifier.fc1.load_state_dict(source.classifier.fc1.state_dict())
    port.classifier.fc2.load_state_dict(source.classifier.fc2.state_dict())


def compare(
    upstream_root: Path,
    port_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = upstream_root.expanduser().resolve()
    observed = {name: _sha256(root / name) for name in EXPECTED_SHA256}
    if observed != EXPECTED_SHA256:
        raise ValueError("Fixed CCIM source hash mismatch")
    if _git(root, "rev-parse", "HEAD") != CCIM_UPSTREAM_COMMIT:
        raise ValueError("Fixed CCIM source commit mismatch")
    if _git(root, "status", "--porcelain"):
        raise ValueError("Fixed CCIM source checkout must be clean")
    namespace = _load_source_namespace(root / "CCIM.py")

    errors: Dict[str, float] = {}
    for strategy in ("dp_cause", "ad_cause"):
        torch.manual_seed(371)
        source = namespace["CCIM"](7, 11, strategy).double().eval()
        port = CCIMBackdoorIntervention(
            joint_dim=7,
            confounder_dim=11,
            hidden_dim=128,
            attention_dim=256,
            strategy=strategy,
            dropout=0.5,
        ).double().eval()
        _copy_shared_state(source, port)
        joint = torch.randn(4, 7, dtype=torch.double)
        dictionary = torch.randn(6, 11, dtype=torch.double)
        prior = torch.rand(6, 1, dtype=torch.double)
        prior = prior / prior.sum()
        with torch.no_grad():
            reference_intervention = source.causal_intervention(
                dictionary, joint, prior
            )
            port_intervention, attention = port.intervene(
                joint, dictionary, prior
            )
            reference_logits = source(joint, dictionary, prior, "EMOTIC")
            port_hidden, forward_attention = port(joint, dictionary, prior)
            port_logits = source.emotic_fc(port_hidden)
        errors[f"{strategy}_intervention_max_abs_error"] = float(
            (reference_intervention - port_intervention).abs().max()
        )
        errors[f"{strategy}_logits_max_abs_error"] = float(
            (reference_logits - port_logits).abs().max()
        )
        errors[f"{strategy}_attention_reuse_max_abs_error"] = float(
            (attention - forward_attention).abs().max()
        )

    repository = port_root or REPOSITORY_ROOT
    port_files = (
        repository / "benchmarks/emotic_mlcil/methods/emot_net_ccim_ft/model.py",
        repository / "benchmarks/emotic_mlcil/methods/emot_net_ccim_ft/method.py",
        repository / "scripts/emotic-mlcil/prepare_ccim_task0_dictionary.py",
    )
    return {
        "schema_version": 1,
        "method": "EMOT-Net+CCIM-FT",
        "upstream": {
            "repository": CCIM_UPSTREAM_REPOSITORY,
            "commit": CCIM_UPSTREAM_COMMIT,
            "license": "MIT",
            "source_copied_into_repository": False,
            "verified_file_sha256": observed,
        },
        "operator_equivalence": errors,
        "incremental_mapping": {
            "track": "B",
            "strategy": "sequential_finetuning",
            "host": "official EMOT-Net native backbone",
            "dictionary_size": 256,
            "dictionary_scope": "Task-0-accessible train images only, frozen",
            "future_task_images_used": False,
            "current_label_only": True,
            "distillation": False,
            "replay": False,
            "adapter": False,
        },
        "port_file_sha256": {
            str(path.relative_to(repository)): _sha256(path) for path in port_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = compare(args.upstream_root)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
