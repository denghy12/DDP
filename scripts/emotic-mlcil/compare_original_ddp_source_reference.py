#!/usr/bin/env python3
"""Audit Original-DDP-Tau2 against the fixed external DDP snapshot.

The external source is never copied into this repository.  The script verifies
the complete snapshot, extracts the released BCE and DDP forward operator with
AST, and compares them with the adapter-free benchmark execution path.  The
registered PCD schedule is intentionally *not* claimed equivalent: its exact
source and benchmark values are both emitted as an explicit deviation record.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.original_ddp import OriginalDDPLoss
from benchmarks.emotic_mlcil.methods.original_ddp.method import (
    SOURCE_DRIVER_SHA256,
    SOURCE_LOSS_SHA256,
    SOURCE_MODEL_SHA256,
    SOURCE_SNAPSHOT_TREE_SHA256,
    original_ddp_temperature,
)
from models.ddp import DDP as PortDDP
from models.ddp import TextEncoder as PortTextEncoder


EXPECTED_FILE_COUNT = 30
EXPECTED_TOTAL_BYTES = 1_470_709
SOURCE_ARCHIVE_SHA256 = (
    "b99892e1c12c942b4a8506b89049f8f35933001e0b8018e928883ee60db847a5"
)
EXPECTED_FILES = {
    "DDP.py": SOURCE_DRIVER_SHA256,
    "README.md": "43fbf65296ec5bde43ff2dbb298092f61373a9440e5bbea6ee90216e9ffd9406",
    "bce_loss.py": SOURCE_LOSS_SHA256,
    "build_cfg.py": "32fda8566779f97492c5ce21fe437d27636851386653b5d8f4c0374bad40d210",
    "clip/model.py": "b04d0d4f8e7d0dcab3dc9493a97e9eee7c5e0c58f85379292ec3ebf5a398c952",
    "configs/models/vitb16_ep50.yaml": "93034d5a382da508624d689cbc95c8f6ef544d5d8d9290f8e6bfe5f1b3158b8a",
    "models/ddp.py": SOURCE_MODEL_SHA256,
    "opts.py": "0fd3b993a832ca7196da11dfa689865e83f7bf903267afe4525153d1d2caa968",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_records(root: Path):
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.name == ".DS_Store" or path.suffix == ".pyc":
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return records


def _snapshot_tree_sha256(records) -> str:
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_value(root: Path, *arguments: str) -> Optional[str]:
    if not (root / ".git").exists():
        return None
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_source(root: Path, archive: Optional[Path] = None) -> Dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = _snapshot_records(root)
    tree_sha = _snapshot_tree_sha256(records)
    file_count = len(records)
    total_bytes = sum(int(record["bytes"]) for record in records)
    if tree_sha != SOURCE_SNAPSHOT_TREE_SHA256:
        raise ValueError(
            "Original DDP source snapshot differs: "
            f"{tree_sha} != {SOURCE_SNAPSHOT_TREE_SHA256}"
        )
    if file_count != EXPECTED_FILE_COUNT or total_bytes != EXPECTED_TOTAL_BYTES:
        raise ValueError("Original DDP snapshot file count/byte count differs")
    observed = {}
    for relative, expected in EXPECTED_FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(
                f"Original DDP source {relative} differs: {digest} != {expected}"
            )
        observed[relative] = digest
    license_files = sorted(
        record["path"]
        for record in records
        if Path(record["path"]).name.lower().startswith(
            ("license", "licence", "copying", "notice")
        )
    )
    if license_files:
        raise ValueError("Unexpected license declaration in fixed snapshot")
    observed_archive_sha256 = None
    if archive is not None:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        observed_archive_sha256 = _sha256(archive)
        if observed_archive_sha256 != SOURCE_ARCHIVE_SHA256:
            raise ValueError(
                "Original DDP portable archive differs: "
                f"{observed_archive_sha256} != {SOURCE_ARCHIVE_SHA256}"
            )
    return {
        "source_label": "user-collected_original_DDP_snapshot",
        "repository": None,
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_tree": _git_value(root, "rev-parse", "HEAD^{tree}"),
        "snapshot_tree_sha256": tree_sha,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "registered_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "observed_archive_sha256": observed_archive_sha256,
        "verified_file_sha256": observed,
        "license_files": license_files,
        "license_status": "no_license_file_observed_reference_only",
        "source_copied_into_repository": False,
        "oracle_mode": "runtime_AST_extraction_from_external_fixed_snapshot",
    }


def _extract_definition(path: Path, name: str, node_type, namespace: Dict[str, Any]):
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
        raise ValueError(f"Fixed Original DDP source has no {name}")
    extracted = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace[name]


def _definition_ast_sha256(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef))
            and item.name == name
        ),
        None,
    )
    if node is None:
        raise ValueError(f"Missing prompt-aware CLIP definition: {name}")
    payload = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt_aware_clip_audit(root: Path) -> Dict[str, Any]:
    source_path = root / "clip" / "model.py"
    port_path = REPOSITORY_ROOT / "clip" / "model.py"
    definitions = (
        "LayerNorm",
        "QuickGELU",
        "ResidualAttentionBlock_visual",
        "ResidualAttentionBlock",
        "Transformer",
        "Transformer_visual",
        "VisionTransformer",
        "CLIP_conv_proj",
        "convert_weights",
        "build_model_conv_proj",
    )
    hashes = {}
    for name in definitions:
        source_hash = _definition_ast_sha256(source_path, name)
        port_hash = _definition_ast_sha256(port_path, name)
        if source_hash != port_hash:
            raise ValueError(f"Prompt-aware CLIP definition differs: {name}")
        hashes[name] = source_hash
    prompt_source = _definition_ast_sha256(
        root / "models" / "ddp.py", "MLCPromptLearner"
    )
    prompt_port = _definition_ast_sha256(
        REPOSITORY_ROOT / "models" / "ddp.py", "MLCPromptLearner"
    )
    if prompt_source != prompt_port:
        raise ValueError("Original DDP prompt learner definition differs")
    hashes["MLCPromptLearner"] = prompt_source
    return {
        "prompt_aware_clip_ast_exact_match": True,
        "verified_definition_count": len(hashes),
        "verified_definition_ast_sha256": hashes,
    }


class _PromptOracle(nn.Module):
    def forward(self, cls_id: Tuple[int, int]):
        low, high = cls_id
        ids = torch.arange(low, high, dtype=torch.float64).unsqueeze(1)
        negative = torch.cat(
            [ids + 0.25, ids + 0.5, ids + 0.75, ids + 1.0], dim=1
        )
        positive = negative + 0.375
        prompts = torch.cat([negative, positive], dim=0)
        return prompts, torch.arange(2 * (high - low)).reshape(-1, 1)


class _TextOracle(nn.Module):
    def forward(self, prompts: torch.Tensor, tokenized: torch.Tensor):
        del tokenized
        return prompts


class _ImageOracle(nn.Module):
    def forward(self, images: torch.Tensor, visual_prompts: torch.Tensor):
        image_signal = images.flatten(1).mean(dim=1, keepdim=True)
        prompt_signal = visual_prompts[:, :, :4].mean(dim=1)
        token0 = prompt_signal + image_signal
        token1 = prompt_signal * 0.5 + image_signal * 1.5 + 0.2
        token2 = prompt_signal * 1.25 - image_signal * 0.25 + 0.4
        return torch.stack([token0, token1, token2], dim=1)


def _make_ddp_oracle(model_class, visual_prompts: torch.Tensor):
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.prompt_learner = _PromptOracle()
    model.text_encoder = _TextOracle()
    model.image_encoder = _ImageOracle()
    model.visual_prompts = nn.Parameter(visual_prompts.clone())
    model.text_feature_cache = {}
    model.n_cls = visual_prompts.shape[0] // 2
    model.l_vp = 16
    model.width = 768
    model.dtype = torch.float64
    model.device = torch.device("cpu")
    if model_class is PortDDP:
        model.feature_adapter = None
        model.feature_adapter_bank = None
        model.feature_adapter_correction = "linear_residual"
    return model


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach() - right.detach())).cpu())


def _text_encoder_forward_audit(root: Path) -> float:
    upstream_class = _extract_definition(
        root / "models" / "ddp.py",
        "TextEncoder",
        ast.ClassDef,
        {"torch": torch, "nn": nn},
    )

    def build(model_class):
        encoder = model_class.__new__(model_class)
        nn.Module.__init__(encoder)
        encoder.transformer = nn.Identity()
        encoder.positional_embedding = nn.Parameter(
            torch.linspace(-0.2, 0.3, 20, dtype=torch.float64).reshape(5, 4)
        )
        encoder.ln_final = nn.Identity()
        encoder.text_projection = nn.Parameter(
            torch.tensor(
                [
                    [0.3, 0.1, 0.2],
                    [0.2, 0.5, 0.4],
                    [0.7, 0.2, 0.1],
                    [0.1, 0.6, 0.3],
                ],
                dtype=torch.float64,
            )
        )
        encoder.dtype = torch.float64
        return encoder

    prompts = torch.linspace(
        -0.5, 0.8, 40, dtype=torch.float64
    ).reshape(2, 5, 4)
    tokenized = torch.tensor([[1, 2, 9, 3, 0], [1, 4, 5, 2, 9]])
    source = build(upstream_class)(prompts, tokenized)
    port = build(PortTextEncoder)(prompts, tokenized)
    return _max_abs(source, port)


def _source_pcd_constants(path: Path) -> Mapping[str, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ddp_class = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "DDP"
    )
    minimum = maximum = gamma = None
    for node in ast.walk(ddp_class):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(node.value, ast.Constant):
                if target.attr == "T_min":
                    minimum = float(node.value.value)
                elif target.attr == "T_max":
                    maximum = float(node.value.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
            and node.func.attr == "pow"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Constant)
        ):
            gamma = float(node.args[1].value)
    values = {"minimum": minimum, "maximum": maximum, "gamma": gamma}
    if values != {"minimum": 1.0, "maximum": 7.0, "gamma": 0.2}:
        raise ValueError(f"Unexpected source PCD constants: {values}")
    return values


def _operator_audit(root: Path) -> Dict[str, Any]:
    upstream_loss_class = _extract_definition(
        root / "bce_loss.py",
        "BCELoss",
        ast.ClassDef,
        {"torch": torch, "nn": nn},
    )
    upstream_ddp_class = _extract_definition(
        root / "models" / "ddp.py",
        "DDP",
        ast.ClassDef,
        {"torch": torch, "nn": nn, "F": F},
    )

    logits_source = torch.tensor(
        [[[0.2, -0.5], [1.1, 0.4]], [[-0.3, 0.8], [0.7, -0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits_port = logits_source.detach().clone().requires_grad_(True)
    targets = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    source_loss = upstream_loss_class()(logits_source, targets)
    port_loss = OriginalDDPLoss()(logits_port, targets)
    source_loss.backward()
    port_loss.backward()

    visual = torch.linspace(
        -0.4, 0.6, steps=8 * 16 * 768, dtype=torch.float64
    ).reshape(8, 16, 768)
    source_model = _make_ddp_oracle(upstream_ddp_class, visual)
    port_model = _make_ddp_oracle(PortDDP, visual)
    image = torch.linspace(
        0.0, 1.0, steps=2 * 3 * 2 * 2, dtype=torch.float64
    ).reshape(2, 3, 2, 2)

    source_task0 = source_model(image, cls_id=(0, 2), inference=False)
    port_task0 = port_model(image, cls_id=(0, 2), inference=False)
    source_task1 = source_model(image * 0.9, cls_id=(2, 4), inference=False)
    port_task1 = port_model(image * 0.9, cls_id=(2, 4), inference=False)
    source_seen = source_model(image, cls_id=(0, 4), inference=True)
    port_seen = port_model(image, cls_id=(0, 4), inference=True)

    seen_counts = (5, 8, 11, 14, 17, 20, 23, 26)
    registered = [
        original_ddp_temperature(count, 26, 5) for count in seen_counts
    ]
    expected_registered = [
        1.0 + ((count - 5) / 21.0) ** 0.7 for count in seen_counts
    ]
    source_pcd = _source_pcd_constants(root / "DDP.py")
    return {
        "bce_loss_abs_error": abs(
            float(source_loss.detach()) - float(port_loss.detach())
        ),
        "bce_gradient_max_abs_error": _max_abs(
            logits_source.grad, logits_port.grad
        ),
        "task0_forward_max_abs_error": _max_abs(source_task0, port_task0),
        "task1_forward_max_abs_error": _max_abs(source_task1, port_task1),
        "seen_inference_max_abs_error": _max_abs(source_seen, port_seen),
        "text_encoder_forward_max_abs_error": _text_encoder_forward_audit(root),
        "adapter_free": True,
        "source_pcd_temperature": source_pcd,
        "registered_pcd_temperature": {
            "minimum": 1.0,
            "maximum": 2.0,
            "gamma": 0.7,
        },
        "registered_temperature_values": registered,
        "registered_temperature_formula_max_abs_error": max(
            abs(left - right)
            for left, right in zip(registered, expected_registered)
        ),
        "pcd_is_deliberate_benchmark_deviation": True,
        "pcd_deviation_reason": (
            "user-requested direct comparison with modified DDP at T=1->2, gamma=0.7"
        ),
        **_prompt_aware_clip_audit(root),
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
            / "benchmarks/emotic_mlcil/methods/original_ddp/method.py"
        ),
        "repository_ddp_sha256": _sha256(REPOSITORY_ROOT / "models/ddp.py"),
        "comparison_script_sha256": _sha256(Path(__file__).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-archive")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.source_root).resolve()
    archive = Path(args.source_archive).resolve() if args.source_archive else None
    payload = {
        "schema_version": 1,
        "method": "Original-DDP-Tau2",
        "source": _verify_source(root, archive),
        "operator_audit": _operator_audit(root),
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
