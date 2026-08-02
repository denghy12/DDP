#!/usr/bin/env python3
"""Compare the independent MULTI-LANE port with fixed external source code.

The fixed upstream ``PreT_Attention`` and ``forward_head`` bodies are compiled
from the external extraction at runtime.  No upstream source is copied into
the benchmark repository.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.multi_lane.model import MultiLaneModel


UPSTREAM_COMMIT = "5ee982c9298d4cfd6af471d9bb2ef3c0aad05373"
UPSTREAM_ARCHIVE_SHA256 = (
    "dfe84ea31f6d7e51877c2661791c716aed7d888b8d783d16ce067cb8ce022d49"
)
UPSTREAM_TREE_SHA256 = (
    "f5e8ffee846121626444f7e14d0513d00a64739657eda1067b911218acaefe99"
)
UPSTREAM_BLOCKS_SHA256 = (
    "79f1d5496919a22a6c940d3caac5ff937ab8c6b2e7a9a186fb9034ef1ae24af7"
)
UPSTREAM_VIT_SHA256 = (
    "6d4777e79888b35ba3816ea60373bec7d9d503c9dbdd507cef72090ebfda3e95"
)


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
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_upstream(root: Path, archive: Optional[Path]) -> Dict[str, Any]:
    blocks = root / "multi_lane" / "blocks.py"
    vision_transformer = root / "multi_lane" / "vision_transformer.py"
    if not blocks.is_file() or not vision_transformer.is_file():
        raise FileNotFoundError(
            "Expected fixed MULTI-LANE extraction with blocks.py and "
            "vision_transformer.py"
        )
    observed = {
        "tree_sha256": _tree_sha256(root),
        "blocks_sha256": _sha256(blocks),
        "vision_transformer_sha256": _sha256(vision_transformer),
    }
    expected = {
        "tree_sha256": UPSTREAM_TREE_SHA256,
        "blocks_sha256": UPSTREAM_BLOCKS_SHA256,
        "vision_transformer_sha256": UPSTREAM_VIT_SHA256,
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            raise ValueError(
                f"MULTI-LANE upstream {key} differs: "
                f"{observed[key]} != {expected_value}"
            )
    archive_sha = None
    if archive is not None:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archive_sha = _sha256(archive)
        if archive_sha != UPSTREAM_ARCHIVE_SHA256:
            raise ValueError(
                "MULTI-LANE archive SHA-256 differs: "
                f"{archive_sha} != {UPSTREAM_ARCHIVE_SHA256}"
            )
    return {
        "repository": "https://github.com/tdemin16/multi-lane",
        "commit": UPSTREAM_COMMIT,
        "archive_sha256": archive_sha or UPSTREAM_ARCHIVE_SHA256,
        **observed,
        "source_copied_into_repository": False,
        "oracle_mode": "runtime_AST_extraction_from_external_fixed_source",
    }


def _extract_class(path: Path, class_name: str, namespace: Dict[str, Any]):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        (
            child
            for child in tree.body
            if isinstance(child, ast.ClassDef) and child.name == class_name
        ),
        None,
    )
    if node is None:
        raise ValueError(f"Fixed upstream source has no {class_name}")
    extracted = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace[class_name]


def _extract_forward_head(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VisionTransformer":
            method = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef)
                    and child.name == "forward_head"
                ),
                None,
            )
            break
    if method is None:
        raise ValueError("Fixed upstream source has no forward_head")
    oracle = ast.ClassDef(
        name="ForwardHeadOracle",
        bases=[],
        keywords=[],
        body=[copy.deepcopy(method)],
        decorator_list=[],
    )
    extracted = ast.Module(body=[oracle], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: Dict[str, Any] = {"torch": torch, "F": F}
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace["ForwardHeadOracle"]


class _TinyResidualBlock(nn.Module):
    def __init__(self, width: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 2)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(values)
        attended = self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        values = values + attended
        return values + self.mlp(self.ln_2(values))


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([_TinyResidualBlock()])


class _TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=1, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(8))
        self.positional_embedding = nn.Parameter(torch.randn(5, 8))
        self.ln_pre = nn.LayerNorm(8)
        self.transformer = _TinyTransformer()
        self.ln_post = nn.LayerNorm(8)
        self.proj = nn.Parameter(torch.randn(8, 6))
        self.output_dim = 6


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach() - right.detach())))


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
        "model_sha256": _sha256(
            REPOSITORY_ROOT
            / "benchmarks/emotic_mlcil/methods/multi_lane/model.py"
        ),
        "comparison_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _attention_oracle(upstream_root: Path) -> Dict[str, float]:
    namespace: Dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
    oracle_class = _extract_class(
        upstream_root / "multi_lane" / "blocks.py",
        "PreT_Attention",
        namespace,
    )
    torch.manual_seed(2026080204)
    model = MultiLaneModel(
        _TinyVisual(),
        task_sizes=(2, 2),
        num_selectors=3,
        num_prompts=2,
        num_prompt_layers=1,
        normalize="pre-head",
    )
    block = model.visual_encoder.transformer.resblocks[0]
    oracle = oracle_class(8, num_heads=2, qkv_bias=True, id=0)
    oracle.init(
        SimpleNamespace(
            num_selectors=3,
            detach=False,
            disable_dandr=False,
            tome=0,
            num_tasks=2,
            num_prompts=2,
            prompt_init="orthogonal",
        ),
        use_prompts=True,
    )
    with torch.no_grad():
        oracle.qkv.weight.copy_(block.attn.in_proj_weight)
        oracle.qkv.bias.copy_(block.attn.in_proj_bias)
        oracle.proj.weight.copy_(block.attn.out_proj.weight)
        oracle.proj.bias.copy_(block.attn.out_proj.bias)
        oracle.prompts.copy_(model.prompts[0])

    errors: Dict[str, float] = {}
    model.activate_task(0)
    for mode, lane_ids in (("train_current", [0]), ("eval_seen", [0, 1])):
        if mode == "eval_seen":
            model.activate_task(1)
            oracle.next_task()
            oracle.eval()
        else:
            oracle.t = 0
            oracle.train()
        image_tokens = torch.randn(3, 5, 8)
        lane_tokens = model._initial_lane_tokens(3, lane_ids)
        normalized_image = block.ln_1(image_tokens)
        normalized_lane = block.ln_1(lane_tokens)
        _, upstream_update = oracle(normalized_image, normalized_lane)
        upstream_output = lane_tokens + upstream_update
        upstream_output = upstream_output + block.mlp(
            block.ln_2(upstream_output)
        )
        port_output = model._lane_block(
            block, image_tokens, lane_tokens, lane_ids, 0
        )
        errors[f"{mode}_lane_block_max_abs_error"] = _max_abs(
            upstream_output, port_output
        )
    return errors


def _concat_oracle(upstream_root: Path) -> Dict[str, float]:
    oracle_class = _extract_forward_head(
        upstream_root / "multi_lane" / "vision_transformer.py"
    )
    torch.manual_seed(2026080205)
    model = MultiLaneModel(
        _TinyVisual(),
        task_sizes=(2, 2),
        num_selectors=3,
        num_prompts=2,
        num_prompt_layers=1,
        normalize="pre-head",
    )
    model.activate_task(0)
    model.activate_task(1)
    model.eval()
    images = torch.randn(4, 1, 2, 2)
    lane_features = model.encode_lanes(images, all_seen_lanes=True)
    task_tokens = lane_features.permute(1, 0, 2).unsqueeze(2)
    oracle = oracle_class()
    oracle.global_pool = "token"
    oracle.head_mode = "concat"
    oracle.normalize = "none"
    oracle.head = model.head
    oracle.class_mask = [[0, 1], [2, 3]]
    frozen_tokens = torch.zeros(4, 1, model.output_dim)
    upstream_logits = oracle.forward_head(
        frozen_tokens, task_tokens, eval=True
    )[0]
    port_logits = model.seen_logits(images)
    return {
        "concat_logits_max_abs_error": _max_abs(
            upstream_logits, port_logits
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-archive", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream_root = args.upstream_root.resolve()
    payload = {
        "schema_version": 1,
        "comparison": (
            "independent_multi_lane_track_a_vs_fixed_upstream_execution_oracle"
        ),
        "scope": (
            "same-input same-weight selector/prompt/drop-and-replace and "
            "concat-mask audit; not an EMOTIC result"
        ),
        "upstream": _verify_upstream(
            upstream_root,
            args.upstream_archive.resolve()
            if args.upstream_archive is not None
            else None,
        ),
        "port": _port_provenance(),
        "operator_equivalence": {
            **_attention_oracle(upstream_root),
            **_concat_oracle(upstream_root),
        },
    }
    worst = max(payload["operator_equivalence"].values())
    payload["operator_equivalence"]["worst_max_abs_error"] = worst
    if worst >= 1.0e-6:
        raise RuntimeError(
            f"MULTI-LANE upstream equivalence failed: worst error {worst}"
        )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
