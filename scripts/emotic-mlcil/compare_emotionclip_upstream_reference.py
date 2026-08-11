#!/usr/bin/env python3
"""Verify the independent EmotionCLIP visual port against fixed source."""

from __future__ import annotations

import argparse
import __future__
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.emotionclip_ft import EmotionCLIPVisualTransformer


UPSTREAM_COMMIT = "ca142dd4c9664acb2b59c8b14ef3169049f1180b"
UPSTREAM_TREE_SHA256 = "c313af40432b83d4ef41700fb216de2ddecc9ecb882390d939e0b0c9690f79a7"
EXPECTED_FILES = {
    "src/models/base.py": "92664be471e43d9e9514d51d2c3b7d9f7e08e5db5afae3f7d9e9868662e08004",
    "src/datasets/emotic.py": "0a6e427c6b7e50dd890d998ac3ca0f3142fa3384fffb01dfad5fc09bfea0c316",
    "src/models/model_configs/ViT-B-32.json": "743f4fcee3aea1705c1865f89a3767d816513769f57eb483a9eea3e610ebe633",
    "linear_eval.py": "44449794271ad4405869739a17cafaa6a722aa844af707e18c9d7d4111775300",
    "src/options.py": "f4ef471ccbeb24f18bdf0a18a41d997eb6b5466375309bc722f18791d26c7cb7",
    "LICENSE": "ce64e3784936a46f2e6a73038f11198093f1cc42aa8017c785de628697bf7f44",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_upstream_module(path: Path) -> types.ModuleType:
    """Load the fixed Python-3.10 source without editing the snapshot.

    EmotionCLIP uses ``A | B`` annotations that Python 3.9 otherwise evaluates
    eagerly.  Compiling with the standard postponed-annotations future flag
    preserves the upstream executable operators while keeping the immutable
    source bytes covered by ``EXPECTED_FILES`` unchanged.
    """

    name = "fixed_emotionclip_base"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    code = compile(
        path.read_text(encoding="utf-8"),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, module.__dict__)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    observed = {}
    for relative, expected in EXPECTED_FILES.items():
        path = args.upstream_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256(path)
        if observed[relative] != expected:
            raise ValueError(f"EmotionCLIP source differs: {relative}")
    observed_git_commit = None
    snapshot_path = args.upstream_root / "SOURCE_SNAPSHOT.json"
    if (args.upstream_root / ".git").exists():
        observed_git_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=args.upstream_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_git_commit != UPSTREAM_COMMIT:
            raise ValueError("EmotionCLIP Git checkout differs from fixed commit")
    elif snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("commit") != UPSTREAM_COMMIT:
            raise ValueError("EmotionCLIP snapshot commit differs")
    else:
        raise FileNotFoundError(
            "EmotionCLIP source requires either fixed .git metadata or SOURCE_SNAPSHOT.json"
        )

    module = load_upstream_module(args.upstream_root / "src/models/base.py")
    torch.manual_seed(31)
    source = module.VisualTransformer(32, 16, 16, 2, 2, 4.0, 8).eval()
    port = EmotionCLIPVisualTransformer(32, 16, 16, 2, 2, 4.0, 8).eval()
    port.load_state_dict(source.state_dict(), strict=True)
    image = torch.randn(3, 3, 32, 32)
    mask = torch.zeros(3, 32, 32)
    mask[0, 2:17, 4:23] = 1
    mask[1, 8:31, 1:15] = 1
    with torch.no_grad():
        expected = source(image, mask)
        actual = port(image, mask)
    max_error = float((expected - actual).abs().max())
    if max_error > 1.0e-7:
        raise ValueError(f"EmotionCLIP operator mismatch: {max_error}")
    linear_eval = (args.upstream_root / "linear_eval.py").read_text(encoding="utf-8")
    for marker in ("F.normalize(features, dim=-1)", "C=2.5", "OneVsRestClassifier"):
        if marker not in linear_eval:
            raise ValueError(f"EmotionCLIP linear-eval marker missing: {marker}")
    payload = {
        "schema_version": 1,
        "method": "EmotionCLIP-FT",
        "upstream": {
            "repository": "https://github.com/Xeaver/EmotionCLIP",
            "commit": UPSTREAM_COMMIT,
            "snapshot_tree_sha256": UPSTREAM_TREE_SHA256,
            "observed_git_commit": observed_git_commit,
            "verified_file_sha256": observed,
            "source_copied_into_repository": False,
        },
        "operator_equivalence": {"visual_forward_max_abs_error": max_error},
        "static_reference": {
            "emotic_mAP": 32.91,
            "protocol": "normalized frozen features + one-vs-rest LogisticRegression(C=2.5)",
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
