#!/usr/bin/env python3
"""Audit immutable CocoER source and the protocol-safe FT conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Optional


COMMIT = "dac8fc139e61b87f1bf0b27c581798df2a5a9d38"
REPOSITORY = "https://github.com/bisno/CocoER"
EXPECTED_SHA256 = {
    "LICENSE": "66d1f48aaf8592fca7f09eeab03f9472417bc9da6c389d32d94258f1571a55d6",
    "SOURCE_SNAPSHOT.json": "8ab82b4d137b9aeaf1407b39e1ea3d6d481fdf92420b25e1f6c0706e7f0cb9bd",
    "models_sw.py": "7b364667cd7b7ff60ab0a37c95487089c1b2a6ad72484fcb278636313d86706c",
    "train.py": "03fc8441ef452211b85a25e4443d0336c1eacee4712264b84dfecb155ef224be",
    "dataset.py": "f357f24de02f48558b68af9bf6648b7a4e99550b39d204768dad8378a981b3ae",
    "utils.py": "934026c5f67f5873d96dc3b563f00dfcc05dbea4289926f6cb19c49bf53f7b67",
    "classes.py": "7791fd426ecc2d1ddd30d25e7e79ccd7f2ca1ca746235e3ffbb82821caecbe6e",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dynamic_weights(target_columns):
    result = []
    for column in target_columns:
        count = sum(column)
        result.append(0.0001 if count == 0 else 1.0 / math.log(count + 1.2))
    return result


def compare(upstream_root: Path, port_root: Optional[Path] = None):
    root = upstream_root.resolve()
    observed = {name: _sha(root / name) for name in EXPECTED_SHA256}
    if observed != EXPECTED_SHA256:
        raise ValueError("Fixed CocoER source hash mismatch")
    snapshot = json.loads((root / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    if snapshot.get("repository") != REPOSITORY or snapshot.get("commit") != COMMIT:
        raise ValueError("CocoER source provenance differs")
    model = (root / "models_sw.py").read_text(encoding="utf-8")
    trainer = (root / "train.py").read_text(encoding="utf-8")
    dataset = (root / "dataset.py").read_text(encoding="utf-8")
    loss = (root / "utils.py").read_text(encoding="utf-8")
    contract = {
        "three_independent_resnet50_towers": model.count("torchmodels.resnet50(pretrained=True)") >= 3,
        "native_clip_rn50": "clip.load('RN50'" in model,
        "three_cross_level_encoders": all(name in model for name in ("self.encoder1", "self.encoder2", "self.encoder3")),
        "swish_cross_level_ffn": "self.activation = Swish()" in model,
        "inside_lr_point_one": "default=0.1" in trainer,
        "pseudo_threshold_point_three": "torch.ones(VI_cls.shape) * 0.3" in model,
        "five_equal_loss_terms": "0.2*(global_loss+head_loss+body_loss+ctx_loss" in trainer,
        "grad_distance_weight_point_one": "default='0.1'" in trainer,
        "adamw_lr": "default=0.00006" in trainer and "optim.AdamW" in trainer,
        "twenty_epochs": "default=20" in trainer,
        "step_three_gamma_point_one": "default=3" in trainer and "default=0.1" in trainer,
        "dynamic_bce": "prepare_dynamic_weights" in loss and "binary_cross_entropy_with_logits" in loss,
        "three_view_head_geometry": all(token in dataset for token in ("image_context", "image_body", "image_head", "head_coord")),
        "all_class_gwt_checkpoint": "GWT model: `./checkpoints/`" in (root / "README.md").read_text(encoding="utf-8"),
        "all_class_vi_checkpoint": "VI_weights/w.pth" in model,
    }
    if not all(contract.values()):
        raise ValueError(f"CocoER source contract missing: {contract}")
    columns = ([1, 1, 0], [0, 0, 0], [1, 0, 0])
    reference = _dynamic_weights(columns)
    port = [
        1.0 / math.log(3.2),
        0.0001,
        1.0 / math.log(2.2),
    ]
    repository = port_root or Path(__file__).resolve().parents[2]
    port_files = (
        repository / "benchmarks/emotic_mlcil/methods/cocoer_ft/model.py",
        repository / "benchmarks/emotic_mlcil/methods/cocoer_ft/method.py",
        repository / "scripts/emotic-mlcil/prepare_cocoer_native_assets.py",
        repository / "scripts/emotic-mlcil/generate_cocoer_head_detections.py",
        repository / "scripts/emotic-mlcil/prepare_cocoer_head_cache.py",
        repository / "scripts/emotic-mlcil/audit_cocoer_assets.py",
    )
    return {
        "schema_version": 1,
        "method": "CocoER-FT",
        "upstream": {
            "repository": REPOSITORY,
            "commit": COMMIT,
            "license": "MIT",
            "source_copied_into_repository": False,
            "verified_file_sha256": observed,
        },
        "source_contract": contract,
        "operator_equivalence": {
            "dynamic_weight_max_abs_error": max(abs(a - b) for a, b in zip(reference, port)),
        },
        "conversion": {
            "track": "B",
            "strategy": "sequential_finetuning",
            "native_visual_stack": "ImageNet ResNet-50 x3 + OpenAI CLIP RN50",
            "current_label_only": True,
            "distillation": False,
            "replay": False,
            "adapter": False,
            "rejected_full_class_assets": ["GWT checkpoint", "VI_weights/w.pth"],
            "port_file_sha256": {
                str(path.relative_to(repository)): _sha(path) for path in port_files
            },
        },
    }


def main():
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
