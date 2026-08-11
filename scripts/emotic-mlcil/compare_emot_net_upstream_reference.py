#!/usr/bin/env python3
"""Audit the fixed Torch7 EMOT-Net source against the Track-B FT port."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional


COMMIT = "69c3a5106aed08121cd12f6a5b359c745136931e"
REPOSITORY = "https://github.com/rkosti/emotic"
EXPECTED_SHA256 = {
    "LICENSE": "356aeea274023317f0d57e47c32de3f512e06f6e91fe33700ef1f68f7e8a1161",
    "SOURCE_SNAPSHOT.json": "5f2726b80cd42ddb43de77442d5dbd64d2ed3b755a130f066af0c5aa51df81fa",
    "src/create_model.lua": "acd2d9e6838cf1d6053d86c86692f04dd7222f9a87b274103a50f842cd3fef13",
    "src/emotic_cnn_model_structure.txt": "2ed8c3024d87b79bca0335235ea58f9e76ce98cff928312dff47e14116b9b7d6",
    "src/funcs.lua": "1f5c07081854238f37bcd3ceb3f6db590ec53c431598ae72a5b3fe94c6f9a22f",
    "src/main.lua": "f82f99929840b6708aeb33f758aa96145d43c6b752b5b3d058bcfc1308002882",
    "src/opts.lua": "8c6cb4b411fb66ec4a37b52d57c7e8487783fa3029619b92aabf6f7cf53bb1e1",
    "src/single_image_inference.lua": "e3de39410d5508bc2b57256862b4ff927141500c331952e3d4ad376c5d1f1f6f",
    "src/train_test.lua": "96491273d052cac615ed83f3d70be2252ab18a4257971872b1ffc19bd601a0ca",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_weights(counts, norm_factor):
    return [
        0.0001 if count < 1 else 1.0 / math.log(norm_factor + count)
        for count in counts
    ]


def _port_weights(counts, norm_factor):
    result = [0.0001 for _ in counts]
    for index, count in enumerate(counts):
        if count >= 1:
            result[index] = 1.0 / math.log(count + norm_factor)
    return result


def _reference_loss(logits, targets, weights):
    values = []
    for row_logits, row_targets in zip(logits, targets):
        for logit, target, weight in zip(row_logits, row_targets, weights):
            probability = 1.0 / (1.0 + math.exp(-logit))
            values.append(weight * (probability - target) ** 2)
    return sum(values) / len(values)


def compare(upstream_root: Path, port_root: Optional[Path] = None) -> Dict[str, object]:
    root = upstream_root.resolve()
    observed = {name: _sha(root / name) for name in EXPECTED_SHA256}
    if observed != EXPECTED_SHA256:
        mismatches = {
            name: {"expected": EXPECTED_SHA256[name], "observed": observed.get(name)}
            for name in EXPECTED_SHA256
            if observed.get(name) != EXPECTED_SHA256[name]
        }
        raise ValueError(f"Fixed EMOT-Net source hash mismatch: {mismatches}")
    snapshot = json.loads((root / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    if snapshot.get("commit") != COMMIT or snapshot.get("repository") != REPOSITORY:
        raise ValueError("EMOT-Net source snapshot provenance differs")

    options = (root / "src/opts.lua").read_text(encoding="utf-8")
    create_model = (root / "src/create_model.lua").read_text(encoding="utf-8")
    functions = (root / "src/funcs.lua").read_text(encoding="utf-8")
    structure = (root / "src/emotic_cnn_model_structure.txt").read_text(encoding="utf-8")
    main_source = (root / "src/main.lua").read_text(encoding="utf-8")
    inference_source = (root / "src/single_image_inference.lua").read_text(encoding="utf-8")
    required = {
        "native_context_default": "model_myVDavg_640_Places.t7" in options,
        "native_body_default": "myVD_ImgNet_66_old.t7" in options,
        "body_context_parallel": "parallel:add(imageModel)" in create_model and "parallel:add(bodyModel)" in create_model,
        "fusion_768_to_256": "nn.Linear(768 -> 256)" in structure,
        "dropout_half": "nn.Dropout(0.500000)" in structure,
        "sigmoid_output": "model:add(nn.Sigmoid())" in create_model,
        "weighted_mse": "nn.weightMSE(gClass_weights,true)" in create_model,
        "joint_discrete_weight_half": "cmd:option('-Wdisc',            1/2" in options,
        "published_structure_fusion_bn_relu": "nn.BatchNormalization (2D) (256)" in structure and "nn.ReLU" in structure,
        "executable_bi_fusion_bn_relu_commented": "--class:add(nn.BatchNormalization(DROP_FIRST_CLASS,1e-3))" in create_model and "--class:add(nn.ReLU(true))" in create_model,
        "weight_formula": "1 / (torch.log(nF + normHist[1][i]))" in functions,
        "epochs_14": bool(re.search(r"-nEpochs'\s*,\s*14", options)),
        "lr_drop_7": bool(re.search(r"-nItersLR'\s*,\s*7", options)),
        "batch_52": "-batchSize',    26*2" in options,
        "body_128_main_comment": "IMGSizes.BODY_SIZE = 224 --default is 128" in main_source,
        "body_128_inference": "IMGSizes.BODY_SIZE = 128" in inference_source,
    }
    if not all(required.values()):
        raise ValueError(f"EMOT-Net source contract missing: {required}")

    counts = [0.0, 1.0, 3.0, 19.0]
    reference_weights = _reference_weights(counts, 1.2)
    port_weights = _port_weights(counts, 1.2)
    weight_error = max(abs(a - b) for a, b in zip(reference_weights, port_weights))
    logits = [[-1.2, 0.0, 0.8, 2.1], [1.3, -0.7, 0.2, -2.0]]
    targets = [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
    loss_reference = _reference_loss(logits, targets, reference_weights)
    loss_port = _reference_loss(logits, targets, port_weights)

    repository = port_root or Path(__file__).resolve().parents[2]
    port_files = (
        repository / "benchmarks/emotic_mlcil/methods/emot_net_ft/model.py",
        repository / "benchmarks/emotic_mlcil/methods/emot_net_ft/method.py",
    )
    return {
        "schema_version": 1,
        "method": "EMOT-Net-FT",
        "upstream": {
            "repository": REPOSITORY,
            "commit": COMMIT,
            "license": "MIT",
            "source_copied_into_repository": False,
            "verified_file_sha256": observed,
            "pretrained_assets_bundled_in_git": False,
            "registered_native_context_asset": "model_myVDavg_640_Places.t7",
            "registered_native_body_asset": "myVD_ImgNet_66_old.t7",
        },
        "source_contract": required,
        "operator_equivalence": {
            "class_weight_max_abs_error": weight_error,
            "weighted_sigmoid_mse_abs_error": abs(loss_reference - loss_port),
        },
        "conversion": {
            "track": "B",
            "strategy": "sequential_finetuning",
            "input": "full context + annotated person crop",
            "current_label_only": True,
            "distillation": False,
            "replay": False,
            "adapter": False,
            "clip_used": False,
            "port_file_sha256": {path.name: _sha(path) for path in port_files},
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
