#!/usr/bin/env python3
"""Freeze externally prepared CocoER head boxes into the benchmark schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


COMMIT = "dac8fc139e61b87f1bf0b27c581798df2a5a9d38"
BUFFALO_L_TREE_SHA256 = "50fa1383e97d137f2902b53de7b7305ffbd35eb4ae32135d95d1e25d5a9d9d3d"
CONVERSION_NAME = "buffalo_l_strict_then_train_median_v0.1"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a JSON sample-id→[x1,y1,x2,y2] mapping produced with the "
            "fixed CocoER/InsightFace preprocessing and write an immutable cache"
        )
    )
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.detections.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != 2:
        raise ValueError("Detection JSON is not a CocoER detection artifact")
    if raw.get("upstream_commit") != COMMIT:
        raise ValueError("Detection JSON CocoER commit differs")
    if raw.get("partial") is not False:
        raise ValueError("Partial CocoER detections cannot be frozen")
    unresolved = raw.get("unresolved")
    if (
        not isinstance(unresolved, list)
        or unresolved
        or int(raw.get("unresolved_samples", -1)) != 0
    ):
        raise ValueError("Unresolved CocoER head geometry cannot be frozen")
    detector = raw.get("detector")
    if (
        not isinstance(detector, dict)
        or detector.get("library") != "insightface"
        or detector.get("version") != "0.7.3"
        or detector.get("model") != "buffalo_l"
        or detector.get("detector_file") != "det_10g.onnx"
        or detector.get("implementation") != "insightface_scrfd_only"
        or detector.get("det_size") != [640, 640]
    ):
        raise ValueError("CocoER detector identity differs from the frozen interface")
    actual_providers = detector.get("actual_providers")
    if not isinstance(actual_providers, list) or not actual_providers:
        raise ValueError("CocoER actual ONNX providers are missing")
    requested_device = detector.get("requested_device")
    expected_provider = (
        "CUDAExecutionProvider"
        if requested_device == "cuda"
        else "CPUExecutionProvider"
    )
    if requested_device not in {"cuda", "cpu"} or actual_providers[0] != expected_provider:
        raise ValueError("CocoER actual ONNX provider differs from the request")
    equivalence = detector.get("faceanalysis_equivalence")
    faceanalysis_providers = (
        equivalence.get("faceanalysis_detector_actual_providers")
        if isinstance(equivalence, dict)
        else None
    )
    if (
        not isinstance(equivalence, dict)
        or int(equivalence.get("samples", 0)) <= 0
        or equivalence.get("integer_boxes_exact_match") is not True
        or float(equivalence.get("max_abs_error_after_integer_clipping", math.inf))
        != 0.0
        or not isinstance(faceanalysis_providers, list)
        or not faceanalysis_providers
        or faceanalysis_providers[0] != expected_provider
    ):
        raise ValueError("CocoER SCRFD-only equivalence was not established")
    model_sha = str(detector.get("model_tree_sha256", "")).lower()
    if model_sha != BUFFALO_L_TREE_SHA256:
        raise ValueError("CocoER detector model-tree SHA-256 differs")
    conversion = raw.get("conversion")
    if (
        not isinstance(conversion, dict)
        or conversion.get("name") != CONVERSION_NAME
        or conversion.get("sample_preserving") is not True
        or conversion.get("native_source") != "buffalo_l_strict"
        or conversion.get("fallback_source") != "train_median_relative_geometry"
        or conversion.get("fallback_calibration_split") != "train"
        or conversion.get("fallback_uses_labels") is not False
        or conversion.get("fallback_uses_val_or_test_statistics") is not False
        or conversion.get("fallback_statistic") != "componentwise_median"
        or conversion.get("fallback_rounding")
        != "clip_to_image_then_round_half_up"
    ):
        raise ValueError("CocoER head conversion contract differs")
    median_box = conversion.get("median_relative_head_box")
    if (
        not isinstance(median_box, list)
        or len(median_box) != 4
        or not all(math.isfinite(float(value)) for value in median_box)
        or float(median_box[2]) <= float(median_box[0])
        or float(median_box[3]) <= float(median_box[1])
        or int(conversion.get("train_native_calibration_samples", 0)) <= 0
    ):
        raise ValueError("CocoER train-only fallback calibration is invalid")
    entries = raw.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Detection JSON must contain a non-empty mapping")
    fallback_sample_ids = raw.get("fallback_sample_ids")
    if (
        not isinstance(fallback_sample_ids, list)
        or len(fallback_sample_ids) != len(set(fallback_sample_ids))
        or len(fallback_sample_ids) != int(raw.get("fallback_samples", -1))
        or len(fallback_sample_ids) != int(raw.get("native_unresolved_samples", -1))
    ):
        raise ValueError("CocoER fallback sample accounting differs")
    normalized = {}
    for sample_id, box in sorted(entries.items()):
        if not isinstance(sample_id, str) or not sample_id.startswith("emotic:"):
            raise ValueError(f"Invalid EMOTIC sample ID: {sample_id!r}")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Invalid head box for {sample_id}")
        values = [float(value) for value in box]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"Degenerate head box for {sample_id}")
        normalized[sample_id] = values
    if not set(fallback_sample_ids).issubset(normalized):
        raise ValueError("CocoER fallback IDs are absent from the head mapping")
    processed_samples = int(raw.get("processed_samples", -1))
    native_resolved_samples = int(raw.get("native_resolved_samples", -1))
    native_unresolved_samples = int(raw.get("native_unresolved_samples", -1))
    fallback_samples = int(raw.get("fallback_samples", -1))
    if (
        processed_samples != len(normalized)
        or native_resolved_samples + fallback_samples != len(normalized)
        or native_unresolved_samples != fallback_samples
    ):
        raise ValueError("CocoER sample totals differ from the head mapping")
    split_fields = (
        "processed_by_split",
        "native_resolved_by_split",
        "native_unresolved_by_split",
        "fallback_by_split",
    )
    split_counts = {}
    for field in split_fields:
        counts = raw.get(field)
        if not isinstance(counts, dict) or any(
            split not in {"train", "val", "test"}
            or not isinstance(value, int)
            or value < 0
            for split, value in counts.items()
        ):
            raise ValueError(f"CocoER {field} is invalid")
        split_counts[field] = dict(counts)
    if (
        sum(split_counts["processed_by_split"].values()) != processed_samples
        or sum(split_counts["native_resolved_by_split"].values())
        != native_resolved_samples
        or sum(split_counts["native_unresolved_by_split"].values())
        != native_unresolved_samples
        or sum(split_counts["fallback_by_split"].values()) != fallback_samples
    ):
        raise ValueError("CocoER split totals differ from the sample totals")
    for split in ("train", "val", "test"):
        if (
            split_counts["native_resolved_by_split"].get(split, 0)
            + split_counts["fallback_by_split"].get(split, 0)
            != split_counts["processed_by_split"].get(split, 0)
            or split_counts["native_unresolved_by_split"].get(split, 0)
            != split_counts["fallback_by_split"].get(split, 0)
        ):
            raise ValueError(f"CocoER split accounting differs for {split}")
    source_sha = hashlib.sha256(args.detections.read_bytes()).hexdigest()
    payload = {
        "schema_version": 2,
        "upstream_repository": "https://github.com/bisno/CocoER",
        "upstream_commit": COMMIT,
        "detector": dict(detector),
        "conversion": dict(conversion),
        "detector_model_tree_sha256": model_sha,
        "source_detection_json_sha256": source_sha,
        "processed_samples": processed_samples,
        "native_resolved_samples": native_resolved_samples,
        "native_unresolved_samples": native_unresolved_samples,
        "fallback_samples": len(fallback_sample_ids),
        "processed_by_split": split_counts["processed_by_split"],
        "native_resolved_by_split": split_counts["native_resolved_by_split"],
        "native_unresolved_by_split": split_counts["native_unresolved_by_split"],
        "fallback_by_split": split_counts["fallback_by_split"],
        "fallback_sample_ids": sorted(fallback_sample_ids),
        "entries": normalized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(normalized),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
