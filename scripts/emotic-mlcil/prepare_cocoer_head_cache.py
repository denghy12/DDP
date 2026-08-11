#!/usr/bin/env python3
"""Freeze externally prepared CocoER head boxes into the benchmark schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COMMIT = "dac8fc139e61b87f1bf0b27c581798df2a5a9d38"
BUFFALO_L_TREE_SHA256 = "50fa1383e97d137f2902b53de7b7305ffbd35eb4ae32135d95d1e25d5a9d9d3d"


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
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != 1:
        raise ValueError("Detection JSON is not a CocoER detection artifact")
    if raw.get("upstream_commit") != COMMIT:
        raise ValueError("Detection JSON CocoER commit differs")
    if raw.get("partial") is not False:
        raise ValueError("Partial CocoER detections cannot be frozen")
    unresolved = raw.get("unresolved")
    if not isinstance(unresolved, list) or unresolved:
        raise ValueError("Unresolved CocoER head detections cannot be frozen")
    detector = raw.get("detector")
    if (
        not isinstance(detector, dict)
        or detector.get("library") != "insightface"
        or detector.get("version") != "0.7.3"
        or detector.get("model") != "buffalo_l"
        or detector.get("det_size") != [640, 640]
    ):
        raise ValueError("CocoER detector identity differs from the frozen interface")
    model_sha = str(detector.get("model_tree_sha256", "")).lower()
    if model_sha != BUFFALO_L_TREE_SHA256:
        raise ValueError("CocoER detector model-tree SHA-256 differs")
    entries = raw.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Detection JSON must contain a non-empty mapping")
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
    source_sha = hashlib.sha256(args.detections.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "upstream_repository": "https://github.com/bisno/CocoER",
        "upstream_commit": COMMIT,
        "detector": dict(detector),
        "detector_model_tree_sha256": model_sha,
        "source_detection_json_sha256": source_sha,
        "entries": normalized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(normalized),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
