#!/usr/bin/env python3
"""Freeze externally prepared CocoER head boxes into the benchmark schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COMMIT = "dac8fc139e61b87f1bf0b27c581798df2a5a9d38"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a JSON sample-id→[x1,y1,x2,y2] mapping produced with the "
            "fixed CocoER/InsightFace preprocessing and write an immutable cache"
        )
    )
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detector-name", required=True)
    parser.add_argument("--detector-model-sha256", required=True)
    args = parser.parse_args()
    raw = json.loads(args.detections.read_text(encoding="utf-8"))
    entries = raw.get("entries", raw) if isinstance(raw, dict) else None
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
        "detector_name": args.detector_name,
        "detector_model_sha256": args.detector_model_sha256.lower(),
        "source_detection_json_sha256": source_sha,
        "entries": normalized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(normalized),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
