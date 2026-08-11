#!/usr/bin/env python3
"""Verify immutable DSCT source identity and copied mathematical operators."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.emotic_mlcil.methods.dsct_ft.method import (
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    UPSTREAM_REPOSITORY,
    verify_upstream_source,
)
from benchmarks.emotic_mlcil.methods.dsct_ft.model import target_query_indices


def source_face_matching(reference, boxes):
    area_reference = (reference[2] - reference[0] + 1) * (reference[3] - reference[1] + 1)
    area_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(reference[0], boxes[:, 0]); yy1 = np.maximum(reference[1], boxes[:, 1])
    xx2 = np.minimum(reference[2], boxes[:, 2]); yy2 = np.minimum(reference[3], boxes[:, 3])
    intersection = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
    return int((intersection / (area_reference + area_boxes - intersection)).argsort()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    hashes = verify_upstream_source(args.upstream_root)
    predicted = torch.tensor([[[0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.3, 0.4], [0.8, 0.8, 0.1, 0.1]]])
    target = torch.tensor([[0.5, 0.5, 0.3, 0.4]])
    ours = int(target_query_indices(predicted, target, torch.tensor([[100.0, 100.0]])).item())
    boxes = np.array([[10, 10, 30, 30], [35, 30, 65, 70], [80, 80, 90, 90]], dtype=float)
    source = source_face_matching(np.array([35, 30, 65, 70], dtype=float), boxes)
    if ours != source or ours != 1:
        raise RuntimeError("DSCT target-query matching differs from fixed source")
    payload = {
        "schema_version": 1,
        "method": "DSCT-FT",
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT,
                     "verified_file_sha256": hashes, "registered_file_sha256": UPSTREAM_FILES,
                     "source_copied_into_repository": False},
        "operator_equivalence": {"face_matching_exact_index": True, "selected_query": ours},
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
