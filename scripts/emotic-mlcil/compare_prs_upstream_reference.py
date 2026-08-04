#!/usr/bin/env python3
"""Verify the independent PRS policy against immutable upstream source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.replay_memory import (
    PartitionedReservoirReplayBuffer,
    ReplayRecord,
)


COMMIT = "136cee1863af03cc914dc05dfd41bda8b7bc0bf2"
TREE = "8e08f4ea1ffcce30c73f9453d6db522d7cba7c77"
FILE_SHA = {
    "LICENSE": "116db92b1b611a171ebe1a30c2290b91b42936a7cc6ce33b43b666b5fd07755a",
    "code/configs/mlab_prs-coco.yaml": "23150ec815627f96725c947be30e56a9279ed24156e646942404da2dd9d0074a",
    "code/models/reservoir/mlab_stratified_reservoir.py": "d1e4c97e7c21b28e6ce63a441b358185db40a148a95feddfe155771bd942adfb",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_class(source_root: Path):
    tensorboard = types.ModuleType("tensorboardX")
    tensorboard.SummaryWriter = object
    sys.modules.setdefault("tensorboardX", tensorboard)

    class _Styled:
        def __init__(self, value):
            self.styled_string = str(value)

    colorful = types.ModuleType("colorful")
    colorful.bold_cyan = _Styled
    colorful.bold_red = _Styled
    sys.modules.setdefault("colorful", colorful)

    package = types.ModuleType("prs_fixed_source")
    package.__path__ = []
    sys.modules[package.__name__] = package
    root = source_root / "code/models/reservoir"
    _load_module("prs_fixed_source.base", root / "base.py")
    module = _load_module(
        "prs_fixed_source.mlab_stratified_reservoir",
        root / "mlab_stratified_reservoir.py",
    )
    return module.PRS_mlab


def _record(sample_id: str, labels, num_classes: int) -> ReplayRecord:
    targets = torch.zeros(num_classes)
    targets[list(labels)] = 1.0
    return ReplayRecord(
        image=torch.tensor([float(len(sample_id))]),
        sample_id=sample_id,
        targets=targets,
        visible_mask=torch.ones(num_classes, dtype=torch.bool),
    )


def compare(source_root: Path):
    observed = {}
    for relative, expected in FILE_SHA.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = _sha(path)
        if observed[relative] != expected:
            raise ValueError(f"Fixed PRS source differs: {relative}")

    num_classes = 4
    capacity = 4
    seed = 19
    allocation_power = -0.03
    stream = [
        ("sample-a", (0,)),
        ("sample-b", (0,)),
        ("sample-c", (1,)),
        ("sample-d", (0, 1)),
        ("sample-e", (2,)),
        ("sample-f", (3,)),
        ("sample-g", (0,)),
        ("sample-h", (2, 3)),
        ("sample-i", (1, 2)),
        ("sample-j", (3,)),
        ("sample-k", (0, 2)),
        ("sample-l", (1, 3)),
    ]

    upstream_class = _load_upstream_class(source_root)
    upstream = upstream_class(
        {
            "device": "cpu",
            "reservoir_size": capacity,
            "q_poa": allocation_power,
            "nb_classes": num_classes,
            "model_name": "mlab_reservoir",
        }
    )
    port = PartitionedReservoirReplayBuffer(
        num_classes=num_classes,
        seed=seed,
        allocation_power=allocation_power,
    )
    port.set_capacity(capacity)
    random.seed(seed)
    for sample_id, labels in stream:
        item = _record(sample_id, labels, num_classes)
        upstream.update(
            imgs=item.image.unsqueeze(0),
            cats=item.targets.long().unsqueeze(0),
            sample_ids=[sample_id],
        )
        port.observe(item)

    upstream_ids = list(upstream.rsvr["sample_ids"][: len(upstream)])
    port_ids = [item.sample_id for item in port.records]
    upstream_counts = [
        float(upstream.substreams._data[index].n)
        if index in upstream.substreams._data
        else 0.0
        for index in range(num_classes)
    ]
    upstream_proportions = [
        float(upstream.substreams._data[index].proportion)
        if index in upstream.substreams._data
        else 0.0
        for index in range(num_classes)
    ]
    errors = {
        "retained_id_exact_match": upstream_ids == port_ids,
        "observed_count_max_abs_error": max(
            abs(left - right)
            for left, right in zip(upstream_counts, port.observed_positive_counts)
        ),
        "target_proportion_max_abs_error": max(
            abs(left - right)
            for left, right in zip(upstream_proportions, port.target_proportions)
        ),
    }
    if not errors["retained_id_exact_match"] or max(
        errors["observed_count_max_abs_error"],
        errors["target_proportion_max_abs_error"],
    ) > 1.0e-6:
        raise ValueError(
            "PRS operator equivalence failed: "
            f"{errors}; upstream={upstream_ids}; port={port_ids}"
        )
    return {
        "schema_version": 1,
        "method": "PRS",
        "upstream": {
            "repository": "https://github.com/cdjkim/PRS",
            "commit": COMMIT,
            "tree": TREE,
            "license": "MIT",
            "verified_file_sha256": observed,
            "source_copied_into_repository": False,
        },
        "operator_equivalence": errors,
        "track_a_mapping": {
            "source_capacity": 2000,
            "source_replay_to_current_ratio": 1.0,
            "source_allocation_power": -0.03,
            "registered_capacity": "20 * seen classes",
            "registered_update_timing": "one pass at task end",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.upstream_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
