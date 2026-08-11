#!/usr/bin/env python3
"""One-step native BENet-FT memory smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.benet_ft.method import focal_tag_loss
from benchmarks.emotic_mlcil.methods.benet_ft.model import BENetFTModel, load_official_benet_core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BENet memory smoke")
    torch.cuda.reset_peak_memory_stats()
    core, block, provenance = load_official_benet_core(Path(args.source_root), Path(args.pretrained_weights))
    model = BENetFTModel(core, block, provenance).cuda().train()
    model.add_head(5)
    images = torch.randn(args.batch_size, 11, 512, 512, device="cuda")
    images[:, 9].zero_()
    images[:, 9, 128:384, 160:352] = 1
    images[:, 10].fill_(1)
    targets = torch.zeros(args.batch_size, 5, device="cuda")
    targets[:, 0] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
        loss = focal_tag_loss(model.current_logits(images, "bu"), targets)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(json.dumps({
        "batch_size": args.batch_size,
        "loss": float(loss.detach()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "optimizer_stepped": True,
    }, indent=2))


if __name__ == "__main__":
    main()
