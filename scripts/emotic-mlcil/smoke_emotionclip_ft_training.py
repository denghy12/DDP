#!/usr/bin/env python3
"""One-step CUDA memory smoke for EmotionCLIP-FT."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.emotionclip_ft import (
    EmotionCLIPFTModel,
    EmotionCLIPVisualTransformer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the EmotionCLIP memory smoke")
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = EmotionCLIPFTModel(EmotionCLIPVisualTransformer())
    state = {}
    for raw_name, value in payload["model"].items():
        name = str(raw_name).removeprefix("module.")
        if name.startswith("backbone.visual."):
            state[name[len("backbone.visual."):]] = value
    model.load_official_visual_state(state)
    model.add_head(5)
    model.cuda().train()
    optimizer = torch.optim.AdamW(
        (
            {"params": model.visual_encoder.parameters(), "lr": 1.0e-5},
            {"params": model.heads.parameters(), "lr": 1.0e-4},
        ),
        betas=(0.98, 0.9), eps=1.0e-6, weight_decay=1.0e-4,
    )
    inputs = torch.randn(args.batch_size, 4, 224, 224, device="cuda")
    inputs[:, 3].sigmoid_().round_()
    targets = torch.randint(0, 2, (args.batch_size, 5), device="cuda").float()
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss = F.binary_cross_entropy_with_logits(model.current_logits(inputs), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    print(json.dumps({
        "batch_size": args.batch_size,
        "loss": float(loss.detach().cpu()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "optimizer": "AdamW",
        "official_emotionclip_initialization": True,
        "subject_mask_used": True,
    }, indent=2))


if __name__ == "__main__":
    main()
