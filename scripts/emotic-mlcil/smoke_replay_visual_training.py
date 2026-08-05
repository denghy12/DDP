#!/usr/bin/env python3
"""Worst-step CLIP visual + one/two equal-size replay GPU memory smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-draws", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for replay memory smoke")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    from clip import clip

    device = torch.device("cuda")
    clip_model, _ = clip.load(args.clip_model_path, device="cpu", jit=False)
    visual = clip_model.visual.float().to(device)
    del clip_model
    head = nn.Linear(int(visual.output_dim), 26).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": visual.parameters(), "lr": 1.0e-5},
            {"params": head.parameters(), "lr": 1.0e-4},
        ],
        weight_decay=1.0e-4,
    )
    scaler = torch.cuda.amp.GradScaler()
    current_images = torch.rand(args.batch_size, 3, 224, 224, device=device)
    replay_images = torch.rand(args.batch_size, 3, 224, 224, device=device)
    dark_images = (
        torch.rand(args.batch_size, 3, 224, 224, device=device)
        if args.replay_draws == 2
        else None
    )
    current_targets = torch.randint(
        0, 2, (args.batch_size, 3), device=device
    ).float()
    replay_targets = torch.randint(
        0, 2, (args.batch_size, 26), device=device
    ).float()
    replay_mask = torch.ones_like(replay_targets)
    stored_logits = torch.randn_like(replay_targets)

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        current_features = F.normalize(visual(current_images).float(), dim=-1)
        replay_features = F.normalize(visual(replay_images).float(), dim=-1)
        current_losses = F.binary_cross_entropy_with_logits(
            head(current_features)[:, -3:],
            current_targets,
            reduction="none",
        ).mean(dim=1)
        replay_raw = F.binary_cross_entropy_with_logits(
            head(replay_features),
            replay_targets,
            reduction="none",
        )
        replay_losses = (
            replay_raw * replay_mask
        ).sum(dim=1) / replay_mask.sum(dim=1)
        if dark_images is None:
            loss = torch.cat([current_losses, replay_losses]).mean()
        else:
            dark_features = F.normalize(visual(dark_images).float(), dim=-1)
            dark_loss = F.mse_loss(head(dark_features), stored_logits)
            loss = current_losses.mean() + 0.5 * dark_loss + 0.5 * replay_losses.mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(visual.parameters()) + list(head.parameters()), 1.0
    )
    scaler.step(optimizer)
    scaler.update()
    peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    print(
        json.dumps(
            {
                "current_batch_size": args.batch_size,
                "replay_batch_size": args.batch_size,
                "independent_replay_draws": args.replay_draws,
                "seen_classes": 26,
                "optimizer": "AdamW",
                "peak_mib": round(peak, 1),
                "device": torch.cuda.get_device_name(0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
