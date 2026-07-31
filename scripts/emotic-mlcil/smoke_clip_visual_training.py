#!/usr/bin/env python
"""One-batch GPU memory smoke for the heaviest continual CLIP states."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def peak_mebibytes() -> float:
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CLIP training smoke")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    from clip import clip

    device = torch.device("cuda")
    clip_model, _ = clip.load(args.clip_model_path, device="cpu", jit=False)
    visual = clip_model.visual.float().to(device)
    del clip_model
    head = nn.Linear(int(visual.output_dim), 5).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": visual.parameters(), "lr": 1.0e-5},
            {"params": head.parameters(), "lr": 1.0e-4},
        ],
        weight_decay=1.0e-4,
    )
    scaler = torch.cuda.amp.GradScaler()
    images = torch.rand(args.batch_size, 3, 224, 224, device=device)
    targets = torch.randint(
        0,
        2,
        (args.batch_size, 5),
        device=device,
    ).float()

    teacher = copy.deepcopy(visual).eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        student_features = F.normalize(visual(images).float(), dim=-1)
        logits = head(student_features)
        with torch.no_grad():
            teacher_features = F.normalize(teacher(images).float(), dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + F.mse_loss(student_features, teacher_features)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    lwf_peak = peak_mebibytes()
    del teacher, teacher_features, student_features, logits, loss
    torch.cuda.empty_cache()

    means = {
        name: parameter.detach().clone()
        for name, parameter in visual.named_parameters()
    }
    fisher = {
        name: torch.full_like(parameter, 1.0e-8)
        for name, parameter in visual.named_parameters()
    }
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        features = F.normalize(visual(images).float(), dim=-1)
        logits = head(features)
        classification = F.binary_cross_entropy_with_logits(logits, targets)
        penalty = torch.zeros((), device=device)
        for name, parameter in visual.named_parameters():
            penalty = penalty + (
                fisher[name] * (parameter - means[name]).pow(2)
            ).sum()
        loss = classification + 50.0 * penalty
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    ewc_peak = peak_mebibytes()

    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "lwf_peak_MiB": round(lwf_peak, 1),
                "ewc_peak_MiB": round(ewc_peak, 1),
                "device": torch.cuda.get_device_name(0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
