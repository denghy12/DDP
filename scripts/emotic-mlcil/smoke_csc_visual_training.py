#!/usr/bin/env python3
"""Worst-task CLIP/CSC teacher + Adam-step GPU memory smoke."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from benchmarks.emotic_mlcil.methods.csc import CSCBenchmarkMethod
from benchmarks.emotic_mlcil.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--protocol",
        default="configs/emotic_mlcil/protocol_b5c3.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CSC GPU smoke requires CUDA")
    protocol = load_protocol(args.protocol)
    method = CSCBenchmarkMethod(
        protocol,
        clip_model_path=args.clip_model_path,
        device="cuda",
    )
    model = method.model
    for task_id in range(protocol.num_tasks - 1):
        model.add_task(len(protocol.current_class_indices(task_id)))
    teacher = copy.deepcopy(model).cuda().eval()
    teacher.requires_grad_(False)
    final_task = protocol.num_tasks - 1
    current_classes = len(protocol.current_class_indices(final_task))
    old_classes = model.num_classes
    model.add_task(current_classes)
    model.cuda().train()

    optimizer = method._optimizer()
    scaler = torch.cuda.amp.GradScaler(
        enabled=method._amp_enabled,
        init_scale=1.0,
    )
    images = torch.rand(args.batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(
        0,
        2,
        (args.batch_size, current_classes),
        device="cuda",
    ).float()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), method._autocast():
        teacher_targets = torch.sigmoid(teacher(images)["logits"].float())
    with method._autocast():
        logits = model(images)["logits"].float()
        current_loss = F.binary_cross_entropy_with_logits(
            logits[:, old_classes:], targets
        )
        distillation_loss = F.binary_cross_entropy_with_logits(
            logits[:, :old_classes], teacher_targets
        )
        entropy_loss = method._entropy_regularizer(
            logits, method.options.entropy_strength
        )
        loss = (
            method.options.alpha * current_loss
            + (1.0 - method.options.alpha) * distillation_loss
            + entropy_loss
        )
    scale_before_step = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = float(scaler.get_scale()) >= scale_before_step
    if not optimizer_stepped:
        raise RuntimeError("CSC memory smoke did not apply its Adam update")
    optimizer_state_tensors = sum(
        int(torch.is_tensor(value))
        for state in optimizer.state.values()
        for value in state.values()
    )
    if optimizer_state_tensors == 0:
        raise RuntimeError("CSC memory smoke did not initialize Adam state")
    torch.cuda.synchronize()
    print(
        {
            "batch_size": args.batch_size,
            "tasks": protocol.num_tasks,
            "classes": protocol.num_classes,
            "old_classes": old_classes,
            "current_classes": current_classes,
            "current_loss": float(current_loss.detach()),
            "distillation_loss": float(distillation_loss.detach()),
            "entropy_loss": float(entropy_loss.detach()),
            "optimizer": "Adam",
            "optimizer_stepped": optimizer_stepped,
            "optimizer_state_tensors": optimizer_state_tensors,
            "amp": method._amp_enabled,
            "peak_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }
    )


if __name__ == "__main__":
    main()
