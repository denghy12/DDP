#!/usr/bin/env python3
"""Worst-task MULTI-LANE train/eval GPU memory smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.multi_lane import MultiLaneBenchmarkMethod
from benchmarks.emotic_mlcil.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--protocol",
        default="configs/emotic_mlcil/protocol_b5c3.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("MULTI-LANE GPU smoke requires CUDA")
    protocol = load_protocol(args.protocol)
    method = MultiLaneBenchmarkMethod(
        protocol,
        clip_model_path=args.clip_model_path,
        device="cuda",
    )
    model = method.model
    for task_id in range(protocol.num_tasks):
        model.activate_task(task_id)
    final_task = protocol.num_tasks - 1
    current_indices = list(protocol.current_class_indices(final_task))
    current_classes = len(current_indices)
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
    model.train()
    with method._autocast():
        logits = model.current_all_logits(images)
        full_targets = torch.zeros_like(logits, dtype=torch.float32)
        full_targets[:, current_indices] = targets
        hidden_indices = [
            index
            for index in range(protocol.num_classes)
            if index not in current_indices
        ]
        masked_logits = logits.index_fill(
            1,
            torch.tensor(hidden_indices, device="cuda"),
            0.0,
        )
        loss = F.binary_cross_entropy_with_logits(
            masked_logits.float() / method.options.temperature,
            full_targets,
        )
    scale_before_step = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = float(scaler.get_scale()) >= scale_before_step
    if not optimizer_stepped:
        raise RuntimeError("MULTI-LANE smoke did not apply its Adam update")
    optimizer_state_tensors = sum(
        int(torch.is_tensor(value))
        for state in optimizer.state.values()
        for value in state.values()
    )
    if optimizer_state_tensors == 0:
        raise RuntimeError("MULTI-LANE smoke did not initialize Adam state")
    model.assert_visual_frozen()
    if any(
        parameter.grad is not None
        for parameter in model.visual_encoder.parameters()
    ):
        raise RuntimeError("Frozen visual parameters unexpectedly received gradients")

    del images, logits, masked_logits, full_targets, targets
    model.eval()
    with torch.no_grad(), method._autocast():
        eval_images = torch.rand(
            args.eval_batch_size, 3, 224, 224, device="cuda"
        )
        seen_logits = model.seen_logits(eval_images)
    if not torch.isfinite(seen_logits).all():
        raise RuntimeError("MULTI-LANE concat evaluation produced non-finite logits")
    torch.cuda.synchronize()
    print(
        {
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "tasks": protocol.num_tasks,
            "classes": protocol.num_classes,
            "current_classes": current_classes,
            "loss": float(loss.detach()),
            "eval_shape": tuple(seen_logits.shape),
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
