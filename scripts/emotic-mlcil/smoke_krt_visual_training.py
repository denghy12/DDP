#!/usr/bin/env python3
"""Worst-task CLIP/KRT optimizer-step GPU memory smoke."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from benchmarks.emotic_mlcil.methods.krt import KRTBenchmarkMethod
from benchmarks.emotic_mlcil.methods.krt.method import AsymmetricLoss
from benchmarks.emotic_mlcil.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
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
        raise RuntimeError("KRT GPU smoke requires CUDA")
    protocol = load_protocol(args.protocol)
    method = KRTBenchmarkMethod(
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
    model.add_task(len(protocol.current_class_indices(final_task)))
    model.cuda().train()
    model.freeze_old_task_parameters()

    optimizer = method._optimizer(method.options.incremental_learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1.0)
    smoke_batch_size = max(args.batch_size, method.options.replay_batch_size)
    images = torch.rand(smoke_batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(
        0,
        2,
        (smoke_batch_size, protocol.num_classes),
        device="cuda",
    ).float()
    criterion = AsymmetricLoss()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.cuda.amp.autocast():
        old_output = teacher(images)
    with torch.cuda.amp.autocast():
        output = model(images)
        classification = criterion(output["logits"].float(), targets)
        current_old_tokens = torch.cat(output["tokens"][:-1], dim=1)
        teacher_tokens = torch.cat(old_output["tokens"], dim=1)
        token_loss = (
            method.options.token_distillation_weight
            * math.sqrt(
                protocol.num_classes
                / len(protocol.current_class_indices(final_task))
            )
            * F.cosine_embedding_loss(
                current_old_tokens,
                teacher_tokens,
                torch.ones(smoke_batch_size, device="cuda"),
            )
        )
        loss = classification + token_loss
    scale_before_step = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = float(scaler.get_scale()) >= scale_before_step
    if not optimizer_stepped:
        raise RuntimeError("KRT memory smoke did not apply its Adam update")
    optimizer_state_tensors = sum(
        int(torch.is_tensor(value))
        for state in optimizer.state.values()
        for value in state.values()
    )
    if optimizer_state_tensors == 0:
        raise RuntimeError("KRT memory smoke did not initialize Adam state")
    torch.cuda.synchronize()
    print(
        {
            "requested_train_batch_size": args.batch_size,
            "replay_batch_size": method.options.replay_batch_size,
            "smoke_batch_size": smoke_batch_size,
            "tasks": protocol.num_tasks,
            "classes": protocol.num_classes,
            "classification_loss": float(classification.detach()),
            "token_loss": float(token_loss.detach()),
            "optimizer": "Adam",
            "optimizer_stepped": optimizer_stepped,
            "optimizer_state_tensors": optimizer_state_tensors,
            "peak_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }
    )


if __name__ == "__main__":
    main()
