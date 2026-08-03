#!/usr/bin/env python3
"""Task-0 training and final-task inference memory smoke for Original-DDP-Tau2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.original_ddp import OriginalDDPBenchmarkMethod
from benchmarks.emotic_mlcil.protocol import load_protocol
from benchmarks.emotic_mlcil.types import TaskContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--protocol", default="configs/emotic_mlcil/protocol_b5c3.yaml"
    )
    return parser.parse_args()


def _context(protocol, task_id: int) -> TaskContext:
    current = protocol.current_class_indices(task_id)
    seen = protocol.seen_class_indices(task_id)
    future = protocol.future_class_indices(task_id)
    return TaskContext(
        task_id=task_id,
        protocol_id=protocol.protocol_id,
        class_order=protocol.class_order,
        class_order_hash=protocol.class_order_hash,
        protocol_hash=protocol.protocol_hash,
        current_class_indices=current,
        seen_class_indices=seen,
        future_class_indices=future,
        current_class_names=tuple(protocol.class_order[index] for index in current),
        seen_class_names=tuple(protocol.class_order[index] for index in seen),
        future_class_names=tuple(protocol.class_order[index] for index in future),
        seed=protocol.seed,
        track=protocol.track,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size != 8:
        raise ValueError("Registered Original-DDP-Tau2 train batch size is 8")
    if args.eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Original-DDP-Tau2 memory smoke requires CUDA")

    protocol = load_protocol(args.protocol)
    method = OriginalDDPBenchmarkMethod(
        protocol,
        clip_model_path=args.clip_model_path,
        device="cuda",
    )
    method.begin_task(_context(protocol, 0))
    current_classes = len(protocol.current_class_indices(0))
    images = torch.rand(args.batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(
        0, 2, (args.batch_size, current_classes), device="cuda"
    ).float()
    scaler = torch.cuda.amp.GradScaler(enabled=method._amp_enabled, init_scale=1.0)

    method.model.train()
    method.optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    with method._autocast():
        logits = method.model(images, cls_id=(0, current_classes), inference=False)
        loss = method.options.loss_weight * method.criterion(logits, targets)
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.step(method.optimizer)
    scaler.update()
    optimizer_stepped = float(scaler.get_scale()) >= scale_before
    if not optimizer_stepped:
        raise RuntimeError("Original-DDP-Tau2 smoke did not apply its Adam update")
    optimizer_state_tensors = sum(
        int(torch.is_tensor(value))
        for state in method.optimizer.state.values()
        for value in state.values()
    )
    if optimizer_state_tensors == 0:
        raise RuntimeError("Original-DDP-Tau2 smoke did not initialize Adam state")
    train_peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    train_peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024**2)
    if any(
        parameter.grad is not None
        for name, parameter in method.model.named_parameters()
        if id(parameter) not in {id(item) for item in method._optimizer_parameters()}
    ):
        raise RuntimeError("A frozen Original DDP parameter received a gradient")

    del images, targets, logits
    method.optimizer.zero_grad(set_to_none=True)
    method._completed_task_id = protocol.num_tasks - 1
    method._rebuild_text_feature_cache()
    method.model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), method._autocast():
        eval_images = torch.rand(
            args.eval_batch_size, 3, 224, 224, device="cuda"
        )
        final_logits = method.model(
            eval_images,
            cls_id=(0, protocol.num_classes),
            inference=True,
        )
        final_scores = torch.softmax(
            final_logits.float()
            / method._temperature(protocol.num_classes),
            dim=1,
        )[:, 1, :]
    if tuple(final_scores.shape) != (args.eval_batch_size, protocol.num_classes):
        raise RuntimeError("Original-DDP-Tau2 final score shape differs")
    if not torch.isfinite(final_scores).all():
        raise RuntimeError("Original-DDP-Tau2 smoke produced non-finite scores")
    torch.cuda.synchronize()
    print(
        {
            "method": method.method_name,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "task0_classes": current_classes,
            "final_classes": protocol.num_classes,
            "loss": float(loss.detach()),
            "optimizer": "Adam",
            "optimizer_stepped": optimizer_stepped,
            "optimizer_state_tensors": optimizer_state_tensors,
            "adapter_free": True,
            "replay_samples": 0,
            "pcd_task0": method._temperature(current_classes),
            "pcd_task7": method._temperature(protocol.num_classes),
            "pcd_gamma": method.options.temperature_gamma,
            "amp": method._amp_enabled,
            "train_peak_mib": train_peak_mib,
            "train_peak_reserved_mib": train_peak_reserved_mib,
            "eval_peak_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "eval_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }
    )


if __name__ == "__main__":
    main()
