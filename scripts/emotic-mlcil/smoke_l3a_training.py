#!/usr/bin/env python3
"""Exercise L3A's Task-0 optimizer and full-width analytic solve on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.l3a import AsymmetricLoss, L3ABenchmarkMethod
from benchmarks.emotic_mlcil.protocol import load_protocol
from benchmarks.emotic_mlcil.types import TaskContext


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--analytic-batch-size", type=int, default=64)
    parser.add_argument(
        "--protocol", default="configs/emotic_mlcil/protocol_b5c3.yaml"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.analytic_batch_size <= 0:
        raise ValueError("L3A smoke batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("L3A GPU smoke requires CUDA")
    protocol = load_protocol(args.protocol)
    method = L3ABenchmarkMethod(
        protocol,
        clip_model_path=args.clip_model_path,
        device="cuda",
    )
    method.begin_task(_context(protocol, 0))
    optimizer = torch.optim.Adam(
        method._optimizer_groups(method.model, method.options.weight_decay),
        lr=method.options.base_learning_rate,
        weight_decay=0.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=method._amp_enabled, init_scale=1.0)
    images = torch.rand(args.batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(
        0,
        2,
        (args.batch_size, len(protocol.current_class_indices(0))),
        device="cuda",
    ).float()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with method._autocast():
        loss = AsymmetricLoss()(method.model.base_logits(images).float(), targets)
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = float(scaler.get_scale()) >= scale_before
    if not optimizer_stepped or not torch.isfinite(loss):
        raise RuntimeError("L3A Task-0 optimizer smoke did not finish cleanly")
    optimizer_state_tensors = sum(
        int(torch.is_tensor(value))
        for state in optimizer.state.values()
        for value in state.values()
    )
    if optimizer_state_tensors == 0:
        raise RuntimeError("L3A Task-0 Adam state was not initialized")
    torch.cuda.synchronize()
    gradient_peak = torch.cuda.max_memory_allocated() / (1024**2)
    gradient_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    loss_value = float(loss.detach())

    del optimizer, scaler, images, targets, loss
    method.model.zero_grad(set_to_none=True)
    method.model.initialize_analytic()
    method.model.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    hidden = method.options.hidden_dim
    classes = protocol.num_classes
    analytic_a = torch.eye(hidden, device="cuda", dtype=torch.float64)
    analytic_c = torch.zeros(hidden, classes, device="cuda", dtype=torch.float64)
    analytic_images = torch.rand(
        args.analytic_batch_size, 3, 224, 224, device="cuda"
    )
    with torch.no_grad():
        features = method.model.analytic_features(analytic_images).double()
        analytic_a.add_(features.t() @ features)
        analytic_c.add_(
            features.t()
            @ torch.rand(args.analytic_batch_size, classes, device="cuda").double()
        )
        solution = torch.linalg.inv(
            analytic_a
            + method.options.ridge
            * torch.eye(hidden, device="cuda", dtype=torch.float64)
        ) @ analytic_c
    if not torch.isfinite(solution).all():
        raise RuntimeError("L3A full-width analytic solve produced non-finite values")
    torch.cuda.synchronize()
    analytic_peak = torch.cuda.max_memory_allocated() / (1024**2)
    analytic_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    print(
        json.dumps(
            {
                "task0_train_batch_size": args.batch_size,
                "analytic_batch_size": args.analytic_batch_size,
                "hidden_dim": hidden,
                "final_classes": classes,
                "optimizer": "Adam",
                "optimizer_stepped": optimizer_stepped,
                "optimizer_state_tensors": optimizer_state_tensors,
                "task0_loss": loss_value,
                "task0_peak_mib": gradient_peak,
                "task0_peak_reserved_mib": gradient_reserved,
                "analytic_peak_mib": analytic_peak,
                "analytic_peak_reserved_mib": analytic_reserved,
                "analytic_solution_finite": True,
                "amp": method._amp_enabled,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
