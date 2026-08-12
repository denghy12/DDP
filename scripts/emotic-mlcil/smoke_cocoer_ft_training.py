#!/usr/bin/env python3
"""Worst-task full-graph CUDA memory smoke for native CocoER-FT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet50-init", required=True, type=Path)
    parser.add_argument("--clip-rn50", required=True, type=Path)
    parser.add_argument("--head-cache", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--protocol",
        default="configs/emotic_mlcil/protocol_b5c3_track_b.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    import torch

    from benchmarks.emotic_mlcil.methods.cocoer_ft import (
        CocoERFTBenchmarkMethod,
        dynamic_bce,
    )
    from benchmarks.emotic_mlcil.protocol import load_protocol

    if not torch.cuda.is_available():
        raise RuntimeError("CocoER-FT GPU smoke requires CUDA")
    for asset in (args.resnet50_init, args.clip_rn50, args.head_cache):
        if not asset.is_file():
            raise FileNotFoundError(asset)

    protocol = load_protocol(args.protocol)
    method = CocoERFTBenchmarkMethod(
        protocol,
        resnet50_initialization_path=args.resnet50_init,
        clip_rn50_path=args.clip_rn50,
        head_box_cache_path=args.head_cache,
        device="cuda",
    )
    for task in protocol.tasks:
        method.model.add_head(len(task))
    method.model.cuda().train()
    method.model.clip_image_encoder.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = method.options.tf32
    torch.backends.cudnn.allow_tf32 = method.options.tf32

    parameters = [
        parameter
        for parameter in method.model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=method.options.learning_rate,
        betas=(method.options.beta1, method.options.beta2),
        weight_decay=method.options.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=method._amp_enabled,
        init_scale=1.0,
    )
    current_classes = len(protocol.current_class_indices(protocol.num_tasks - 1))
    raw_height, raw_width = 480, 640
    raw_images = [
        torch.randint(
            0, 256, (3, raw_height, raw_width), dtype=torch.uint8
        ).pin_memory()
        for _ in range(args.batch_size)
    ]
    raw_geometry = torch.tensor(
        [[[80.0, 50.0, 560.0, 460.0], [240.0, 60.0, 400.0, 200.0]]]
    ).expand(args.batch_size, -1, -1).contiguous().pin_memory()
    image_sizes = torch.tensor(
        [[raw_height, raw_width]] * args.batch_size, dtype=torch.int64
    ).pin_memory()
    targets = torch.randint(
        0,
        2,
        (args.batch_size, current_classes),
        device="cuda",
    ).float()
    targets[0].fill_(1.0)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    preprocess_start = torch.cuda.Event(enable_timing=True)
    preprocess_end = torch.cuda.Event(enable_timing=True)
    preprocess_start.record()
    images, geometry = method.gpu_preprocessor(
        raw_images, raw_geometry, image_sizes, train=True
    )
    preprocess_end.record()
    with method._autocast():
        output = method.model(images, geometry)
        global_loss = dynamic_bce(
            output["logits"][:, -current_classes:], targets
        )
        vi_loss = dynamic_bce(
            output["vi_logits"][:, -current_classes:], targets
        )
        pseudo_loss = (
            output["pseudo_head_loss"]
            + output["pseudo_body_loss"]
            + output["pseudo_context_loss"]
        )
        loss = 0.2 * (global_loss + vi_loss + pseudo_loss) + (
            method.options.grad_distance_weight
            * output["competition_distance"]
        )
    scale_before_step = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        parameters, method.options.gradient_clip_norm
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = float(scaler.get_scale()) >= scale_before_step
    if not optimizer_stepped:
        raise RuntimeError("CocoER-FT smoke did not apply its AdamW update")
    optimizer_state_tensors = sum(
        int(torch.is_tensor(value))
        for state in optimizer.state.values()
        for value in state.values()
    )
    if optimizer_state_tensors == 0:
        raise RuntimeError("CocoER-FT smoke did not initialize AdamW state")
    if any(
        parameter.grad is not None
        for parameter in method.model.clip_image_encoder.parameters()
    ):
        raise RuntimeError("Frozen CocoER CLIP RN50 received gradients")
    if not all(
        torch.isfinite(value).all()
        for value in (
            loss,
            output["logits"],
            output["vi_logits"],
            output["competition_distance"],
        )
    ):
        raise RuntimeError("CocoER-FT smoke produced non-finite values")

    torch.cuda.synchronize()
    payload = {
        "schema_version": 1,
        "method": "CocoER-FT",
        "protocol_id": protocol.protocol_id,
        "simulated_task_id": protocol.num_tasks - 1,
        "active_head_sizes": [len(task) for task in protocol.tasks],
        "seen_classes": protocol.num_classes,
        "current_classes": current_classes,
        "train_batch_size": args.batch_size,
        "raw_input_shape_per_sample": [3, raw_height, raw_width],
        "model_input_shape": list(images.shape),
        "complete_paths": ["head", "body", "context", "vi", "global"],
        "preprocessing_backend": "cuda_v0.2",
        "preprocessing_milliseconds": preprocess_start.elapsed_time(preprocess_end),
        "preprocessing_samples_per_second": (
            1000.0 * args.batch_size / preprocess_start.elapsed_time(preprocess_end)
        ),
        "optimizer": "AdamW",
        "optimizer_stepped": optimizer_stepped,
        "optimizer_state_tensors": optimizer_state_tensors,
        "amp": method._amp_enabled,
        "tf32": method.options.tf32,
        "clip_rn50_frozen": True,
        "loss": float(loss.detach().cpu()),
        "global_loss": float(global_loss.detach().cpu()),
        "vi_loss": float(vi_loss.detach().cpu()),
        "pseudo_loss": float(pseudo_loss.detach().cpu()),
        "competition_distance": float(
            output["competition_distance"].detach().cpu()
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in method.model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in method.model.parameters()
            if parameter.requires_grad
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
