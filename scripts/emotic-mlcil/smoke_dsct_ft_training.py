#!/usr/bin/env python3
"""One-step worst-case DSCT-FT CUDA memory smoke."""

import argparse
import json
import time

import torch

from benchmarks.emotic_mlcil.methods.dsct_ft.method import DSCTFTOptions, build_upstream_model
from benchmarks.emotic_mlcil.methods.dsct_ft.model import dsct_current_task_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1333)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--benchmark-steps", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    options = DSCTFTOptions()
    model, provenance = build_upstream_model(args.source_root, args.pretrained_weights, options, device)
    gpu_count = min(args.batch_size, torch.cuda.device_count())
    if gpu_count < 1:
        raise RuntimeError("No visible CUDA device")
    if args.batch_size != 4 or gpu_count not in (1, 4):
        raise ValueError("DSCT smoke requires global batch 4 on one or four visible GPUs")
    model.add_head(5); model.to(device).train()
    parallel = (
        torch.nn.DataParallel(model, device_ids=list(range(gpu_count)), output_device=0)
        if gpu_count > 1 else model
    )
    images = torch.zeros(args.batch_size, 5, args.height, args.width, device=device)
    images[:, :3].normal_()
    images[:, 3, args.height // 4:3 * args.height // 4,
           args.width // 4:3 * args.width // 4] = 1
    images[:, 4] = 1
    targets = torch.zeros(args.batch_size, 5, device=device); targets[:, 0] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=options.amp)
    for index in range(gpu_count):
        torch.cuda.reset_peak_memory_stats(index)

    def loss_fp32(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.float()
        if isinstance(value, dict):
            return {key: loss_fp32(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(loss_fp32(item) for item in value)
        return value

    def training_step():
        optimizer.zero_grad(set_to_none=True)
        if gpu_count > 1:
            with torch.cuda.amp.autocast(enabled=options.amp):
                output = parallel(images)
            output = loss_fp32(output)
            loss, _ = dsct_current_task_loss(output, targets, 5)
            with torch.cuda.amp.autocast(enabled=False):
                for auxiliary in output.get("aux_outputs", []):
                    auxiliary = dict(auxiliary); auxiliary["target_boxes"] = output["target_boxes"]
                    value, _ = dsct_current_task_loss(auxiliary, targets, 5); loss = loss + value
            scaler.scale(loss).backward()
        else:
            loss = images.new_zeros(())
            for index in range(args.batch_size):
                with torch.cuda.amp.autocast(enabled=options.amp):
                    output = model(images[index:index + 1])
                output = loss_fp32(output)
                with torch.cuda.amp.autocast(enabled=False):
                    micro_loss, _ = dsct_current_task_loss(output, targets[index:index + 1], 5)
                    for auxiliary in output.get("aux_outputs", []):
                        auxiliary = dict(auxiliary); auxiliary["target_boxes"] = output["target_boxes"]
                        value, _ = dsct_current_task_loss(
                            auxiliary, targets[index:index + 1], 5
                        ); micro_loss = micro_loss + value
                scaler.scale(micro_loss / args.batch_size).backward()
                loss = loss + micro_loss.detach() / args.batch_size
                del output, micro_loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        return loss

    durations = []
    loss = None
    for step in range(args.warmup_steps + args.benchmark_steps):
        torch.cuda.synchronize()
        started = time.monotonic()
        loss = training_step()
        torch.cuda.synchronize()
        if step >= args.warmup_steps:
            durations.append(time.monotonic() - started)
    peak_allocated = [torch.cuda.max_memory_allocated(index) / 2**20 for index in range(gpu_count)]
    peak_reserved = [torch.cuda.max_memory_reserved(index) / 2**20 for index in range(gpu_count)]
    print(json.dumps({"global_batch_size": args.batch_size,
                      "per_gpu_micro_batch_size": 1,
                      "visible_gpu_count": gpu_count,
                      "height": args.height, "width": args.width,
                      "amp": options.amp, "tf32": options.tf32,
                      "amp_bridge_operator_count": model.amp_bridge_operator_count,
                      "loss": float(loss.detach().cpu()),
                      "benchmark_steps": args.benchmark_steps,
                      "mean_optimizer_step_seconds": sum(durations) / len(durations),
                      "peak_allocated_mib_by_logical_gpu": peak_allocated,
                      "peak_reserved_mib_by_logical_gpu": peak_reserved,
                      "pretrained_weights_sha256": provenance["pretrained_weights_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
