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
    parser.add_argument("--eval-batch-sizes", default="4,8,16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    options = DSCTFTOptions()
    model, provenance = build_upstream_model(args.source_root, args.pretrained_weights, options, device)
    gpu_count = min(args.batch_size, torch.cuda.device_count())
    if gpu_count < 1:
        raise RuntimeError("No visible CUDA device")
    if args.batch_size != 4 or gpu_count not in (1, 2, 4):
        raise ValueError("DSCT smoke requires global batch 4 on 1, 2, or 4 visible GPUs")
    model.add_head(5); model.to(device)
    if options.channels_last:
        model.to(memory_format=torch.channels_last)
    model.train()
    parallel = (
        torch.nn.DataParallel(model, device_ids=list(range(gpu_count)), output_device=0)
        if gpu_count > 1 else model
    )
    images = torch.zeros(args.batch_size, 5, args.height, args.width, device=device)
    images[:, :3].normal_()
    images[:, 3, args.height // 4:3 * args.height // 4,
           args.width // 4:3 * args.width // 4] = 1
    images[:, 4] = 1
    if options.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
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
        with torch.cuda.amp.autocast(enabled=options.amp):
            output = parallel(images)
        output = loss_fp32(output)
        with torch.cuda.amp.autocast(enabled=False):
            loss, _ = dsct_current_task_loss(output, targets, 5)
            for auxiliary in output.get("aux_outputs", []):
                auxiliary = dict(auxiliary); auxiliary["target_boxes"] = output["target_boxes"]
                value, _ = dsct_current_task_loss(auxiliary, targets, 5); loss = loss + value
        scaler.scale(loss).backward()
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
    training_summary = {"global_batch_size": args.batch_size,
                      "per_gpu_micro_batch_size": args.batch_size // gpu_count,
                      "visible_gpu_count": gpu_count,
                      "height": args.height, "width": args.width,
                      "amp": options.amp, "tf32": options.tf32,
                      "channels_last": options.channels_last,
                      "amp_bridge_operator_count": model.amp_bridge_operator_count,
                      "loss": float(loss.detach().cpu()),
                      "benchmark_steps": args.benchmark_steps,
                      "mean_optimizer_step_seconds": sum(durations) / len(durations),
                      "peak_allocated_mib_by_logical_gpu": peak_allocated,
                      "peak_reserved_mib_by_logical_gpu": peak_reserved,
                      "pretrained_weights_sha256": provenance["pretrained_weights_sha256"]}

    del images, targets, loss
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    model.eval()
    evaluation = []
    for eval_batch_size in [
        int(value) for value in args.eval_batch_sizes.split(",") if value.strip()
    ]:
        row = {"batch_size": eval_batch_size}
        try:
            eval_images = torch.zeros(
                eval_batch_size, 5, args.height, args.width, device=device
            )
            eval_images[:, :3].normal_()
            eval_images[:, 3, args.height // 4:3 * args.height // 4,
                        args.width // 4:3 * args.width // 4] = 1
            eval_images[:, 4] = 1
            if options.channels_last:
                eval_images = eval_images.contiguous(memory_format=torch.channels_last)
            for index in range(gpu_count):
                torch.cuda.reset_peak_memory_stats(index)
            eval_durations = []
            with torch.inference_mode():
                for step in range(args.warmup_steps + args.benchmark_steps):
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    with torch.cuda.amp.autocast(enabled=options.amp):
                        eval_output = parallel(eval_images)
                        eval_scores = model.selected_scores(eval_output)
                    torch.cuda.synchronize()
                    if step >= args.warmup_steps:
                        eval_durations.append(time.monotonic() - started)
            row.update({
                "status": "passed",
                "mean_batch_seconds": sum(eval_durations) / len(eval_durations),
                "samples_per_second": eval_batch_size * len(eval_durations)
                / sum(eval_durations),
                "peak_allocated_mib_by_logical_gpu": [
                    torch.cuda.max_memory_allocated(index) / 2**20
                    for index in range(gpu_count)
                ],
                "peak_reserved_mib_by_logical_gpu": [
                    torch.cuda.max_memory_reserved(index) / 2**20
                    for index in range(gpu_count)
                ],
                "score_shape": list(eval_scores.shape),
            })
            del eval_images, eval_output, eval_scores
        except torch.cuda.OutOfMemoryError as error:
            row.update({"status": "oom", "error": str(error)})
        finally:
            torch.cuda.empty_cache()
        evaluation.append(row)
    training_summary["evaluation_candidates"] = evaluation
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
