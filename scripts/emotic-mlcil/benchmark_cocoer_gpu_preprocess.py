#!/usr/bin/env python3
"""Compare registered CPU v0.1 and CUDA v0.2 CocoER preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--head-cache", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/emotic_mlcil/protocol_b5c3_track_b.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--equivalence-samples", type=int, default=8)
    parser.add_argument("--geometry-atol", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.batches, args.equivalence_samples) <= 0:
        raise ValueError("batch/batches/equivalence-samples must be positive")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch

    from benchmarks.emotic_mlcil.data_module import EMOTICMLCILDataModule
    from benchmarks.emotic_mlcil.methods.cocoer_ft import CocoERGPUPreprocessor
    from benchmarks.emotic_mlcil.protocol import load_protocol
    from benchmarks.emotic_mlcil.runner import _cocoer_transforms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = load_protocol(args.protocol)
    gpu_train, gpu_eval = _cocoer_transforms(args.head_cache)
    cpu_train, cpu_eval = _cocoer_transforms(
        args.head_cache, gpu_preprocessing=False
    )
    gpu_data = EMOTICMLCILDataModule(
        protocol, args.data_root, gpu_train, gpu_eval, "cocoer_multilevel"
    )
    cpu_data = EMOTICMLCILDataModule(
        protocol, args.data_root, cpu_train, cpu_eval, "cocoer_multilevel"
    )
    gpu_loader = gpu_data.method_loader(
        protocol.num_tasks - 1,
        args.batch_size,
        args.workers,
        split=protocol.validation_split,
        shuffle=False,
    )
    cpu_loader = cpu_data.method_loader(
        protocol.num_tasks - 1,
        args.batch_size,
        args.workers,
        split=protocol.validation_split,
        shuffle=False,
    )
    preprocessor = CocoERGPUPreprocessor(torch.device("cuda"), protocol.seed)

    def timed_cpu():
        total = 0
        start = time.perf_counter()
        for index, batch in enumerate(cpu_loader):
            batch.images = batch.images.cuda(non_blocking=True)
            batch.geometry = batch.geometry.cuda(non_blocking=True)
            total += batch.images.shape[0]
            if index + 1 >= args.batches:
                break
        torch.cuda.synchronize()
        return total, time.perf_counter() - start

    def timed_gpu(train: bool):
        total = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for index, batch in enumerate(gpu_loader):
            images, geometry = preprocessor(
                batch.images,
                batch.geometry,
                batch.image_sizes,
                train=train,
            )
            if not torch.isfinite(images).all() or not torch.isfinite(geometry).all():
                raise RuntimeError("GPU preprocessing returned non-finite data")
            total += images.shape[0]
            if index + 1 >= args.batches:
                break
        torch.cuda.synchronize()
        return (
            total,
            time.perf_counter() - start,
            torch.cuda.max_memory_allocated() / 2**20,
            torch.cuda.max_memory_reserved() / 2**20,
        )

    # Warm both worker pools and CUDA kernels before measurement.
    next(iter(cpu_loader))
    warm = next(iter(gpu_loader))
    preprocessor(warm.images, warm.geometry, warm.image_sizes, train=False)
    torch.cuda.synchronize()
    cpu_samples, cpu_seconds = timed_cpu()
    eval_samples, eval_seconds, eval_allocated, eval_reserved = timed_gpu(False)
    train_samples, train_seconds, train_allocated, train_reserved = timed_gpu(True)

    cpu_iterator = iter(cpu_loader)
    gpu_iterator = iter(gpu_loader)
    compared = 0
    tensor_max, tensor_sum, tensor_count, geometry_max = 0.0, 0.0, 0, 0.0
    sample_ids_equal = True
    while compared < args.equivalence_samples:
        cpu_batch = next(cpu_iterator)
        gpu_batch = next(gpu_iterator)
        sample_ids_equal &= cpu_batch.sample_ids == gpu_batch.sample_ids
        gpu_images, gpu_geometry = preprocessor(
            gpu_batch.images,
            gpu_batch.geometry,
            gpu_batch.image_sizes,
            train=False,
        )
        take = min(args.equivalence_samples - compared, cpu_batch.images.shape[0])
        delta = (
            cpu_batch.images[:take].cuda(non_blocking=True) - gpu_images[:take]
        ).abs()
        geometry_delta = (
            cpu_batch.geometry[:take].cuda(non_blocking=True)
            - gpu_geometry[:take]
        ).abs()
        tensor_max = max(tensor_max, float(delta.max().cpu()))
        tensor_sum += float(delta.sum().cpu())
        tensor_count += delta.numel()
        geometry_max = max(geometry_max, float(geometry_delta.max().cpu()))
        compared += take

    payload = {
        "schema_version": 1,
        "method": "CocoER-FT",
        "comparison": "cpu_pil_v0.1_vs_cuda_v0.2",
        "split": "val",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "measured_batches": args.batches,
        "throughput": {
            "cpu_v0.1_samples": cpu_samples,
            "cpu_v0.1_seconds": cpu_seconds,
            "cpu_v0.1_samples_per_second": cpu_samples / cpu_seconds,
            "cuda_v0.2_eval_samples_per_second": eval_samples / eval_seconds,
            "cuda_v0.2_train_samples_per_second": train_samples / train_seconds,
            "eval_speedup_over_cpu": (
                (eval_samples / eval_seconds) / (cpu_samples / cpu_seconds)
            ),
        },
        "equivalence": {
            "samples": compared,
            "sample_ids_equal": sample_ids_equal,
            "geometry_max_abs_error": geometry_max,
            "normalized_tensor_max_abs_error": tensor_max,
            "normalized_tensor_mean_abs_error": tensor_sum / tensor_count,
            "pixel_exact_expected": False,
            "reason": "PIL and tensor antialiased bilinear implementations differ slightly",
        },
        "cuda_memory_mib": {
            "eval_peak_allocated": eval_allocated,
            "eval_peak_reserved": eval_reserved,
            "train_peak_allocated": train_allocated,
            "train_peak_reserved": train_reserved,
        },
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not sample_ids_equal:
        raise RuntimeError("CPU/GPU preprocessing sample IDs differ")
    if geometry_max > args.geometry_atol:
        raise RuntimeError(
            f"CPU/GPU preprocessing geometry differs: {geometry_max}"
        )


if __name__ == "__main__":
    main()
