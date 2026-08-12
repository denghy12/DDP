#!/usr/bin/env python3
"""One-step CUDA memory smoke for EMOT-Net+CCIM-FT."""

import argparse
import json
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft import EMOTNetCCIMFTModel
from benchmarks.emotic_mlcil.methods.emot_net_ft.method import weighted_sigmoid_mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-init", required=True, type=Path)
    parser.add_argument("--ccim-dictionary", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=52)
    parser.add_argument("--tower-model-parallel", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the EMOT-Net+CCIM memory smoke")
    native = torch.load(args.native_init, map_location="cpu")
    resource = torch.load(args.ccim_dictionary, map_location="cpu")
    model = EMOTNetCCIMFTModel(resource["dictionary"], resource["prior"])
    model.load_native_initialization(native)
    model.add_head(5)
    model.cuda().train()
    if args.tower_model_parallel:
        if torch.cuda.device_count() < 3:
            raise RuntimeError("Tower model-parallel smoke requires three visible GPUs")
        model.enable_tower_model_parallel("cuda:0", "cuda:1", "cuda:2")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5.0e-4
    )
    images = torch.randn(args.batch_size, 2, 3, 224, 224, device="cuda")
    images[:, 1, :, 128:, :] = 0
    images[:, 1, :, :, 128:] = 0
    targets = torch.randint(0, 2, (args.batch_size, 5), device="cuda").float()
    smoke_devices = range(3) if args.tower_model_parallel else range(1)
    for device_index in smoke_devices:
        torch.cuda.reset_peak_memory_stats(device_index)
    optimizer.zero_grad(set_to_none=True)
    loss = weighted_sigmoid_mse(
        model.current_logits(images), targets, torch.ones(5, device="cuda")
    )
    loss.backward()
    optimizer.step()
    for device_index in smoke_devices:
        torch.cuda.synchronize(device_index)
    peak_allocated = {
        f"cuda:{device_index}": torch.cuda.max_memory_allocated(device_index) / 2**20
        for device_index in smoke_devices
    }
    peak_reserved = {
        f"cuda:{device_index}": torch.cuda.max_memory_reserved(device_index) / 2**20
        for device_index in smoke_devices
    }
    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "loss": float(loss.detach().cpu()),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "peak_allocated_mib_by_device": peak_allocated,
                "peak_reserved_mib_by_device": peak_reserved,
                "optimizer": "SGD",
                "dictionary_size": model.dictionary_size,
                "confounder_dim": model.confounder_dim,
                "ccim_strategy": model.ccim.strategy,
                "native_emot_net": True,
                "clip_used": False,
                "tower_model_parallel": args.tower_model_parallel,
                "tower_devices": list(model.tower_model_parallel_devices),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
