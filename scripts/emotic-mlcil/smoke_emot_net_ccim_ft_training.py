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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the EMOT-Net+CCIM memory smoke")
    native = torch.load(args.native_init, map_location="cpu")
    resource = torch.load(args.ccim_dictionary, map_location="cpu")
    model = EMOTNetCCIMFTModel(resource["dictionary"], resource["prior"])
    model.load_native_initialization(native)
    model.add_head(5)
    model.cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5.0e-4
    )
    images = torch.randn(args.batch_size, 2, 3, 224, 224, device="cuda")
    images[:, 1, :, 128:, :] = 0
    images[:, 1, :, :, 128:] = 0
    targets = torch.randint(0, 2, (args.batch_size, 5), device="cuda").float()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    loss = weighted_sigmoid_mse(
        model.current_logits(images), targets, torch.ones(5, device="cuda")
    )
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "loss": float(loss.detach().cpu()),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "optimizer": "SGD",
                "dictionary_size": model.dictionary_size,
                "confounder_dim": model.confounder_dim,
                "ccim_strategy": model.ccim.strategy,
                "native_emot_net": True,
                "clip_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
