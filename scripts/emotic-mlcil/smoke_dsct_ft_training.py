#!/usr/bin/env python3
"""One-step worst-case DSCT-FT CUDA memory smoke."""

import argparse
import json

import torch

from benchmarks.emotic_mlcil.methods.dsct_ft.method import DSCTFTOptions, build_upstream_model
from benchmarks.emotic_mlcil.methods.dsct_ft.model import dsct_current_task_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    options = DSCTFTOptions()
    model, provenance = build_upstream_model(args.source_root, args.pretrained_weights, options, device)
    model.add_head(5); model.to(device).train()
    images = torch.zeros(args.batch_size, 5, 480, 640, device=device)
    images[:, :3].normal_(); images[:, 3, 100:380, 180:460] = 1; images[:, 4] = 1
    targets = torch.zeros(args.batch_size, 5, device=device); targets[:, 0] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss, _ = dsct_current_task_loss(output, targets, 5)
    for auxiliary in output.get("aux_outputs", []):
        auxiliary = dict(auxiliary); auxiliary["target_boxes"] = output["target_boxes"]
        value, _ = dsct_current_task_loss(auxiliary, targets, 5); loss = loss + value
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip_norm); optimizer.step()
    print(json.dumps({"batch_size": args.batch_size, "loss": float(loss.detach().cpu()),
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                      "pretrained_weights_sha256": provenance["pretrained_weights_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
