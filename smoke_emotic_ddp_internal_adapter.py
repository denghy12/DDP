import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import (
    dense_labels,
    eval_transform,
    load_frozen_ddp,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify zero-init internal Adapter preserves real DDP logits"
    )
    parser.add_argument("--ddp_checkpoint", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_frozen_ddp(
        args.ddp_checkpoint, args.clip_model_path, device
    )
    dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=eval_transform(),
        input_mode="full",
    )
    labels = dense_labels(dataset)
    indices = torch.nonzero(
        labels[:, :5].sum(dim=1).gt(0), as_tuple=False
    ).flatten()[: args.batch_size]
    images, _ = next(
        iter(DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size))
    )
    images = images.to(device).float()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        baseline = model(images, cls_id=(0, 5), inference=True).float()
        model.enable_feature_adapter(args.adapter_dim, args.residual_scale)
        identity = model(images, cls_id=(0, 5), inference=True).float()
    difference = (identity - baseline).abs()
    result = {
        "samples": int(images.shape[0]),
        "shape": list(baseline.shape),
        "max_abs_error": float(difference.max().cpu()),
        "mean_abs_error": float(difference.mean().cpu()),
        "tolerance": args.tolerance,
        "passed": bool(difference.max().item() <= args.tolerance),
        "ddp_checkpoint": args.ddp_checkpoint,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("Zero-init internal Adapter changed real DDP logits")


if __name__ == "__main__":
    main()
