import argparse
import json
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from build_cfg import setup_cfg
from eval_emotic_threshold_sweep import (
    checkpoint_model_args,
    rebuild_text_feature_cache,
)
from models import ddp
from src.helper_functions.detail_report import DetailReport
from src.helper_functions.emotic_loader import EMOTIC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a joint EMOTIC upper-bound checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./datasets/EMOTIC")
    parser.add_argument("--clip-model-path", default="./pretrained/clip/ViT-B-16.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("val", "test"),
        default=("test",),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classnames = checkpoint["classnames"]
    if len(classnames) != 26:
        raise ValueError(f"Expected 26 EMOTIC classes, found {len(classnames)}")

    model_args = checkpoint_model_args(checkpoint, args)
    model = ddp(setup_cfg(model_args), classnames)
    model = model.module if hasattr(model, "module") else model
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    rebuild_text_feature_cache(model, 26)

    transform = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=tuple(args.eval_splits),
        transform=transform,
        input_mode="full",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Evaluation splits={list(args.eval_splits)} "
        f"person_samples={len(dataset)} threshold={args.threshold:.4f}",
        flush=True,
    )

    output_batches = []
    target_batches = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(inputs, cls_id=(0, 26), inference=True)
            output_batches.append(outputs.float().cpu())
            target_batches.append(targets.float().cpu())

    logits = torch.cat(output_batches)
    targets = torch.cat(target_batches)
    scores = torch.softmax(logits, dim=1)[:, 1, :]
    loss = F.binary_cross_entropy(
        scores.clamp(1e-6, 1 - 1e-6), targets, reduction="mean"
    ).item()

    os.makedirs(args.output_dir, exist_ok=True)
    report = DetailReport(
        args.output_dir,
        args.name,
        classnames,
        [list(range(26))],
        dataset.targets,
    )
    overall = report.update(0, scores, targets, args.threshold, loss)
    summary = {
        "eval_splits": list(args.eval_splits),
        "threshold": args.threshold,
        "temperature": 1.0,
        "checkpoint": os.path.abspath(args.checkpoint),
        **overall,
    }
    with open(
        os.path.join(args.output_dir, "evaluation_summary.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    torch.save(
        {"scores": scores, "targets": targets, "logits": logits},
        os.path.join(args.output_dir, "task0_scores.pt"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {report.path}", flush=True)


if __name__ == "__main__":
    main()
