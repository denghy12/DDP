"""Numerically verify DDP's prompt-free route against official vanilla CLIP."""

import argparse
import json
import os
from pathlib import Path

import torch

from clip import clip
from train_emotic_ddp_internal_adapter import load_frozen_ddp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddp_checkpoint",
        default=(
            "./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth"
        ),
    )
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument(
        "--output",
        default=(
            "./output/emotic_ddp_prompt_free_auxiliary_equivalence/"
            "equivalence.json"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def error_metrics(left, right):
    difference = (left.float() - right.float()).abs()
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


def main():
    args = parse_args()
    if args.tolerance <= 0:
        raise ValueError("tolerance must be positive")
    device = torch.device(args.device)
    ddp_model, _ = load_frozen_ddp(
        args.ddp_checkpoint, args.clip_model_path, device
    )
    clip_path = os.path.abspath(os.path.expanduser(args.clip_model_path))
    vanilla, _ = clip.load(clip_path, device=device, jit=False)
    # DDP explicitly converts its CLIP backbone to float32 when constructed;
    # compare in the same dtype instead of conflating architecture and fp16
    # rounding differences.
    vanilla.float().eval()
    for parameter in vanilla.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator().manual_seed(17)
    image = torch.randn(2, 3, 224, 224, generator=generator).to(device)
    tokens = clip.tokenize(
        [
            "a photo of a person clearly feeling happiness.",
            "a photo of a person not feeling happiness.",
        ]
    ).to(device)
    with torch.no_grad():
        integrated_image = ddp_model.encode_prompt_free_image(
            image, normalize=False
        )
        vanilla_image = vanilla.encode_image(image).float()
        integrated_text = ddp_model.encode_prompt_free_text(
            tokens, normalize=False
        )
        vanilla_text = vanilla.encode_text(tokens).float()

    image_error = error_metrics(integrated_image, vanilla_image)
    text_error = error_metrics(integrated_text, vanilla_text)
    passed = (
        image_error["max_abs_error"] <= args.tolerance
        and text_error["max_abs_error"] <= args.tolerance
    )
    result = {
        "passed": passed,
        "tolerance": args.tolerance,
        "ddp_checkpoint": os.path.abspath(args.ddp_checkpoint),
        "clip_checkpoint": clip_path,
        "comparison_only_second_model": True,
        "training_uses_second_model": False,
        "image_cls": image_error,
        "fixed_text": text_error,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise RuntimeError(
            "DDP prompt-free route is not numerically equivalent to vanilla CLIP"
        )


if __name__ == "__main__":
    main()
