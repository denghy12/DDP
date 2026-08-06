"""GPU integration smoke test for the Transformer Adapter insertion."""

import argparse
import json
from pathlib import Path

import torch

from emotic_transformer_adapter_bank import (
    TaskRoutedTransformerAdapterBank,
    TransformerTaskAdapter,
)
from train_emotic_ddp_transformer_adapter import load_frozen_task_ddp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ddp_checkpoint",
        default="./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth",
    )
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_transformer_adapter_bank_smoke",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    helper = argparse.Namespace(
        ddp_checkpoint=args.ddp_checkpoint,
        clip_model_path=args.clip_model_path,
    )
    model, _ = load_frozen_task_ddp(helper, expected_task=0, device=device)
    torch.manual_seed(0)
    images = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        baseline = model(images, cls_id=(0, 5), inference=False).float()

    adapter = TransformerTaskAdapter().to(device)
    bank = TaskRoutedTransformerAdapterBank({0: adapter}).to(device)
    model.enable_transformer_adapter_bank(bank, freeze=False)
    model.eval()
    bank.train()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        identity = model(images, cls_id=(0, 5), inference=False).float()
    max_abs_error = float((identity - baseline).abs().max())

    model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        train_logits = model(images, cls_id=(0, 5), inference=False).float()
        loss = train_logits.square().mean()
    loss.backward()
    gradient_sum = float(
        sum(
            parameter.grad.abs().sum()
            for parameter in adapter.parameters()
            if parameter.grad is not None
        )
    )
    passed = max_abs_error <= 1e-7 and gradient_sum > 0
    result = {
        "passed": passed,
        "zero_adapter_max_abs_logit_error": max_abs_error,
        "adapter_gradient_abs_sum": gradient_sum,
        "logit_shape": list(baseline.shape),
        "adapter_parameters_per_task": adapter.parameter_count,
        "ddp_checkpoint": str(Path(args.ddp_checkpoint).resolve()),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit("Transformer Adapter integration smoke test failed")


if __name__ == "__main__":
    main()
