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
    temperature_for_task,
)
from models import ddp
from src.helper_functions.detail_report import DetailReport
from src.helper_functions.emotic_loader import EMOTIC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate all EMOTIC B5-C3 checkpoints at one threshold."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", default="./datasets/EMOTIC")
    parser.add_argument("--clip-model-path", default="./pretrained/clip/ViT-B-16.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("val", "test"),
        default=("val", "test"),
        help="EMOTIC annotation splits to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--t-min", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=2.0)
    parser.add_argument("--t-gamma", type=float, default=0.7)
    return parser.parse_args()


def class_mask():
    return [list(range(0, 5))] + [
        list(range(low, min(low + 3, 26))) for low in range(5, 26, 3)
    ]


def write_class_order(output_dir, run_name, classnames):
    detail_dir = os.path.join(output_dir, "detail")
    os.makedirs(detail_dir, exist_ok=True)
    rows = []
    for task_id, class_ids in enumerate(class_mask()):
        rows.append(
            {
                "task": task_id,
                "classes": [
                    {
                        "class_id": class_id,
                        "class_name": classnames[class_id],
                    }
                    for class_id in class_ids
                ],
            }
        )
    path = os.path.join(detail_dir, f"{run_name}_class_order.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = [
        os.path.join(args.checkpoint_dir, f"task{task_id}.pth")
        for task_id in range(8)
    ]
    for path in checkpoint_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    classnames = first_checkpoint["classnames"]
    first_saved_args = first_checkpoint.get("args", {})
    training_protocol = first_checkpoint.get(
        "ddp_main_loss_protocol",
        {
            "ddp_main_classification_loss": first_saved_args.get(
                "ddp_classification_loss", "two_way_bce"
            ),
            "loss_w": float(first_saved_args.get("loss_w", 0.03)),
            "checkpoint_rule": "fixed_last_epoch",
        },
    )
    model_args = checkpoint_model_args(first_checkpoint, args)
    cfg = setup_cfg(model_args)
    model = ddp(cfg, classnames)
    model = model.module if hasattr(model, "module") else model
    model.to(device).eval()

    eval_transform = transforms.Compose(
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
        transform=eval_transform,
        input_mode="full",
    )
    print(
        f"Evaluation splits={list(args.eval_splits)} "
        f"person_samples={len(dataset)} threshold={args.threshold:.2f}",
        flush=True,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    write_class_order(args.output_dir, args.name, classnames)
    report = DetailReport(
        args.output_dir,
        args.name,
        classnames,
        class_mask(),
        dataset.targets,
    )
    summary = {
        "eval_splits": list(args.eval_splits),
        "threshold": args.threshold,
        "t_min": args.t_min,
        "t_max": args.t_max,
        "t_gamma": args.t_gamma,
        "training_protocol": training_protocol,
        "rows": [],
    }

    for task_id, checkpoint_path in enumerate(checkpoint_paths):
        high_range = 5 if task_id == 0 else min(5 + task_id * 3, 26)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_loss = checkpoint.get("args", {}).get(
            "ddp_classification_loss", "two_way_bce"
        )
        expected_loss = training_protocol["ddp_main_classification_loss"]
        if checkpoint_loss != expected_loss:
            raise RuntimeError(
                f"Task {task_id} loss mismatch: {checkpoint_loss} != "
                f"{expected_loss}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        rebuild_text_feature_cache(model, high_range)

        indices = [
            index
            for index, target in enumerate(dataset.targets)
            if any(class_id < high_range for class_id in target)
        ]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        output_batches = []
        target_batches = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True).float()
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs = model(
                        inputs, cls_id=(0, high_range), inference=True
                    )
                output_batches.append(outputs.float().cpu())
                target_batches.append(targets.float().cpu())

        logits = torch.cat(output_batches)
        targets_full = torch.cat(target_batches)
        temperature = temperature_for_task(
            high_range,
            len(classnames),
            5,
            args.t_min,
            args.t_max,
            args.t_gamma,
        )
        scores_seen = torch.softmax(logits / temperature, dim=1)[:, 1, :]
        loss = F.binary_cross_entropy(
            scores_seen.clamp(1e-6, 1 - 1e-6),
            targets_full[:, :high_range],
            reduction="mean",
        ).item()
        scores_full = torch.zeros(len(indices), len(classnames))
        scores_full[:, :high_range] = scores_seen
        overall = report.update(
            task_id,
            scores_full,
            targets_full,
            args.threshold,
            loss,
        )
        row = {
            **overall,
            "temperature": temperature,
            "threshold": args.threshold,
            "checkpoint": checkpoint_path,
        }
        summary["rows"].append(row)
        torch.save(
            {
                "scores": scores_seen,
                "targets": targets_full[:, :high_range],
                "logits": logits,
                "temperature": temperature,
                "threshold": args.threshold,
            },
            os.path.join(args.output_dir, f"task{task_id}_scores.pt"),
        )
        print(
            f"[Task {task_id}] samples={len(indices)} T={temperature:.4f} "
            f"threshold={args.threshold:.2f} mAP={overall['mAP']:.2f} "
            f"amAP={overall['amAP']:.2f} oF1={overall['oF1']:.2f} "
            f"cF1={overall['cF1']:.2f}",
            flush=True,
        )

    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"Saved {summary_path}")
    print(f"Saved {report.path}")


if __name__ == "__main__":
    main()
