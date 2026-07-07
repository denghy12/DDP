import argparse
import csv
import json
import math
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from build_cfg import setup_cfg
from models import ddp
from opts import arg_parser
from src.helper_functions.detail_report import (
    average_precision,
    binary_counts,
)
from src.helper_functions.emotic_loader import EMOTIC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an EMOTIC DDP checkpoint across global thresholds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./datasets/EMOTIC")
    parser.add_argument("--clip-model-path", default="./pretrained/clip/ViT-B-16.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("val", "test"),
        default=("val", "test"),
        help="EMOTIC annotation splits to evaluate",
    )
    parser.add_argument("--threshold-start", type=float, default=0.30)
    parser.add_argument("--threshold-end", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--t-min", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=2.0)
    parser.add_argument("--t-gamma", type=float, default=0.7)
    return parser.parse_args()


def checkpoint_model_args(checkpoint, cli_args):
    saved = checkpoint.get("args", {})
    arguments = [
        "--config_file",
        saved.get("config_file") or "configs/models/vitb16_ep50.yaml",
        "--dataset_config_file",
        saved.get("dataset_config_file") or "configs/datasets/emotic.yaml",
        "--clip_model_path",
        cli_args.clip_model_path,
        "--positive_prompt",
        saved.get("positive_prompt") or "a photo of a person clearly feeling",
        "--negative_prompt",
        saved.get("negative_prompt") or "a photo of a person not feeling",
    ]
    if saved.get("csc", True):
        arguments.append("--csc")
    return arg_parser().parse_args(arguments)


@torch.no_grad()
def rebuild_text_feature_cache(model, high_range):
    prompts, tokenized_prompts = model.prompt_learner((0, high_range))
    text_features = model.text_encoder(prompts, tokenized_prompts)
    text_features = F.normalize(text_features, dim=-1)
    model.text_feature_cache.clear()
    model.text_feature_cache[(0, high_range)] = {
        "neg": text_features[:high_range].detach().cpu(),
        "pos": text_features[high_range:].detach().cpu(),
    }


def temperature_for_task(
    high_range, total_classes, base_classes, t_min, t_max, gamma
):
    progress = (high_range - base_classes) / (total_classes - base_classes)
    progress = max(0.0, min(1.0, progress))
    return t_min + (t_max - t_min) * math.pow(progress, gamma)


def thresholds(start, end, step):
    count = int(round((end - start) / step))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    values.extend([2.0 / 3.0, 0.8])
    return sorted({value for value in values if start <= value <= end})


def metrics_at_threshold(scores, targets, threshold):
    predictions = scores.gt(threshold)
    targets_bool = targets.bool()
    overall = binary_counts(predictions, targets_bool)
    class_rows = []
    for class_id in range(scores.shape[1]):
        counts = binary_counts(
            predictions[:, class_id], targets_bool[:, class_id]
        )
        counts["ap"] = average_precision(
            scores[:, class_id], targets[:, class_id]
        )
        class_rows.append(counts)

    mean = lambda key: 100.0 * sum(row[key] for row in class_rows) / len(
        class_rows
    )
    return {
        "threshold": float(threshold),
        "mAP": mean("ap"),
        "oPrecision": 100.0 * overall["precision"],
        "oRecall": 100.0 * overall["recall"],
        "oF1": 100.0 * overall["f1"],
        "cPrecision": mean("precision"),
        "cRecall": mean("recall"),
        "cF1": mean("f1"),
        "predicted_positive": overall["predicted_positive"],
        "support": overall["support"],
        "tp": overall["tp"],
        "fp": overall["fp"],
        "fn": overall["fn"],
    }


def write_results(output_dir, metadata, rows):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "threshold_sweep.json")
    csv_path = os.path.join(output_dir, "threshold_sweep.csv")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {"metadata": metadata, "rows": rows},
            fp,
            ensure_ascii=False,
            indent=2,
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    task_id = int(checkpoint["task"])
    classnames = checkpoint["classnames"]
    total_classes = len(classnames)
    base_classes = 5
    is_upper_bound = bool(checkpoint.get("args", {}).get("upper_bound", False))
    if is_upper_bound:
        high_range = total_classes
    elif task_id == 0:
        high_range = base_classes
    else:
        high_range = min(base_classes + task_id * 3, total_classes)

    model_args = checkpoint_model_args(checkpoint, args)
    cfg = setup_cfg(model_args)
    model = ddp(cfg, classnames)
    model = model.module if hasattr(model, "module") else model
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    rebuild_text_feature_cache(model, high_range)

    image_size = 224
    val_transform = transforms.Compose(
        [
            transforms.Resize(
                int((256 / 224) * image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=tuple(args.eval_splits),
        transform=val_transform,
        input_mode="full",
    )
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
        for inputs, targets_batch in loader:
            inputs = inputs.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(
                    inputs, cls_id=(0, high_range), inference=True
                )
            output_batches.append(output.float().cpu())
            target_batches.append(targets_batch[:, :high_range].float())

    logits = torch.cat(output_batches)
    targets_tensor = torch.cat(target_batches)
    temperature = 1.0 if is_upper_bound else temperature_for_task(
        high_range, total_classes, base_classes,
        args.t_min, args.t_max, args.t_gamma,
    )
    scores = torch.softmax(logits / temperature, dim=1)[:, 1, :]
    rows = [
        metrics_at_threshold(scores, targets_tensor, threshold)
        for threshold in thresholds(
            args.threshold_start,
            args.threshold_end,
            args.threshold_step,
        )
    ]
    best_of1 = max(rows, key=lambda row: row["oF1"])
    best_cf1 = max(rows, key=lambda row: row["cF1"])
    best_balanced = max(
        rows, key=lambda row: (row["oF1"] + row["cF1"]) / 2
    )
    metadata = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "task": task_id,
        "seen_classes": high_range,
        "samples": len(indices),
        "eval_splits": list(args.eval_splits),
        "temperature": temperature,
        "upper_bound": is_upper_bound,
        "best_oF1": best_of1,
        "best_cF1": best_cf1,
        "best_mean_oF1_cF1": best_balanced,
    }
    json_path, csv_path = write_results(args.output_dir, metadata, rows)
    torch.save(
        {"scores": scores, "targets": targets_tensor, "logits": logits},
        os.path.join(args.output_dir, "task_scores.pt"),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
