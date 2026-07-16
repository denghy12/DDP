import argparse
from pathlib import Path

import torch

from eval_emotic_ddp_internal_adapter import (
    TASK_SEEN_CLASSES,
    build_model,
    task_feature_cache,
)
from eval_emotic_threshold_sweep import rebuild_text_feature_cache
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache class-specific DDP CLS path features"
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument(
        "--cache_dir", default="./output/emotic_ddp_cls_internal_feature_cache"
    )
    parser.add_argument("--tasks", nargs="+", type=int, default=(0,))
    parser.add_argument(
        "--splits", nargs="+", choices=("val", "test"), default=("val",)
    )
    parser.add_argument("--feature_batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.feature_source = "cls"
    checkpoint_dir = Path(args.checkpoint_dir)
    first = torch.load(checkpoint_dir / "task0.pth", map_location="cpu")
    model = build_model(first, args, torch.device(args.device))
    datasets = {
        split: EMOTIC(
            args.data_root,
            train=False,
            eval_splits=(split,),
            transform=eval_transform(),
            input_mode="full",
        )
        for split in args.splits
    }
    labels = {split: dense_labels(dataset) for split, dataset in datasets.items()}
    for task_id in args.tasks:
        if not 0 <= task_id < len(TASK_SEEN_CLASSES):
            raise ValueError(f"Invalid task id: {task_id}")
        seen_classes = TASK_SEEN_CLASSES[task_id]
        checkpoint_path = checkpoint_dir / f"task{task_id}.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.feature_adapter = None
        model.load_state_dict(checkpoint["model"], strict=True)
        model.text_feature_cache.clear()
        rebuild_text_feature_cache(model, seen_classes)
        model.eval()
        for split in args.splits:
            payload = task_feature_cache(
                model,
                datasets[split],
                labels[split],
                split,
                task_id,
                seen_classes,
                checkpoint_path,
                args,
            )
            print(
                f"Cached task={task_id} split={split} "
                f"cls_shape={tuple(payload['path_features'].shape)} "
                f"pooled_shape={tuple(payload['pooled_features'].shape)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
