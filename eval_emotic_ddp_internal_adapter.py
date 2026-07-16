import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from build_cfg import setup_cfg
from ddp_internal_adapter import CORRECTION_MODES
from eval_emotic_prototype_fusion import score_metrics, select_threshold
from eval_emotic_threshold_sweep import (
    checkpoint_model_args,
    rebuild_text_feature_cache,
    temperature_for_task,
)
from evaluation_metrics import mAP
from models import ddp
from src.helper_functions.detail_report import DetailReport
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform


TASK_SEEN_CLASSES = (5, 8, 11, 14, 17, 20, 23, 26)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict all-task evaluation of a frozen internal DDP Adapter"
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--adapter_checkpoint")
    parser.add_argument(
        "--ddp_only",
        action="store_true",
        help="Evaluate the frozen original DDP logits without an Adapter",
    )
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--cache_dir", default="./output/emotic_ddp_internal_feature_cache"
    )
    parser.add_argument(
        "--feature_source",
        choices=("pooled", "cls"),
        default="pooled",
        help="Apply the transferred Adapter to pooled path features or CLS tokens",
    )
    parser.add_argument(
        "--correction_mode",
        choices=CORRECTION_MODES,
        default=None,
        help=(
            "Override the Adapter checkpoint correction mode. If omitted, "
            "use checkpoint metadata and fall back to linear_residual."
        ),
    )
    parser.add_argument(
        "--baseline_scores_dir",
        default="./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050",
    )
    parser.add_argument("--feature_batch_size", type=int, default=4)
    parser.add_argument("--adapter_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=2.0)
    parser.add_argument("--t_gamma", type=float, default=0.7)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def class_mask():
    return [list(range(5))] + [
        list(range(low, min(low + 3, 26))) for low in range(5, 26, 3)
    ]


def build_model(first_checkpoint, args, device):
    helper_args = argparse.Namespace(clip_model_path=args.clip_model_path)
    cfg = setup_cfg(checkpoint_model_args(first_checkpoint, helper_args))
    model = ddp(cfg, first_checkpoint["classnames"])
    model = model.module if hasattr(model, "module") else model
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_task_model(
    model,
    ddp_checkpoint,
    adapter_checkpoint,
    seen_classes,
    correction_mode=None,
    ddp_only=False,
):
    model.feature_adapter = None
    model.load_state_dict(ddp_checkpoint["model"], strict=True)
    model.text_feature_cache.clear()
    rebuild_text_feature_cache(model, seen_classes)
    if ddp_only:
        return "none"
    adapter_args = adapter_checkpoint["args"]
    checkpoint_mode = adapter_args.get("correction_mode", "linear_residual")
    resolved_mode = correction_mode or checkpoint_mode
    model.enable_feature_adapter(
        bottleneck_dim=int(adapter_args["adapter_dim"]),
        residual_scale=float(adapter_args["residual_scale"]),
        correction_mode=resolved_mode,
    )
    model.feature_adapter.load_state_dict(adapter_checkpoint["model"], strict=True)
    model.eval()
    return resolved_mode


def task_feature_cache(
    model,
    dataset,
    labels,
    split,
    task_id,
    seen_classes,
    checkpoint_path,
    args,
):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_suffix = "cls_" if args.feature_source == "cls" else ""
    path = cache_dir / f"task{task_id}_{split}_{source_suffix}path_features.pt"
    source_indices = torch.nonzero(
        labels[:, :seen_classes].sum(dim=1).gt(0), as_tuple=False
    ).flatten()
    metadata = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "split": split,
        "task": task_id,
        "seen_classes": seen_classes,
        "source_indices": source_indices.tolist(),
    }
    if args.feature_source == "cls":
        metadata["feature_source"] = "cls"
    lock_path = Path(str(path) + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.is_file() and not args.force_recache:
            payload = torch.load(path, map_location="cpu")
            if payload.get("metadata") != metadata:
                raise RuntimeError(f"Feature cache metadata mismatch: {path}")
            if not (
                args.feature_source == "cls"
                and "pooled_features" not in payload
            ):
                return payload
            print(
                f"Rebuilding legacy CLS cache without pooled_features: {path}",
                flush=True,
            )

        loader = DataLoader(
            Subset(dataset, source_indices.tolist()),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=args.device.startswith("cuda"),
        )
        pooled_batches = []
        ddp_pooled_batches = []
        base_logit_batches = []
        text_features = None
        with torch.no_grad():
            for batch_id, (images, _) in enumerate(loader):
                images = images.to(args.device, non_blocking=True).float()
                with torch.cuda.amp.autocast(
                    enabled=args.device.startswith("cuda")
                ):
                    extracted = model.extract_path_features(
                        images,
                        cls_id=(0, seen_classes),
                        inference=True,
                        return_cls_features=args.feature_source == "cls",
                    )
                    if args.feature_source == "cls":
                        pooled, base_logits, batch_text, cls_features = extracted
                        selected_features = cls_features
                        ddp_pooled_batches.append(pooled.half().cpu())
                    else:
                        pooled, base_logits, batch_text = extracted
                        selected_features = pooled
                pooled_batches.append(selected_features.half().cpu())
                base_logit_batches.append(base_logits.float().cpu())
                text_features = batch_text.float().cpu()
                if batch_id % 100 == 0:
                    print(
                        f"[Task {task_id} {split}] features "
                        f"{batch_id + 1}/{len(loader)}",
                        flush=True,
                    )
        payload = {
            "path_features": torch.cat(pooled_batches),
            "base_path_logits": torch.cat(base_logit_batches),
            "text_features": text_features,
            "targets": labels[source_indices].float(),
            "metadata": metadata,
            "cache_schema_version": 2,
        }
        if args.feature_source == "cls":
            payload["pooled_features"] = torch.cat(ddp_pooled_batches)
        temporary = Path(str(path) + f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return payload


def predict_from_cache(model, payload, seen_classes, args):
    features = payload.get("path_features")
    if features is None:
        if args.feature_source != "pooled":
            raise RuntimeError("CLS cache does not contain path_features")
        features = payload["pooled_features"]
    correction_mode = getattr(
        model, "feature_adapter_correction", "linear_residual"
    )
    uses_feature_correction = (
        not args.ddp_only and correction_mode == "feature_correction"
    )
    if uses_feature_correction:
        ddp_pooled = payload.get("pooled_features")
        if ddp_pooled is None:
            raise RuntimeError(
                "feature_correction requires a CLS cache containing "
                "pooled_features; rebuild the cache with "
                "cache_emotic_ddp_cls_features.py"
            )
        dataset = TensorDataset(
            features,
            ddp_pooled,
            payload["base_path_logits"],
        )
    else:
        dataset = TensorDataset(features, payload["base_path_logits"])
    loader = DataLoader(
        dataset,
        batch_size=args.adapter_batch_size,
        shuffle=False,
    )
    text_features = payload["text_features"].to(args.device)
    temperature = temperature_for_task(
        seen_classes, 26, 5, args.t_min, args.t_max, args.t_gamma
    )
    scores = []
    with torch.no_grad():
        for batch in loader:
            if uses_feature_correction:
                pooled, ddp_pooled, base_logits = batch
            else:
                pooled, base_logits = batch
                ddp_pooled = None
            if args.ddp_only:
                logits = base_logits.to(args.device).float().reshape(
                    pooled.shape[0], 2, seen_classes
                )
            else:
                logits = model.logits_from_path_features(
                    pooled.to(args.device).float(),
                    base_logits.to(args.device),
                    text_features,
                    ddp_pooled_features=(
                        None
                        if ddp_pooled is None
                        else ddp_pooled.to(args.device).float()
                    ),
                )
            scores.append(
                torch.softmax(logits / temperature, dim=1)[:, 1, :].cpu()
            )
    return torch.cat(scores), temperature


def baseline_map(path):
    payload = torch.load(path, map_location="cpu")
    value, _ = mAP(payload["targets"].numpy(), payload["scores"].numpy())
    return float(value)


def forgetting(rows, classnames):
    per_class = []
    for class_id, class_name in enumerate(classnames):
        intro = 0 if class_id < 5 else 1 + (class_id - 5) // 3
        history = [
            rows[task_id]["test"]["per_class_ap"][class_name]
            for task_id in range(intro, 8)
        ]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "introduction_task": intro,
                "peak_ap": max(history),
                "final_ap": history[-1],
                "forgetting": max(history) - history[-1],
            }
        )
    old = [row for row in per_class if row["introduction_task"] < 7]
    return {
        "average_forgetting_old_classes": float(
            np.mean([row["forgetting"] for row in old])
        ),
        "per_class": per_class,
    }


def main():
    args = parse_args()
    if args.ddp_only and args.adapter_checkpoint is not None:
        raise ValueError("--ddp_only cannot be combined with --adapter_checkpoint")
    if not args.ddp_only and args.adapter_checkpoint is None:
        raise ValueError("--adapter_checkpoint is required unless --ddp_only is set")
    if args.ddp_only and args.correction_mode is not None:
        raise ValueError("--ddp_only cannot be combined with --correction_mode")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [
        Path(args.checkpoint_dir) / f"task{task_id}.pth" for task_id in range(8)
    ]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    adapter_checkpoint = None
    if not args.ddp_only:
        adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    classnames = list(first_checkpoint["classnames"])
    if adapter_checkpoint is not None and classnames != list(
        adapter_checkpoint["classnames"]
    ):
        raise RuntimeError("Adapter and DDP class orders differ")
    model = build_model(first_checkpoint, args, torch.device(args.device))

    val_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=eval_transform(),
        input_mode="full",
    )
    test_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("test",),
        transform=eval_transform(),
        input_mode="full",
    )
    val_labels = dense_labels(val_dataset)
    test_labels = dense_labels(test_dataset)
    report = DetailReport(
        str(output_dir), args.name, classnames, class_mask(), test_dataset.targets
    )
    rows = []
    resolved_correction_mode = None
    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        ddp_checkpoint = torch.load(checkpoint_paths[task_id], map_location="cpu")
        task_correction_mode = load_task_model(
            model,
            ddp_checkpoint,
            adapter_checkpoint,
            seen_classes,
            correction_mode=args.correction_mode,
            ddp_only=args.ddp_only,
        )
        if resolved_correction_mode is None:
            resolved_correction_mode = task_correction_mode
        elif resolved_correction_mode != task_correction_mode:
            raise RuntimeError("Correction mode changed across tasks")
        val_cache = task_feature_cache(
            model,
            val_dataset,
            val_labels,
            "val",
            task_id,
            seen_classes,
            checkpoint_paths[task_id],
            args,
        )
        test_cache = task_feature_cache(
            model,
            test_dataset,
            test_labels,
            "test",
            task_id,
            seen_classes,
            checkpoint_paths[task_id],
            args,
        )
        val_scores, temperature = predict_from_cache(
            model, val_cache, seen_classes, args
        )
        test_scores, _ = predict_from_cache(model, test_cache, seen_classes, args)
        threshold, threshold_rows = select_threshold(
            val_cache["targets"][:, :seen_classes], val_scores, args
        )
        val_metrics = score_metrics(
            val_cache["targets"][:, :seen_classes],
            val_scores,
            threshold["threshold"],
            classnames[:seen_classes],
        )
        test_metrics = score_metrics(
            test_cache["targets"][:, :seen_classes],
            test_scores,
            threshold["threshold"],
            classnames[:seen_classes],
        )
        full_scores = torch.zeros(test_scores.shape[0], 26)
        full_scores[:, :seen_classes] = test_scores
        loss = F.binary_cross_entropy(
            test_scores.clamp(1e-6, 1 - 1e-6),
            test_cache["targets"][:, :seen_classes],
        ).item()
        report_metrics = report.update(
            task_id,
            full_scores,
            test_cache["targets"],
            threshold["threshold"],
            loss,
        )
        baseline_path = Path(args.baseline_scores_dir) / f"task{task_id}_scores.pt"
        ddp_map = baseline_map(baseline_path)
        row = {
            "task": task_id,
            "seen_classes": seen_classes,
            "temperature": temperature,
            "selection_split": "val",
            "selected_threshold": threshold,
            "threshold_sweep": threshold_rows,
            "val": val_metrics,
            "test": test_metrics,
            "baseline_ddp_test_mAP": ddp_map,
            "test_mAP_gain": test_metrics["mAP"] - ddp_map,
            "report": report_metrics,
        }
        rows.append(row)
        torch.save(
            {
                "scores": test_scores,
                "targets": test_cache["targets"][:, :seen_classes],
                "temperature": temperature,
                "threshold": threshold["threshold"],
                "adapter_checkpoint": args.adapter_checkpoint,
                "correction_mode": task_correction_mode,
                "norm_preserving": task_correction_mode == "feature_correction",
                "feature_source": args.feature_source,
                "evaluation_mode": "ddp_only" if args.ddp_only else "adapter",
            },
            output_dir / f"task{task_id}_scores.pt",
        )
        print(
            f"[Task {task_id}] test mAP {ddp_map:.4f} -> "
            f"{test_metrics['mAP']:.4f} "
            f"delta={test_metrics['mAP'] - ddp_map:+.4f}",
            flush=True,
        )

    task_maps = [row["test"]["mAP"] for row in rows]
    baseline_maps = [row["baseline_ddp_test_mAP"] for row in rows]
    summary = {
        "protocol": {
            "name": (
                "EMOTIC B5-C3 frozen original DDP"
                if args.ddp_only
                else "EMOTIC B5-C3 frozen DDP internal shared Adapter"
            ),
            "adapter_training_task": None if args.ddp_only else 0,
            "adapter_frozen_after_task0": None if args.ddp_only else True,
            "validation_role": (
                "per-task decision-threshold selection only"
                if args.ddp_only
                else "global residual-scale selection recorded by the Adapter "
                "checkpoint, plus per-task decision-threshold selection"
            ),
            "test_used_for_selection": False,
            "external_prototype_fusion": False,
            "adapter_training_source": (
                None
                if adapter_checkpoint is None
                else adapter_checkpoint.get("transfer", {}).get(
                    "adapter_training_source"
                )
            ),
            "training_only_prompt_free_auxiliary": (
                False
                if adapter_checkpoint is None
                else adapter_checkpoint.get("transfer", {}).get(
                    "adapter_training_source"
                )
                == "ddp_owned_prompt_free_auxiliary_branch"
            ),
            "prompt_free_auxiliary_retained_for_inference": False,
            "task_specific_alpha": False,
            "class_specific_gate": False,
            "feature_source": args.feature_source,
            "correction_mode": resolved_correction_mode,
            "norm_preserving": (
                resolved_correction_mode == "feature_correction"
            ),
            "feature_correction_target": (
                "ddp_pooled_features"
                if resolved_correction_mode == "feature_correction"
                else None
            ),
            "ddp_only": args.ddp_only,
        },
        "inputs": {
            "checkpoint_dir": args.checkpoint_dir,
            "adapter_checkpoint": args.adapter_checkpoint,
            "adapter_epoch": (
                None if adapter_checkpoint is None else adapter_checkpoint.get("epoch")
            ),
            "checkpoint_correction_mode": (
                None
                if adapter_checkpoint is None
                else adapter_checkpoint["args"].get(
                    "correction_mode", "linear_residual"
                )
            ),
            "transfer_selection": (
                None
                if adapter_checkpoint is None
                else adapter_checkpoint.get("transfer")
            ),
        },
        "tasks": rows,
        "aggregate": {
            "average_mAP": float(np.mean(task_maps)),
            "final_mAP": task_maps[-1],
            "baseline_average_mAP": float(np.mean(baseline_maps)),
            "baseline_final_mAP": baseline_maps[-1],
            "average_mAP_gain": float(np.mean(task_maps) - np.mean(baseline_maps)),
            "final_mAP_gain": task_maps[-1] - baseline_maps[-1],
            "forgetting": forgetting(rows, classnames),
        },
        "args": vars(args),
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"Saved {output_dir / 'evaluation_summary.json'}", flush=True)
    print(f"Saved {report.path}", flush=True)


if __name__ == "__main__":
    main()
