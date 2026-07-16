import argparse
import json
import os
import random
from html import escape
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from build_cfg import setup_cfg
from ddp_internal_adapter import (
    CORRECTION_MODES,
    feature_identity_loss,
    masked_ddp_bce,
)
from eval_emotic_threshold_sweep import (
    checkpoint_model_args,
    rebuild_text_feature_cache,
)
from evaluation_metrics import mAP
from models import ddp
from prototype_fewshot import masked_pos_weight, sample_multilabel_kshot
from src.helper_functions.emotic_loader import EMOTIC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a shared adapter inside frozen task0 DDP features"
    )
    parser.add_argument("--ddp_checkpoint", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--feature_cache_dir",
        default=None,
        help="Optional shared task0 path-feature cache directory",
    )
    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument(
        "--full_base5",
        action="store_true",
        help="Use every task0 Base5 training sample instead of K-shot sampling",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument(
        "--correction_mode",
        choices=CORRECTION_MODES,
        default="linear_residual",
        help=(
            "Adapter output rule used during internal training. "
            "feature_correction is reserved for paired CLS-to-pooled transfer."
        ),
    )
    parser.add_argument("--identity_weight", type=float, default=0.1)
    parser.add_argument(
        "--loss_balance",
        choices=("balanced", "unweighted"),
        default="balanced",
    )
    parser.add_argument("--eval_every_steps", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def dense_labels(dataset, total_classes=26):
    labels = torch.zeros(len(dataset), total_classes, dtype=torch.float32)
    for index, target in enumerate(dataset.targets):
        labels[index, [int(class_id) for class_id in target]] = 1.0
    return labels


def load_frozen_ddp(checkpoint_path, clip_model_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint["task"]) != 0:
        raise ValueError("Internal Adapter training requires the task0 checkpoint")
    helper_args = argparse.Namespace(clip_model_path=clip_model_path)
    cfg = setup_cfg(checkpoint_model_args(checkpoint, helper_args))
    model = ddp(cfg, checkpoint["classnames"])
    model = model.module if hasattr(model, "module") else model
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rebuild_text_feature_cache(model, 5)
    return model, checkpoint


def extract_path_cache(model, dataset, source_indices, labels, args, cache_path):
    expected = {
        "ddp_checkpoint": os.path.abspath(args.ddp_checkpoint),
        "source_indices": [int(index) for index in source_indices],
        "seen_classes": 5,
    }
    if cache_path.is_file() and not args.force_recache:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") != expected:
            raise RuntimeError(f"Feature cache metadata mismatch: {cache_path}")
        return payload

    loader = DataLoader(
        Subset(dataset, source_indices),
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    pooled_batches = []
    base_logit_batches = []
    text_features = None
    with torch.no_grad():
        for batch_id, (images, _) in enumerate(loader):
            images = images.to(args.device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                pooled, base_logits, batch_text = model.extract_path_features(
                    images, cls_id=(0, 5), inference=True
                )
            pooled_batches.append(pooled.half().cpu())
            base_logit_batches.append(base_logits.float().cpu())
            text_features = batch_text.float().cpu()
            if batch_id % 25 == 0:
                print(
                    f"Feature extraction {cache_path.name}: "
                    f"{batch_id + 1}/{len(loader)}",
                    flush=True,
                )
    payload = {
        "pooled_features": torch.cat(pooled_batches),
        "base_path_logits": torch.cat(base_logit_batches),
        "text_features": text_features,
        "labels": labels[source_indices, :5].float(),
        "metadata": expected,
    }
    torch.save(payload, cache_path)
    return payload


def adapter_validation(model, payload, batch_size, device):
    loader = DataLoader(
        TensorDataset(
            payload["pooled_features"], payload["base_path_logits"]
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    text_features = payload["text_features"].to(device)
    outputs = []
    residual_ratios = []
    model.eval()
    with torch.no_grad():
        for pooled, base_logits in loader:
            logits, aux = model.logits_from_path_features(
                pooled.to(device).float(),
                base_logits.to(device),
                text_features,
                return_adapter_aux=True,
            )
            outputs.append(torch.softmax(logits, dim=1)[:, 1, :].cpu())
            residual = (aux["adapted"] - aux["original"]).norm(dim=-1)
            original = aux["original"].norm(dim=-1).clamp_min(1e-12)
            residual_ratios.append((residual / original).cpu())
    scores = torch.cat(outputs)
    score, _ = mAP(
        payload["labels"].numpy(),
        scores.numpy(),
    )
    parameter_norm = sum(
        parameter.detach().float().pow(2).sum().item()
        for parameter in model.feature_adapter.parameters()
    ) ** 0.5
    ratios = torch.cat(residual_ratios)
    return {
        "mAP": float(score),
        "parameter_l2_norm": float(parameter_norm),
        "residual_ratio_mean": float(ratios.mean()),
        "residual_ratio_max": float(ratios.max()),
    }


def write_history_html(path, history):
    rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in history[0])
        + "</tr>"
        for row in history
    )
    headers = "".join(f"<th>{escape(key)}</th>" for key in history[0])
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Internal Adapter</title>"
        "<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:right}</style>"
        f"<h1>DDP Internal Adapter Training</h1><table><tr>{headers}</tr>"
        f"{rows}</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.correction_mode == "feature_correction":
        raise ValueError(
            "feature_correction requires paired CLS and DDP pooled features. "
            "Use screen_emotic_ddp_internal_adapter_transfer.py and "
            "eval_emotic_ddp_internal_adapter.py for the external-to-internal "
            "CLS transfer experiment."
        )
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if args.early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be positive")

    model, checkpoint = load_frozen_ddp(
        args.ddp_checkpoint, args.clip_model_path, device
    )
    train_dataset = EMOTIC(
        args.data_root, train=True, transform=eval_transform(), input_mode="full"
    )
    val_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=eval_transform(),
        input_mode="full",
    )
    if list(train_dataset.classes) != list(checkpoint["classnames"]):
        raise RuntimeError("DDP checkpoint and EMOTIC class orders differ")
    train_labels = dense_labels(train_dataset)
    val_labels = dense_labels(val_dataset)
    base_source = torch.nonzero(
        train_labels[:, :5].sum(dim=1).gt(0), as_tuple=False
    ).flatten()
    if args.full_base5:
        selected_source = base_source
        supervision_mask = torch.ones(
            len(selected_source), 5, dtype=torch.bool
        )
        sampling_rows = []
    else:
        selected_local, supervision_mask, sampling_rows = sample_multilabel_kshot(
            train_labels[base_source],
            torch.arange(5),
            args.shots_per_class,
            args.seed,
        )
        supervision_mask = supervision_mask[:, :5]
        selected_source = base_source[selected_local]
    val_source = torch.nonzero(
        val_labels[:, :5].sum(dim=1).gt(0), as_tuple=False
    ).flatten()
    sampling = {
        "mode": "full_base5" if args.full_base5 else "fewshot",
        "shots_per_class": None if args.full_base5 else args.shots_per_class,
        "seed": args.seed,
        "unique_training_samples": int(selected_source.numel()),
        "selected_train_indices": selected_source.tolist(),
        "sampling": sampling_rows,
    }
    with open(output_dir / "fewshot_sampling.json", "w", encoding="utf-8") as fp:
        json.dump(sampling, fp, indent=2, ensure_ascii=False)

    cache_dir = (
        Path(args.feature_cache_dir)
        if args.feature_cache_dir is not None
        else output_dir
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_name = (
        "task0_train_full_base5_path_features.pt"
        if args.full_base5
        else "task0_train_path_features.pt"
    )
    train_cache = extract_path_cache(
        model,
        train_dataset,
        selected_source.tolist(),
        train_labels,
        args,
        cache_dir / train_cache_name,
    )
    val_cache = extract_path_cache(
        model,
        val_dataset,
        val_source.tolist(),
        val_labels,
        args,
        cache_dir / "task0_val_path_features.pt",
    )
    model.enable_feature_adapter(
        args.adapter_dim,
        args.residual_scale,
        correction_mode=args.correction_mode,
    )
    adapter = model.feature_adapter
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = None
    if args.loss_balance == "balanced":
        pos_weight = masked_pos_weight(
            train_cache["labels"], supervision_mask, torch.arange(5)
        ).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            train_cache["pooled_features"],
            train_cache["base_path_logits"],
            train_cache["labels"],
            supervision_mask,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    train_text = train_cache["text_features"].to(device)

    first_pooled = train_cache["pooled_features"][:2].to(device).float()
    first_base = train_cache["base_path_logits"][:2].to(device)
    with torch.no_grad():
        zero_logits = model.logits_from_path_features(
            first_pooled, first_base, train_text
        )
    expected_logits = first_base.reshape(2, 2, 5)
    identity_max_abs_error = float((zero_logits - expected_logits).abs().max())
    if identity_max_abs_error > 1e-7:
        raise RuntimeError(
            f"Zero-init Adapter changed DDP logits by {identity_max_abs_error}"
        )

    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }
    zero_validation = adapter_validation(
        model, val_cache, args.batch_size, device
    )
    zero_val_map = zero_validation["mAP"]
    best_map = zero_val_map
    best_epoch = -1
    best_step = 0
    best_state = initial_state
    best_diagnostics = zero_validation
    history = [
        {
            "epoch": -1,
            "step": 0,
            "loss": None,
            "class_loss": None,
            "identity_loss": 0.0,
            "val_mAP": zero_val_map,
            **{
                key: value
                for key, value in zero_validation.items()
                if key != "mAP"
            },
        }
    ]
    global_step = 0
    checks_without_improvement = 0
    stopped_early = False
    for epoch in range(args.epochs):
        adapter.train()
        for pooled, base_logits, labels, mask in loader:
            pooled = pooled.to(device).float()
            base_logits = base_logits.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            logits, aux = model.logits_from_path_features(
                pooled,
                base_logits,
                train_text,
                return_adapter_aux=True,
            )
            class_loss = masked_ddp_bce(
                logits, labels, mask, pos_weight=pos_weight
            )
            identity_loss = feature_identity_loss(
                aux["adapted"], aux["original"]
            )
            loss = class_loss + args.identity_weight * identity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % args.eval_every_steps != 0:
                continue
            validation = adapter_validation(
                model, val_cache, args.batch_size, device
            )
            val_map = validation["mAP"]
            row = {
                "epoch": epoch,
                "step": global_step,
                "loss": loss.item(),
                "class_loss": class_loss.item(),
                "identity_loss": identity_loss.item(),
                "val_mAP": val_map,
                **{
                    key: value
                    for key, value in validation.items()
                    if key != "mAP"
                },
            }
            history.append(row)
            if val_map > best_map:
                best_map = val_map
                best_epoch = epoch
                best_step = global_step
                best_diagnostics = validation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in adapter.state_dict().items()
                }
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            print(
                f"epoch={epoch:03d} step={global_step:04d} "
                f"loss={row['loss']:.6f} val_mAP={val_map:.4f} "
                f"best_gain={best_map - zero_val_map:+.4f}",
                flush=True,
            )
            if checks_without_improvement >= args.early_stop_patience:
                stopped_early = True
                break
        if stopped_early:
            break

    adapter.load_state_dict(best_state)
    adapter_checkpoint = {
        "model": best_state,
        "epoch": best_epoch,
        "step": best_step,
        "val_mAP": best_map,
        "zero_shot_val_mAP": zero_val_map,
        "best_val_gain": best_map - zero_val_map,
        "adaptation_selected": best_step > 0,
        "diagnostics": best_diagnostics,
        "identity_max_abs_error": identity_max_abs_error,
        "classnames": list(checkpoint["classnames"]),
        "args": vars(args),
        "sampling": sampling,
    }
    torch.save(adapter_checkpoint, output_dir / "best_adapter.pth")
    summary = {
        "protocol": {
            "ddp_frozen": True,
            "training_task": 0,
            "active_classes": list(range(5)),
            "future_labels_used": False,
            "test_used": False,
        },
        "best_epoch": best_epoch,
        "best_step": best_step,
        "zero_shot_val_mAP": zero_val_map,
        "best_val_mAP": best_map,
        "best_val_gain": best_map - zero_val_map,
        "adaptation_selected": best_step > 0,
        "best_diagnostics": best_diagnostics,
        "stopped_early": stopped_early,
        "identity_max_abs_error": identity_max_abs_error,
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "sampling": sampling,
        "history": history,
        "args": vars(args),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_history_html(output_dir / "training_history.html", history)
    print(
        f"Best epoch={best_epoch}, val_mAP={best_map:.4f}; "
        f"saved {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
