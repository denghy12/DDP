"""Train one strict task-specific Final-token Adapter on frozen DDP tokens.

Only train and validation splits are constructed.  The Adapter consumes all
197 final projected ViT tokens for the current task's +/- prompted paths, then
the exact original DDP pooling is recomputed.  The saved checkpoint is always
the fixed last epoch; validation is reporting-only and never selects weights.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import shutil
from html import escape
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from emotic_final_token_adapter import (
    FINAL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
    FINAL_TOKEN_FORMULA,
    SharedResidualFinalTokenAdapter,
    ddp_pool_final_tokens,
    file_sha256,
    final_token_drift_tensors,
    final_token_identity_loss,
    final_token_pooling_aware_losses,
    validate_final_token_checkpoint,
)
from emotic_task_adapter_bank import (
    prepare_task_training_subset,
    task_class_indices,
    task_class_range,
)
from eval_emotic_ddp_internal_adapter import build_model
from evaluation_metrics import mAP
from prototype_fewshot import masked_bce_with_logits, masked_pos_weight
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform, set_seed


CACHE_SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one task-specific EMOTIC Final-token Adapter"
    )
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--training_mode", choices=("full", "16shot"), required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_adapter_checkpoint")
    parser.add_argument("--ddp_checkpoint", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--token_cache_dir",
        default="./output/emotic_ddp_final_token_training_cache",
    )
    parser.add_argument("--feature_batch_size", type=int, default=2)
    parser.add_argument("--adapter_batch_size", type=int, default=4)
    parser.add_argument("--cache_shard_samples", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.03)
    parser.add_argument("--identity_weight", type=float, default=0.1)
    parser.add_argument(
        "--pooling_weight",
        type=float,
        default=0.0,
        help="Weight for pooled-feature cosine preservation.",
    )
    parser.add_argument(
        "--attention_weight",
        type=float,
        default=0.0,
        help="Weight for KL(original DDP attention || adapted attention).",
    )
    parser.add_argument(
        "--margin_weight",
        type=float,
        default=0.0,
        help="Weight for Smooth-L1 preservation of positive-negative margins.",
    )
    parser.add_argument(
        "--margin_beta",
        type=float,
        default=1.0,
        help="Smooth-L1 transition beta for DDP margin preservation.",
    )
    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument("--class_balanced_bce", action="store_true")
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument(
        "--training_health_check",
        action="store_true",
        help=(
            "Audit early gradients, weight updates, and non-zero residuals; "
            "abort immediately if the Adapter remains disconnected."
        ),
    )
    parser.add_argument(
        "--health_check_steps",
        type=int,
        default=2,
        help="Optimizer steps required by the early training-health audit.",
    )
    parser.add_argument(
        "--health_min_signal",
        type=float,
        default=1e-12,
        help="Minimum norm treated as a real gradient/update/residual signal.",
    )
    parser.add_argument(
        "--max_optimizer_steps",
        type=int,
        default=0,
        help="Optional diagnostic step cap; zero runs all configured epochs.",
    )
    parser.add_argument(
        "--full_precision_training",
        action="store_true",
        help=(
            "Disable CUDA autocast/GradScaler for Adapter optimization. "
            "Recommended when a non-zero task anchor makes early scaled "
            "gradients overflow even though the FP32 loss is finite."
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict):
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _parameter_grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().cpu())


def _weight_delta_norm(parameter: torch.Tensor, initial: torch.Tensor) -> float:
    return float((parameter.detach().float() - initial).norm().cpu())


def _cuda_amp_enabled(device, full_precision: bool = False) -> bool:
    """Keep AMP policy explicit so evaluation never depends on CLI scope."""

    return str(device).startswith("cuda") and not bool(full_precision)


def assess_training_health(observations, required_steps=2, min_signal=1e-12):
    """Decide whether a zero-initialized residual Adapter is trainable.

    The first step must update the zero-initialized up projection.  A later
    step must then expose a non-zero residual and gradient to the down
    projection.  This explicitly catches the dead-gradient failure caused by
    an exact-identity branch around the differentiable Adapter path.
    """

    required_steps = int(required_steps)
    min_signal = float(min_signal)
    inspected = list(observations[:required_steps])
    later = inspected[1:]
    finite_keys = (
        "loss",
        "up_grad_norm",
        "down_grad_norm",
        "up_weight_delta_norm",
        "raw_residual_rms",
        "adapted_feature_delta_rms",
    )
    checks = {
        "enough_optimizer_steps": len(inspected) >= required_steps,
        "all_observed_values_finite": bool(inspected)
        and all(
            np.isfinite(float(row[key]))
            for row in inspected
            for key in finite_keys
        ),
        "up_projection_receives_gradient": bool(inspected)
        and float(inspected[0]["up_grad_norm"]) > min_signal,
        "up_projection_updates": bool(inspected)
        and max(float(row["up_weight_delta_norm"]) for row in inspected)
        > min_signal,
        "residual_becomes_nonzero": bool(later)
        and max(float(row["raw_residual_rms"]) for row in later) > min_signal,
        "adapted_features_leave_identity": bool(later)
        and max(float(row["adapted_feature_delta_rms"]) for row in later)
        > min_signal,
        "down_projection_receives_gradient_after_up_update": bool(later)
        and max(float(row["down_grad_norm"]) for row in later) > min_signal,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "required_steps": required_steps,
        "minimum_signal": min_signal,
        "checks": checks,
        "observations": inspected,
    }


def write_health_html(path: Path, report: dict):
    observations = report.get("observations", [])
    keys = list(observations[0]) if observations else []
    headers = "".join(f"<th>{escape(key)}</th>" for key in keys)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in keys)
        + "</tr>"
        for row in observations
    )
    checks = "".join(
        f"<li><b>{escape(name)}</b>: {escape(str(passed))}</li>"
        for name, passed in report.get("checks", {}).items()
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Final-token Adapter Training Health</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #d9deea;padding:7px;text-align:right}"
        "th{background:#416fbd;color:white}"
        ".passed{color:#087f5b}.failed{color:#c92a2a}</style>"
        f"<h1>Final-token Adapter Training Health: "
        f"<span class='{escape(report.get('status', 'failed'))}'>"
        f"{escape(report.get('status', 'unknown').upper())}</span></h1>"
        f"<ul>{checks}</ul>"
        f"<table><tr>{headers}</tr>{rows}</table>"
        f"<h2>Full report</h2><pre>"
        f"{escape(json.dumps(report, indent=2, ensure_ascii=False))}</pre>",
        encoding="utf-8",
    )


def save_health_report(output_dir: Path, report: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "training_health_check.json", report)
    write_health_html(output_dir / "training_health_check.html", report)


def _cache_directory(args, split: str) -> Path:
    root = Path(args.token_cache_dir) / f"task{args.task_id}"
    if split == "val":
        return root / "val_current"
    if args.training_mode == "full":
        return root / "train_full"
    return root / f"train_16shot_seed{args.seed}"


def _expected_cache_metadata(
    args,
    split,
    source_indices,
    class_range,
):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "task_id": int(args.task_id),
        "class_range": list(class_range),
        "ddp_checkpoint": os.path.abspath(args.ddp_checkpoint),
        "ddp_checkpoint_sha256": file_sha256(args.ddp_checkpoint),
        "source_indices": [int(index) for index in source_indices],
        "feature_source": "final_projected_vit_tokens",
        "path_order": "ddp_existing_negative_then_positive",
        "expected_token_count": 197,
        "expected_feature_dim": 512,
    }


def _flush_cache_shard(
    cache_dir: Path,
    shard_id: int,
    token_batches,
    label_batches,
    mask_batches,
):
    tokens = torch.cat(token_batches)
    labels = torch.cat(label_batches)
    masks = torch.cat(mask_batches)
    name = f"shard_{shard_id:05d}.pt"
    temporary = cache_dir / f"{name}.tmp.{os.getpid()}"
    torch.save(
        {
            "token_features": tokens,
            "labels": labels,
            "supervision_mask": masks,
        },
        temporary,
    )
    os.replace(temporary, cache_dir / name)
    return {
        "file": name,
        "samples": int(tokens.shape[0]),
        "shape": list(tokens.shape),
    }


def build_or_load_token_cache(
    model,
    dataset,
    labels,
    source_indices,
    supervision_mask,
    split,
    args,
):
    """Write an FP16, sharded cache without ever concatenating it in RAM."""

    low, high = task_class_range(args.task_id)
    active = task_class_indices(args.task_id)
    cache_dir = _cache_directory(args, split)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(cache_dir) + ".lock")
    expected = _expected_cache_metadata(
        args, split, source_indices, (low, high)
    )
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.is_file() and not args.force_recache:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("metadata") != expected:
                raise RuntimeError(
                    f"Final-token cache metadata mismatch: {manifest_path}"
                )
            for entry in manifest.get("shards", []):
                if not (cache_dir / entry["file"]).is_file():
                    raise RuntimeError(f"Incomplete token cache: {cache_dir}")
            if not (cache_dir / "text_features.pt").is_file():
                raise RuntimeError(f"Missing text features: {cache_dir}")
            return manifest

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        loader = DataLoader(
            Subset(dataset, [int(index) for index in source_indices]),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=args.device.startswith("cuda"),
        )
        shards = []
        token_batches = []
        label_batches = []
        mask_batches = []
        buffered = 0
        offset = 0
        text_features = None
        model.eval()
        with torch.no_grad():
            for batch_id, (images, _) in enumerate(loader):
                images = images.to(args.device, non_blocking=True).float()
                with torch.cuda.amp.autocast(
                    enabled=args.device.startswith("cuda")
                ):
                    pooled, base_logits, batch_text, token_features = (
                        model.extract_path_features(
                            images,
                            cls_id=(low, high),
                            inference=True,
                            return_token_features=True,
                        )
                    )
                del pooled, base_logits
                batch_samples = token_features.shape[0]
                token_batches.append(token_features.half().cpu())
                selected_labels = labels[source_indices[offset : offset + batch_samples]]
                label_batches.append(selected_labels[:, active].float())
                mask_batches.append(
                    supervision_mask[offset : offset + batch_samples, active].bool()
                )
                offset += batch_samples
                buffered += batch_samples
                text_features = batch_text.float().cpu()
                if buffered >= args.cache_shard_samples:
                    shards.append(
                        _flush_cache_shard(
                            cache_dir,
                            len(shards),
                            token_batches,
                            label_batches,
                            mask_batches,
                        )
                    )
                    token_batches, label_batches, mask_batches = [], [], []
                    buffered = 0
                if batch_id % 50 == 0:
                    print(
                        f"[Task {args.task_id} {split}] final tokens "
                        f"{batch_id + 1}/{len(loader)}",
                        flush=True,
                    )
        if token_batches:
            shards.append(
                _flush_cache_shard(
                    cache_dir,
                    len(shards),
                    token_batches,
                    label_batches,
                    mask_batches,
                )
            )
        if offset != len(source_indices):
            raise RuntimeError("Final-token cache sample count changed during extraction")
        torch.save(text_features, cache_dir / "text_features.pt")
        manifest = {
            "metadata": expected,
            "samples": int(offset),
            "shards": shards,
            "text_features": "text_features.pt",
            "storage_dtype": "float16",
        }
        _atomic_json(manifest_path, manifest)
        return manifest


def load_task_ddp(args, device):
    checkpoint = torch.load(args.ddp_checkpoint, map_location="cpu")
    if int(checkpoint.get("task", -1)) != int(args.task_id):
        raise ValueError(
            f"Task {args.task_id} training requires task{args.task_id}.pth; "
            f"checkpoint reports task {checkpoint.get('task')}"
        )
    model = build_model(checkpoint, args, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    low, high = task_class_range(args.task_id)
    model.text_feature_cache.clear()
    with torch.no_grad():
        # Cache only the current task's text paths.  ``inference=True`` during
        # token extraction then reuses these features instead of repeatedly
        # running the frozen text encoder for every image batch.
        model._text_features((low, high), inference=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def final_token_logits(adapter, token_features, text_features):
    token_last = token_features.permute(0, 1, 3, 2).float()
    adapted, original, residual = adapter(token_last)
    adapted_ddp = adapted.permute(0, 1, 3, 2).contiguous()
    _, path_logits, _, path_weights = ddp_pool_final_tokens(
        adapted_ddp, text_features.float()
    )
    logits = path_logits.reshape(
        path_logits.shape[0], 2, path_logits.shape[1] // 2
    )
    return logits, adapted, original, residual, path_weights


def _distribution_summary(values: torch.Tensor):
    values = values.detach().float().flatten().cpu()
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty diagnostic tensor")
    return {
        "mean": float(values.mean()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _finalize_validation_diagnostics(parts):
    attention = torch.cat(parts["attention_kl"])
    pooled = torch.cat(parts["pooled_cosine_drift"])
    logit_delta = torch.cat(parts["path_logit_delta"])
    token_delta = torch.cat(parts["token_delta_l2"])
    absolute_logit_delta = logit_delta.abs()
    return {
        "definition": {
            "attention_kl": "KL(original_positive_attention || adapted_positive_attention)",
            "pooled_feature_cosine_drift": "1 - cosine(adapted_pool, original_pool)",
            "path_logit_drift": "adapted pre-softmax DDP path logit - original path logit",
            "token_delta": "L2 norm of adapted token - original token",
            "cls_token_index": 0,
            "patch_token_indices": "1:197",
        },
        "attention_kl": _distribution_summary(attention),
        "pooled_feature_cosine_drift": _distribution_summary(pooled),
        "path_logit_absolute_drift": _distribution_summary(absolute_logit_delta),
        "path_logit_rms_drift": float(logit_delta.square().mean().sqrt()),
        "path_logit_signed_mean_drift": float(logit_delta.mean()),
        "all_token_delta_l2": _distribution_summary(token_delta),
        "cls_token_delta_l2": _distribution_summary(token_delta[:, :, 0]),
        "patch_token_delta_l2": _distribution_summary(token_delta[:, :, 1:]),
        "samples": int(token_delta.shape[0]),
        "paths_per_sample": int(token_delta.shape[1]),
        "tokens_per_path": int(token_delta.shape[2]),
    }


def evaluate_cache(adapter, cache_dir, device, batch_size):
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    text_features = torch.load(
        cache_dir / manifest["text_features"], map_location="cpu"
    ).to(device)
    scores = []
    targets = []
    drift_parts = {
        "attention_kl": [],
        "pooled_cosine_drift": [],
        "path_logit_delta": [],
        "token_delta_l2": [],
    }
    adapter.eval()
    amp_enabled = _cuda_amp_enabled(device)
    with torch.no_grad():
        for entry in manifest["shards"]:
            shard = torch.load(cache_dir / entry["file"], map_location="cpu")
            for start in range(0, entry["samples"], batch_size):
                stop = start + batch_size
                tokens = shard["token_features"][start:stop].to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits, adapted, original, _, _ = final_token_logits(
                        adapter, tokens, text_features
                    )
                with torch.cuda.amp.autocast(enabled=False):
                    drift = final_token_drift_tensors(
                        adapted, original, text_features
                    )
                scores.append(torch.softmax(logits, dim=1)[:, 1, :].cpu())
                targets.append(shard["labels"][start:stop].cpu())
                drift_parts["attention_kl"].append(
                    drift["attention_kl_original_to_adapted"].cpu()
                )
                drift_parts["pooled_cosine_drift"].append(
                    drift["pooled_feature_cosine_drift"].cpu()
                )
                drift_parts["path_logit_delta"].append(
                    drift["path_logit_delta"].cpu()
                )
                drift_parts["token_delta_l2"].append(
                    drift["token_delta_l2"].cpu()
                )
    scores = torch.cat(scores)
    targets = torch.cat(targets)
    score, per_class = mAP(targets.numpy(), scores.numpy())
    diagnostics = _finalize_validation_diagnostics(drift_parts)
    return (
        float(score),
        [100.0 * float(value) for value in per_class],
        diagnostics,
    )


def train_adapter(adapter, train_cache_dir, val_cache_dir, active_names, args, device):
    train_manifest = json.loads(
        (train_cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    text_features = torch.load(
        train_cache_dir / train_manifest["text_features"], map_location="cpu"
    ).to(device)
    all_labels = []
    all_masks = []
    for entry in train_manifest["shards"]:
        shard = torch.load(train_cache_dir / entry["file"], map_location="cpu")
        all_labels.append(shard["labels"])
        all_masks.append(shard["supervision_mask"])
    labels_for_weight = torch.cat(all_labels)
    masks_for_weight = torch.cat(all_masks)
    pos_weight = None
    if args.class_balanced_bce:
        local_indices = torch.arange(labels_for_weight.shape[1])
        pos_weight = masked_pos_weight(
            labels_for_weight, masks_for_weight, local_indices
        ).to(device)
    del all_labels, all_masks, labels_for_weight, masks_for_weight

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    amp_enabled = _cuda_amp_enabled(device, args.full_precision_training)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    initial_val_map, initial_per_class, initial_val_diagnostics = evaluate_cache(
        adapter, val_cache_dir, device, args.adapter_batch_size
    )
    initial_up_weight = adapter.up.weight.detach().float().clone()
    initial_down_weight = adapter.down.weight.detach().float().clone()
    history = []
    health_observations = []
    health_report = {"enabled": False, "status": "not_requested"}
    health_audited = False
    global_step = 0
    stop_training = False
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        adapter.train()
        shard_order = torch.randperm(
            len(train_manifest["shards"]), generator=generator
        ).tolist()
        totals = {
            "loss": 0.0,
            "classification_loss": 0.0,
            "identity_loss": 0.0,
            "pooling_loss": 0.0,
            "attention_loss": 0.0,
            "margin_loss": 0.0,
        }
        sample_count = 0
        last_step_metrics = None
        for shard_id in shard_order:
            entry = train_manifest["shards"][shard_id]
            shard = torch.load(
                train_cache_dir / entry["file"], map_location="cpu"
            )
            sample_order = torch.randperm(entry["samples"], generator=generator)
            for start in range(0, entry["samples"], args.adapter_batch_size):
                indices = sample_order[start : start + args.adapter_batch_size]
                tokens = shard["token_features"][indices].to(device)
                labels = shard["labels"][indices].to(device)
                masks = shard["supervision_mask"][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits, adapted, original, residual, _ = final_token_logits(
                        adapter, tokens, text_features
                    )
                    margin = logits[:, 1, :] - logits[:, 0, :]
                    class_loss = masked_bce_with_logits(
                        margin, labels, masks, pos_weight=pos_weight
                    )
                    identity_loss = final_token_identity_loss(adapted, original)
                with torch.cuda.amp.autocast(enabled=False):
                    structural_losses = final_token_pooling_aware_losses(
                        adapted.float(),
                        original.float(),
                        text_features.float(),
                        margin_beta=args.margin_beta,
                    )
                    pooling_loss = structural_losses["pooling_loss"]
                    attention_loss = structural_losses["attention_loss"]
                    margin_loss = structural_losses["margin_loss"]
                    loss = (
                        class_loss.float()
                        + args.identity_weight * identity_loss.float()
                        + args.pooling_weight * pooling_loss
                        + args.attention_weight * attention_loss
                        + args.margin_weight * margin_loss
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                global_step += 1
                residual_rms = residual.detach().float().square().mean().sqrt()
                feature_delta = adapted.detach().float() - original.detach().float()
                last_step_metrics = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.detach().float().cpu()),
                    "classification_loss": float(class_loss.detach().float().cpu()),
                    "identity_loss": float(identity_loss.detach().float().cpu()),
                    "pooling_loss": float(pooling_loss.detach().float().cpu()),
                    "attention_loss": float(
                        attention_loss.detach().float().cpu()
                    ),
                    "margin_loss": float(margin_loss.detach().float().cpu()),
                    "up_grad_norm": _parameter_grad_norm(adapter.up.weight),
                    "down_grad_norm": _parameter_grad_norm(adapter.down.weight),
                    "raw_residual_rms": float(residual_rms.cpu()),
                    "scaled_residual_rms": float(
                        (args.residual_scale * residual_rms).cpu()
                    ),
                    "adapted_feature_delta_rms": float(
                        feature_delta.square().mean().sqrt().cpu()
                    ),
                    "adapted_feature_delta_max": float(
                        feature_delta.abs().max().cpu()
                    ),
                }
                scaler.step(optimizer)
                scaler.update()
                last_step_metrics.update(
                    {
                        "up_weight_delta_norm": _weight_delta_norm(
                            adapter.up.weight, initial_up_weight
                        ),
                        "down_weight_delta_norm": _weight_delta_norm(
                            adapter.down.weight, initial_down_weight
                        ),
                    }
                )
                if (
                    args.training_health_check
                    and global_step <= args.health_check_steps
                ):
                    health_observations.append(dict(last_step_metrics))
                if (
                    args.training_health_check
                    and not health_audited
                    and global_step >= args.health_check_steps
                ):
                    health_report = assess_training_health(
                        health_observations,
                        required_steps=args.health_check_steps,
                        min_signal=args.health_min_signal,
                    )
                    health_report.update(
                        {
                            "enabled": True,
                            "task_id": int(args.task_id),
                            "training_mode": args.training_mode,
                            "seed": int(args.seed),
                            "initial_val_mAP": initial_val_map,
                            "implementation": (
                                "always_differentiable_normalized_residual_path"
                            ),
                        }
                    )
                    save_health_report(Path(args.output_dir), health_report)
                    health_audited = True
                    print(
                        "Training health audit: "
                        f"{health_report['status'].upper()} after "
                        f"{global_step} optimizer steps",
                        flush=True,
                    )
                    if health_report["status"] != "passed":
                        raise RuntimeError(
                            "Final-token Adapter failed the early training-health "
                            "audit; see training_health_check.json"
                        )
                count = int(indices.numel())
                sample_count += count
                totals["loss"] += float(loss.detach()) * count
                totals["classification_loss"] += float(class_loss.detach()) * count
                totals["identity_loss"] += float(identity_loss.detach()) * count
                totals["pooling_loss"] += float(pooling_loss.detach()) * count
                totals["attention_loss"] += float(attention_loss.detach()) * count
                totals["margin_loss"] += float(margin_loss.detach()) * count
                if args.max_optimizer_steps > 0 and global_step >= args.max_optimizer_steps:
                    stop_training = True
                    break
            if stop_training:
                break
        if sample_count == 0:
            raise RuntimeError("No samples were processed during Adapter training")
        row = {
            "epoch": epoch,
            "optimizer_steps": global_step,
            "loss": totals["loss"] / sample_count,
            "classification_loss": totals["classification_loss"] / sample_count,
            "identity_loss": totals["identity_loss"] / sample_count,
            "pooling_loss": totals["pooling_loss"] / sample_count,
            "attention_loss": totals["attention_loss"] / sample_count,
            "margin_loss": totals["margin_loss"] / sample_count,
            "weighted_pooling_loss": (
                args.pooling_weight * totals["pooling_loss"] / sample_count
            ),
            "weighted_attention_loss": (
                args.attention_weight * totals["attention_loss"] / sample_count
            ),
            "weighted_margin_loss": (
                args.margin_weight * totals["margin_loss"] / sample_count
            ),
            "up_weight_delta_norm": _weight_delta_norm(
                adapter.up.weight, initial_up_weight
            ),
            "down_weight_delta_norm": _weight_delta_norm(
                adapter.down.weight, initial_down_weight
            ),
        }
        if last_step_metrics is not None:
            row.update(
                {
                    "last_up_grad_norm": last_step_metrics["up_grad_norm"],
                    "last_down_grad_norm": last_step_metrics["down_grad_norm"],
                    "last_raw_residual_rms": last_step_metrics[
                        "raw_residual_rms"
                    ],
                    "last_adapted_feature_delta_rms": last_step_metrics[
                        "adapted_feature_delta_rms"
                    ],
                }
            )
        history.append(row)
        print(
            f"Task {args.task_id} epoch {epoch:03d}: "
            f"loss={row['loss']:.6f}, cls={row['classification_loss']:.6f}, "
            f"identity={row['identity_loss']:.6f}, "
            f"pool={row['pooling_loss']:.6g}, "
            f"attn={row['attention_loss']:.6g}, "
            f"margin={row['margin_loss']:.6g}, "
            f"steps={row['optimizer_steps']}",
            flush=True,
        )
        if stop_training:
            print(
                f"Stopped diagnostic training at optimizer step {global_step}",
                flush=True,
            )
            break
    if args.training_health_check and not health_audited:
        health_report = assess_training_health(
            health_observations,
            required_steps=args.health_check_steps,
            min_signal=args.health_min_signal,
        )
        health_report.update(
            {
                "enabled": True,
                "task_id": int(args.task_id),
                "training_mode": args.training_mode,
                "seed": int(args.seed),
                "initial_val_mAP": initial_val_map,
                "implementation": "always_differentiable_normalized_residual_path",
            }
        )
        save_health_report(Path(args.output_dir), health_report)
        if health_report["status"] != "passed":
            raise RuntimeError(
                "Final-token Adapter did not complete enough healthy optimizer "
                "steps; see training_health_check.json"
            )
    final_val_map, final_per_class, final_val_diagnostics = evaluate_cache(
        adapter, val_cache_dir, device, args.adapter_batch_size
    )
    if args.training_health_check:
        health_report.update(
            {
                "completed_optimizer_steps": global_step,
                "completed_epochs_or_partial_epochs": len(history),
                "final_val_mAP": final_val_map,
                "reporting_val_gain": final_val_map - initial_val_map,
                "final_validation_diagnostics": final_val_diagnostics,
                "final_up_weight_delta_norm": _weight_delta_norm(
                    adapter.up.weight, initial_up_weight
                ),
                "final_down_weight_delta_norm": _weight_delta_norm(
                    adapter.down.weight, initial_down_weight
                ),
            }
        )
        save_health_report(Path(args.output_dir), health_report)
    return {
        "history": history,
        "optimizer_steps": global_step,
        "last_completed_epoch": history[-1]["epoch"],
        "stopped_at_max_optimizer_steps": stop_training,
        "training_health_check": health_report,
        "initial_val_mAP": initial_val_map,
        "initial_per_class_ap": dict(zip(active_names, initial_per_class)),
        "initial_validation_diagnostics": initial_val_diagnostics,
        "final_val_mAP": final_val_map,
        "final_per_class_ap": dict(zip(active_names, final_per_class)),
        "final_validation_diagnostics": final_val_diagnostics,
    }


def write_training_html(path: Path, summary: dict):
    rows = summary["history"]
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in row)
        + "</tr>"
        for row in rows
    )
    headers = "".join(f"<th>{escape(key)}</th>" for key in rows[0]) if rows else ""
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Final-token Adapter Training</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:7px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>Task {summary['final_token_adapter']['task_id']} Final-token Adapter</h1>"
        f"<pre>{escape(json.dumps(summary['protocol'], indent=2, ensure_ascii=False))}</pre>"
        f"<table><tr>{headers}</tr>{body}</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    low, high = task_class_range(args.task_id)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.training_mode == "16shot" and args.shots_per_class != 16:
        raise ValueError("The locked few-shot protocol requires exactly 16 shots")
    if float(args.residual_scale) != 0.03:
        raise ValueError("The locked Final-token protocol requires alpha=0.03")
    if args.health_check_steps < 2:
        raise ValueError("health_check_steps must be at least 2")
    if args.health_min_signal <= 0:
        raise ValueError("health_min_signal must be positive")
    if any(
        weight < 0
        for weight in (
            args.identity_weight,
            args.pooling_weight,
            args.attention_weight,
            args.margin_weight,
        )
    ):
        raise ValueError("all regularization weights must be non-negative")
    if args.margin_beta <= 0:
        raise ValueError("margin_beta must be positive")
    if args.max_optimizer_steps < 0:
        raise ValueError("max_optimizer_steps cannot be negative")
    if (
        args.training_health_check
        and args.max_optimizer_steps > 0
        and args.max_optimizer_steps < args.health_check_steps
    ):
        raise ValueError(
            "max_optimizer_steps must reach all requested health-check steps"
        )
    if args.task_id == 0 and args.init_adapter_checkpoint:
        raise ValueError("Task 0 must use identity initialization")
    if args.task_id > 0 and not args.init_adapter_checkpoint:
        raise ValueError("Tasks 1-7 require the matching task0 anchor")

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, ddp_checkpoint = load_task_ddp(args, device)
    classnames = list(ddp_checkpoint["classnames"])
    train_dataset = EMOTIC(
        args.data_root,
        train=True,
        transform=eval_transform(),
        input_mode="full",
        class_names=classnames,
    )
    val_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=eval_transform(),
        input_mode="full",
        class_names=classnames,
    )
    train_labels = dense_labels(train_dataset)
    val_labels = dense_labels(val_dataset)
    selected, supervision_mask, sampling = prepare_task_training_subset(
        train_labels,
        args.task_id,
        args.training_mode,
        args.seed,
        shots_per_class=args.shots_per_class,
    )
    active = task_class_indices(args.task_id)
    val_selected = torch.nonzero(
        val_labels[:, active].sum(dim=1).gt(0), as_tuple=False
    ).flatten()
    val_mask = torch.zeros(
        (len(val_selected), 26), dtype=torch.bool
    )
    val_mask[:, active] = True
    train_manifest = build_or_load_token_cache(
        model,
        train_dataset,
        train_labels,
        selected,
        supervision_mask,
        "train",
        args,
    )
    val_manifest = build_or_load_token_cache(
        model,
        val_dataset,
        val_labels,
        val_selected,
        val_mask,
        "val",
        args,
    )
    del model
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    adapter = SharedResidualFinalTokenAdapter(
        feature_dim=512,
        bottleneck_dim=args.adapter_dim,
        residual_scale=args.residual_scale,
    ).to(device)
    initialization = {"kind": "identity", "source": None, "sha256": None}
    if args.init_adapter_checkpoint:
        init_path = Path(args.init_adapter_checkpoint)
        init_checkpoint = torch.load(init_path, map_location="cpu")
        validate_final_token_checkpoint(
            init_checkpoint,
            task_id=0,
            training_mode=args.training_mode,
            seed=args.seed,
            classnames=classnames,
        )
        adapter.load_state_dict(init_checkpoint["model"], strict=True)
        initialization = {
            "kind": "task0_anchor",
            "source": os.path.abspath(init_path),
            "sha256": file_sha256(init_path),
        }

    training = train_adapter(
        adapter,
        _cache_directory(args, "train"),
        _cache_directory(args, "val"),
        classnames[low:high],
        args,
        device,
    )
    metadata = {
        "schema_version": FINAL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
        "task_id": args.task_id,
        "class_range": [low, high],
        "class_ids": active.tolist(),
        "class_names": classnames[low:high],
        "training_mode": args.training_mode,
        "seed": args.seed,
        "formula": FINAL_TOKEN_FORMULA,
        "token_count": 197,
        "initialization": initialization,
        "ddp_checkpoint": os.path.abspath(args.ddp_checkpoint),
        "ddp_checkpoint_sha256": file_sha256(args.ddp_checkpoint),
    }
    checkpoint = {
        "model": {
            key: value.detach().cpu().clone()
            for key, value in adapter.state_dict().items()
        },
        "epoch": training["last_completed_epoch"],
        "optimizer_steps": training["optimizer_steps"],
        "selection_split": None,
        "selection_score": None,
        "reporting_val_mAP": training["final_val_mAP"],
        "classnames": classnames,
        "final_token_adapter": metadata,
        "args": {
            **vars(args),
            "feature_dim": 512,
            "adapter_dim": args.adapter_dim,
            "residual_scale": args.residual_scale,
            "formula": FINAL_TOKEN_FORMULA,
        },
        "sampling": sampling,
        "training_source": "task_native_prompted_final_197_tokens",
        "inference_target": "original_ddp_token_attention_pooling",
    }
    checkpoint_path = output_dir / "final_adapter.pth"
    torch.save(checkpoint, checkpoint_path)
    protocol = {
        "dataset": "EMOTIC",
        "incremental_protocol": "B5-C3",
        "task_id": args.task_id,
        "ddp_checkpoint_matches_task": True,
        "feature_source": "197 final projected prompted ViT tokens",
        "pooling": "exact original DDP token attention",
        "current_labels_supervised": True,
        "old_labels_supervised": False,
        "future_labels_supervised": False,
        "test_dataset_constructed": False,
        "test_labels_used": False,
        "test_used_for_selection": False,
        "checkpoint_selection": (
            "fixed final diagnostic optimizer step"
            if args.max_optimizer_steps > 0
            else "fixed last epoch"
        ),
        "validation_role": "post-training reporting only",
        "residual_scale_locked": args.residual_scale,
        "decision_threshold_locked": 0.5,
        "class_specific_gate": False,
        "task_specific_alpha": False,
        "training_health_check_enabled": args.training_health_check,
        "max_optimizer_steps": args.max_optimizer_steps,
        "pooling_aware_regularization": {
            "enabled": any(
                weight > 0
                for weight in (
                    args.pooling_weight,
                    args.attention_weight,
                    args.margin_weight,
                )
            ),
            "teacher": "frozen original DDP prompted-token route",
            "pooling_weight": args.pooling_weight,
            "attention_weight": args.attention_weight,
            "attention_kl_implementation": "log_softmax_stable",
            "margin_weight": args.margin_weight,
            "margin_beta": args.margin_beta,
            "margin_definition": "positive path logit - negative path logit",
        },
        "training_precision": (
            "float32" if args.full_precision_training else "cuda_amp"
        ),
    }
    summary = {
        "protocol": protocol,
        "final_token_adapter": metadata,
        "initial_val_mAP": training["initial_val_mAP"],
        "final_val_mAP": training["final_val_mAP"],
        "reporting_val_gain": (
            training["final_val_mAP"] - training["initial_val_mAP"]
        ),
        "initial_per_class_ap": training["initial_per_class_ap"],
        "initial_validation_diagnostics": training[
            "initial_validation_diagnostics"
        ],
        "final_per_class_ap": training["final_per_class_ap"],
        "final_validation_diagnostics": training[
            "final_validation_diagnostics"
        ],
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "sampling": sampling,
        "train_cache": train_manifest,
        "val_cache": val_manifest,
        "history": training["history"],
        "optimizer_steps": training["optimizer_steps"],
        "stopped_at_max_optimizer_steps": training[
            "stopped_at_max_optimizer_steps"
        ],
        "training_health_check": training["training_health_check"],
        "regularization": protocol["pooling_aware_regularization"],
        "checkpoint": {
            "path": os.path.abspath(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "args": vars(args),
    }
    _atomic_json(output_dir / "training_summary.json", summary)
    write_training_html(output_dir / "training_history.html", summary)
    print(
        json.dumps(
            {
                "task": args.task_id,
                "training_mode": args.training_mode,
                "fixed_epoch": training["last_completed_epoch"],
                "optimizer_steps": training["optimizer_steps"],
                "training_health": training["training_health_check"]["status"],
                "reporting_val_mAP": training["final_val_mAP"],
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
