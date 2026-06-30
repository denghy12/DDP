import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from clip import clip
from evaluation_metrics import mAP
from prototype_adapter import (
    NEGATIVE_TEMPLATES,
    POSITIVE_TEMPLATES,
    ResidualPrototypeAdapter,
    protocol_class_indices,
)
from src.helper_functions.emotic_loader import EMOTIC


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a frozen-CLIP prototype adapter on EMOTIC features"
    )
    parser.add_argument("--datadir", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--cache_dir", default="./output/emotic_clip_feature_cache"
    )
    parser.add_argument("--protocol", choices=("all26", "base5"), required=True)
    parser.add_argument("--base_classes", type=int, default=5)
    parser.add_argument("--total_classes", type=int, default=26)
    parser.add_argument(
        "--emotic_input_mode", choices=("full", "person_crop"), default="full"
    )
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--adapter_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--identity_weight", type=float, default=0.1)
    parser.add_argument("--initial_logit_scale", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--class_balanced_bce", action="store_true")
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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


def build_clip_transform(input_size):
    interpolation = transforms.InterpolationMode.BICUBIC
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=interpolation),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def load_frozen_clip(model_path, device):
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"CLIP model not found: {model_path}")
    # Build an eager model from the official JIT checkpoint's state dict.
    # The legacy JIT graph patcher in clip.load(jit=True) uses a Node API that
    # is incompatible with PyTorch 2.0.1.
    model, _ = clip.load(model_path, device=device, jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def encode_template_ensemble(model, classnames, templates, device):
    template_features = []
    with torch.no_grad():
        for template in templates:
            texts = [template.format(name.lower()) for name in classnames]
            tokens = clip.tokenize(texts).to(device)
            features = model.encode_text(tokens).float()
            template_features.append(F.normalize(features, dim=-1))
    return F.normalize(torch.stack(template_features).mean(dim=0), dim=-1).cpu()


def extract_features(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    feature_batches = []
    label_batches = []
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = model.encode_image(images).float()
            feature_batches.append(F.normalize(features, dim=-1).cpu())
            label_batches.append(labels.float().cpu())
            if batch_id % 50 == 0:
                print(
                    f"Feature extraction: batch {batch_id + 1}/{len(loader)}",
                    flush=True,
                )
    return torch.cat(feature_batches), torch.cat(label_batches)


def cache_metadata(args, classnames, split):
    return {
        "split": split,
        "input_mode": args.emotic_input_mode,
        "input_size": args.input_size,
        "clip_model": os.path.basename(args.clip_model_path),
        "classnames": list(classnames),
    }


def validate_cache(payload, expected_metadata, path):
    if payload.get("metadata") != expected_metadata:
        raise RuntimeError(
            f"Feature cache metadata mismatch at {path}. "
            "Use --force_recache to rebuild it."
        )
    if payload["features"].shape[0] != payload["labels"].shape[0]:
        raise RuntimeError(f"Feature/label count mismatch in {path}")


def load_or_extract_split(model, dataset, split, args, device):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{split}_{args.emotic_input_mode}_{args.input_size}_vitb16.pt"
    )
    metadata = cache_metadata(args, dataset.classes, split)
    if cache_path.is_file() and not args.force_recache:
        payload = torch.load(cache_path, map_location="cpu")
        validate_cache(payload, metadata, cache_path)
        print(f"Loaded feature cache: {cache_path}")
        return payload["features"].float(), payload["labels"].float()

    features, labels = extract_features(
        model,
        dataset,
        args.feature_batch_size,
        args.num_workers,
        device,
    )
    torch.save(
        {"metadata": metadata, "features": features, "labels": labels},
        cache_path,
    )
    print(f"Saved feature cache: {cache_path}")
    return features, labels


def binary_metrics(labels, probabilities, threshold):
    labels = labels.bool()
    predictions = probabilities.ge(threshold)
    tp = (predictions & labels).sum(dim=0).float()
    fp = (predictions & ~labels).sum(dim=0).float()
    fn = (~predictions & labels).sum(dim=0).float()

    class_precision = tp / (tp + fp).clamp_min(1.0)
    class_recall = tp / (tp + fn).clamp_min(1.0)
    class_f1 = (
        2 * class_precision * class_recall
        / (class_precision + class_recall).clamp_min(1e-12)
    )
    overall_precision = tp.sum() / (tp + fp).sum().clamp_min(1.0)
    overall_recall = tp.sum() / (tp + fn).sum().clamp_min(1.0)
    overall_f1 = (
        2
        * overall_precision
        * overall_recall
        / (overall_precision + overall_recall).clamp_min(1e-12)
    )
    return {
        "cPrecision": 100 * class_precision.mean().item(),
        "cRecall": 100 * class_recall.mean().item(),
        "cF1": 100 * class_f1.mean().item(),
        "oPrecision": 100 * overall_precision.item(),
        "oRecall": 100 * overall_recall.item(),
        "oF1": 100 * overall_f1.item(),
    }


def subset_map(labels, probabilities, indices):
    if not indices or labels.shape[0] == 0:
        return None
    score, _ = mAP(
        labels[:, indices].numpy(), probabilities[:, indices].numpy()
    )
    return float(score)


def evaluate_logits(logits, labels, classnames, base_classes, threshold):
    probabilities = torch.sigmoid(logits.cpu())
    total_classes = labels.shape[1]
    base_indices = list(range(min(base_classes, total_classes)))
    novel_indices = list(range(min(base_classes, total_classes), total_classes))
    base_seen_mask = labels[:, base_indices].sum(dim=1).gt(0)
    full_map, per_class_ap = mAP(labels.numpy(), probabilities.numpy())
    return {
        "samples": int(labels.shape[0]),
        "mAP": float(full_map),
        "base_mAP": subset_map(labels, probabilities, base_indices),
        "base_seen_mAP": subset_map(
            labels[base_seen_mask],
            probabilities[base_seen_mask],
            base_indices,
        ),
        "novel_mAP": subset_map(labels, probabilities, novel_indices),
        "threshold": threshold,
        **binary_metrics(labels, probabilities, threshold),
        "per_class_ap": {
            name: 100 * float(per_class_ap[index])
            for index, name in enumerate(classnames)
        },
    }


def evaluate_model(model, features, labels, classnames, args, device):
    model.eval()
    logits = []
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=args.adapter_batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for feature_batch, _ in loader:
            batch_logits, _, _ = model(feature_batch.to(device))
            logits.append(batch_logits.cpu())
    return evaluate_logits(
        torch.cat(logits),
        labels,
        classnames,
        args.base_classes,
        args.threshold,
    )


def compact_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "per_class_ap"}


def make_pos_weight(labels, active_indices):
    active_labels = labels[:, active_indices]
    positives = active_labels.sum(dim=0)
    negatives = active_labels.shape[0] - positives
    return (negatives / positives.clamp_min(1.0)).clamp(max=20.0)


def evaluate_splits(model, split_data, classnames, args, device):
    results = {}
    for split_name, (features, labels) in split_data.items():
        results[split_name] = evaluate_model(
            model, features, labels, classnames, args, device
        )
    return results


def train_adapter(
    model,
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
    classnames,
    args,
    device,
):
    active_indices = protocol_class_indices(
        args.protocol, args.total_classes, args.base_classes
    )
    if args.protocol == "base5":
        sample_mask = train_labels[:, active_indices].sum(dim=1).gt(0)
        train_features = train_features[sample_mask]
        train_labels = train_labels[sample_mask]
    print(
        f"Protocol={args.protocol}, active_classes={active_indices.tolist()}, "
        f"train_samples={len(train_features)}"
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.adapter_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = None
    if args.class_balanced_bce:
        pos_weight = make_pos_weight(train_labels, active_indices).to(device)

    val_test_features = torch.cat([val_features, test_features], dim=0)
    val_test_labels = torch.cat([val_labels, test_labels], dim=0)
    split_data = {
        "val": (val_features, val_labels),
        "test": (test_features, test_labels),
        "val_test": (val_test_features, val_test_labels),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zero_shot = evaluate_splits(model, split_data, classnames, args, device)
    print(
        f"Zero-shot: val_mAP={zero_shot['val']['mAP']:.4f}, "
        f"test_mAP={zero_shot['test']['mAP']:.4f}, "
        f"val+test_mAP={zero_shot['val_test']['mAP']:.4f}"
    )

    best_score = -math.inf
    best_epoch = None
    history = []
    active_list = active_indices.tolist()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        class_loss_sum = 0.0
        identity_loss_sum = 0.0
        sample_count = 0
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            logits, adapted, original = model(feature_batch)
            class_loss = F.binary_cross_entropy_with_logits(
                logits[:, active_list],
                label_batch[:, active_list],
                pos_weight=pos_weight,
            )
            identity_loss = 1 - (adapted * original).sum(dim=-1).mean()
            loss = class_loss + args.identity_weight * identity_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = feature_batch.shape[0]
            sample_count += batch_size
            loss_sum += loss.item() * batch_size
            class_loss_sum += class_loss.item() * batch_size
            identity_loss_sum += identity_loss.item() * batch_size

        val_metrics = evaluate_model(
            model, val_features, val_labels, classnames, args, device
        )
        selection_score = (
            val_metrics["mAP"]
            if args.protocol == "all26"
            else val_metrics["base_seen_mAP"]
        )
        row = {
            "epoch": epoch,
            "loss": loss_sum / sample_count,
            "classification_loss": class_loss_sum / sample_count,
            "identity_loss": identity_loss_sum / sample_count,
            "logit_scale": model.logit_scale.exp().clamp(max=100).item(),
            "val": compact_metrics(val_metrics),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: loss={row['loss']:.6f}, "
            f"val_mAP={val_metrics['mAP']:.4f}, "
            f"val_base_seen={val_metrics['base_seen_mAP']:.4f}, "
            f"val_novel={val_metrics['novel_mAP']:.4f}"
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": best_score,
                    "selection_split": "val",
                    "protocol": args.protocol,
                    "classnames": classnames,
                    "args": vars(args),
                },
                output_dir / "best_adapter.pth",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs - 1,
            "protocol": args.protocol,
            "classnames": classnames,
            "args": vars(args),
        },
        output_dir / "last_adapter.pth",
    )
    best_checkpoint = torch.load(
        output_dir / "best_adapter.pth", map_location=device
    )
    model.load_state_dict(best_checkpoint["model"])
    best = evaluate_splits(model, split_data, classnames, args, device)
    summary = {
        "protocol": args.protocol,
        "selection_split": "val",
        "selection_metric": (
            "mAP" if args.protocol == "all26" else "base_seen_mAP"
        ),
        "best_epoch": best_epoch,
        "zero_shot": zero_shot,
        "best": best,
        "history": history,
        "args": vars(args),
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.total_classes != 26:
        raise ValueError("The EMOTIC prototype experiment expects 26 classes")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.feature_batch_size <= 0 or args.adapter_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    transform = build_clip_transform(args.input_size)
    train_dataset = EMOTIC(
        args.datadir,
        train=True,
        transform=transform,
        input_mode=args.emotic_input_mode,
    )
    val_dataset = EMOTIC(
        args.datadir,
        train=False,
        eval_splits=("val",),
        transform=transform,
        input_mode=args.emotic_input_mode,
    )
    test_dataset = EMOTIC(
        args.datadir,
        train=False,
        eval_splits=("test",),
        transform=transform,
        input_mode=args.emotic_input_mode,
    )
    classnames = list(train_dataset.classes)
    if classnames != list(val_dataset.classes) or classnames != list(
        test_dataset.classes
    ):
        raise RuntimeError("Train, validation, and test class orders do not match")
    if len(classnames) != args.total_classes:
        raise RuntimeError(
            f"Expected {args.total_classes} classes, found {len(classnames)}"
        )

    clip_model = load_frozen_clip(args.clip_model_path, device)
    positive_prototypes = encode_template_ensemble(
        clip_model, classnames, POSITIVE_TEMPLATES, device
    )
    negative_prototypes = encode_template_ensemble(
        clip_model, classnames, NEGATIVE_TEMPLATES, device
    )
    train_features, train_labels = load_or_extract_split(
        clip_model, train_dataset, "train", args, device
    )
    val_features, val_labels = load_or_extract_split(
        clip_model, val_dataset, "val", args, device
    )
    test_features, test_labels = load_or_extract_split(
        clip_model, test_dataset, "test", args, device
    )
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = ResidualPrototypeAdapter(
        positive_prototypes,
        negative_prototypes,
        bottleneck_dim=args.adapter_dim,
        residual_scale=args.residual_scale,
        initial_logit_scale=args.initial_logit_scale,
    ).to(device)
    summary = train_adapter(
        model,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        classnames,
        args,
        device,
    )
    print(
        f"Best epoch={summary['best_epoch']}, "
        f"test_mAP={summary['best']['test']['mAP']:.4f}, "
        f"val+test_mAP={summary['best']['val_test']['mAP']:.4f}, "
        f"test_novel={summary['best']['test']['novel_mAP']:.4f}"
    )


if __name__ == "__main__":
    main()
