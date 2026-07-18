"""Train DDP's shared Adapter through a training-only prompt-free CLIP route.

The script intentionally constructs one DDP model and reuses its frozen CLIP
image/text towers.  It does not load a second vanilla CLIP model and does not
read an external Prototype Adapter checkpoint.  Only the learned shared
Adapter weights are retained for prompted-CLS-to-pooled feature correction.
"""

import argparse
import fcntl
import json
import math
import os
from html import escape
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from clip import clip
from ddp_internal_adapter import (
    PromptFreePrototypeObjective,
)
from evaluation_metrics import mAP
from prototype_adapter import (
    NEGATIVE_TEMPLATES,
    POSITIVE_TEMPLATES,
    protocol_class_indices,
)
from prototype_fewshot import (
    masked_bce_with_logits,
    masked_pos_weight,
    sample_multilabel_kshot,
)
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import load_frozen_ddp, set_seed


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_clip_transform(input_size):
    """Match the official OpenAI CLIP ViT-B/16 preprocessing exactly."""
    return transforms.Compose(
        [
            transforms.Resize(
                input_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def _subset_map(labels, probabilities, indices):
    if not indices or labels.shape[0] == 0:
        return None
    score, _ = mAP(
        labels[:, indices].numpy(), probabilities[:, indices].numpy()
    )
    return float(score)


def _binary_metrics(labels, probabilities, threshold):
    labels = labels.bool()
    predictions = probabilities.ge(threshold)
    tp = (predictions & labels).sum(dim=0).float()
    fp = (predictions & ~labels).sum(dim=0).float()
    fn = (~predictions & labels).sum(dim=0).float()
    class_precision = tp / (tp + fp).clamp_min(1.0)
    class_recall = tp / (tp + fn).clamp_min(1.0)
    class_f1 = (
        2
        * class_precision
        * class_recall
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
        "base_mAP": _subset_map(labels, probabilities, base_indices),
        "base_seen_mAP": _subset_map(
            labels[base_seen_mask],
            probabilities[base_seen_mask],
            base_indices,
        ),
        "novel_mAP": _subset_map(labels, probabilities, novel_indices),
        "threshold": threshold,
        **_binary_metrics(labels, probabilities, threshold),
        "per_class_ap": {
            name: 100 * float(per_class_ap[index])
            for index, name in enumerate(classnames)
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train an EMOTIC Prototype Adapter from DDP's training-only "
            "prompt-free global CLIP branch"
        )
    )
    parser.add_argument(
        "--ddp_checkpoint",
        default=(
            "./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth"
        ),
    )
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--feature_cache_dir",
        default="./output/emotic_ddp_prompt_free_auxiliary_feature_cache",
    )
    parser.add_argument(
        "--protocol", choices=("base5", "all26"), default="base5"
    )
    parser.add_argument("--base_classes", type=int, default=5)
    parser.add_argument("--total_classes", type=int, default=26)
    parser.add_argument(
        "--emotic_input_mode", choices=("full", "person_crop"), default="full"
    )
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--adapter_batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--identity_weight", type=float, default=0.1)
    parser.add_argument("--initial_logit_scale", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--class_balanced_bce", action="store_true")
    parser.add_argument("--shots_per_class", type=int, default=None)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def encode_template_ensemble(model, classnames, templates, device):
    """Build fixed prototypes with DDP's own frozen CLIP text tower."""
    template_features = []
    with torch.no_grad():
        for template in templates:
            texts = [template.format(name.lower()) for name in classnames]
            tokens = clip.tokenize(texts).to(device)
            features = model.encode_prompt_free_text(tokens)
            template_features.append(features)
    return F.normalize(
        torch.stack(template_features).mean(dim=0), dim=-1
    ).cpu()


def extract_prompt_free_features(model, dataset, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    feature_batches = []
    label_batches = []
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = model.encode_prompt_free_image(images)
            feature_batches.append(features.cpu())
            label_batches.append(labels.float().cpu())
            if batch_id % 50 == 0:
                print(
                    "Prompt-free feature extraction: "
                    f"batch {batch_id + 1}/{len(loader)}",
                    flush=True,
                )
    return torch.cat(feature_batches), torch.cat(label_batches)


def cache_metadata(args, classnames, split):
    return {
        "split": split,
        "route": "ddp_owned_prompt_free_global_cls",
        "visual_prompts": None,
        "preprocessing": "openai_clip_resize_center_crop_normalize",
        "input_mode": args.emotic_input_mode,
        "input_size": args.input_size,
        "clip_model": os.path.abspath(args.clip_model_path),
        "ddp_checkpoint": os.path.abspath(args.ddp_checkpoint),
        "classnames": list(classnames),
    }


def load_or_extract_split(model, dataset, split, args, device):
    cache_dir = Path(args.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{split}_{args.emotic_input_mode}_{args.input_size}_vitb16.pt"
    )
    expected = cache_metadata(args, dataset.classes, split)
    lock_path = Path(str(cache_path) + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file() and not args.force_recache:
            payload = torch.load(cache_path, map_location="cpu")
            if payload.get("metadata") != expected:
                raise RuntimeError(
                    f"Feature cache metadata mismatch at {cache_path}; "
                    "use --force_recache"
                )
            print(f"Loaded prompt-free feature cache: {cache_path}")
            return payload["features"].float(), payload["labels"].float()

        features, labels = extract_prompt_free_features(
            model, dataset, args, device
        )
        temporary = Path(str(cache_path) + f".tmp.{os.getpid()}")
        torch.save(
            {"metadata": expected, "features": features, "labels": labels},
            temporary,
        )
        os.replace(temporary, cache_path)
        print(f"Saved prompt-free feature cache: {cache_path}")
        return features, labels


def selection_map(objective, features, labels, active_indices, args, device):
    objective.eval()
    active = active_indices.tolist()
    logits = []
    loader = DataLoader(
        TensorDataset(features),
        batch_size=args.adapter_batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (feature_batch,) in loader:
            batch_logits, _, _ = objective(feature_batch.to(device))
            logits.append(batch_logits[:, active].cpu())
    active_labels = labels[:, active]
    seen_mask = active_labels.sum(dim=1).gt(0)
    score, _ = mAP(
        active_labels[seen_mask].numpy(),
        torch.sigmoid(torch.cat(logits)[seen_mask]).numpy(),
    )
    return float(score)


def evaluate_objective(objective, features, labels, classnames, args, device):
    objective.eval()
    logits = []
    loader = DataLoader(
        TensorDataset(features),
        batch_size=args.adapter_batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (feature_batch,) in loader:
            batch_logits, _, _ = objective(feature_batch.to(device))
            logits.append(batch_logits.cpu())
    return evaluate_logits(
        torch.cat(logits),
        labels,
        classnames,
        args.base_classes,
        args.threshold,
    )


def write_training_html(path, summary):
    history = summary["history"]
    headers = list(history[0]) if history else []
    header_html = "".join(f"<th>{escape(key)}</th>" for key in headers)
    body_html = "\n".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in headers)
        + "</tr>"
        for row in history
    )
    protocol = summary["protocol"]
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>DDP Prompt-Free Auxiliary Adapter</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:7px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>DDP Prompt-Free Auxiliary Adapter</h1>"
        "<p><b>Training:</b> one frozen DDP-owned CLIP, no visual prompts, "
        "fixed prototype supervision.</p>"
        "<p><b>Inference:</b> auxiliary route discarded; W1/W2 are applied "
        "to prompted CLS for CLS-to-pooled feature correction.</p>"
        f"<pre>{escape(json.dumps(protocol, indent=2, ensure_ascii=False))}</pre>"
        f"<table><tr>{header_html}</tr>{body_html}</table>",
        encoding="utf-8",
    )


def prepare_training_subset(train_features, train_labels, active_indices, args):
    source_indices = torch.arange(len(train_labels), dtype=torch.long)
    if args.protocol == "base5":
        keep = train_labels[:, active_indices].sum(dim=1).gt(0)
        train_features = train_features[keep]
        train_labels = train_labels[keep]
        source_indices = source_indices[keep]

    supervision_mask = torch.ones_like(train_labels, dtype=torch.bool)
    sampling = {
        "mode": "full_data",
        "shots_per_class": None,
        "seed": args.seed,
        "unique_training_samples": int(len(train_features)),
        "selected_train_indices": source_indices.tolist(),
        "sampling": [],
    }
    if args.shots_per_class is not None:
        selected, supervision_mask, rows = sample_multilabel_kshot(
            train_labels,
            active_indices,
            args.shots_per_class,
            args.seed,
        )
        train_features = train_features[selected]
        train_labels = train_labels[selected]
        sampling = {
            "mode": "fewshot",
            "shots_per_class": args.shots_per_class,
            "seed": args.seed,
            "unique_training_samples": int(selected.numel()),
            "selected_train_indices": source_indices[selected].tolist(),
            "sampling": rows,
        }
    return train_features, train_labels, supervision_mask, sampling


def train_auxiliary(
    objective,
    train_features,
    train_labels,
    val_features,
    val_labels,
    active_indices,
    args,
    device,
):
    active = active_indices.tolist()
    train_features, train_labels, supervision_mask, sampling = (
        prepare_training_subset(
            train_features, train_labels, active_indices, args
        )
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels, supervision_mask),
        batch_size=args.adapter_batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        objective.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = None
    if args.class_balanced_bce:
        pos_weight = masked_pos_weight(
            train_labels, supervision_mask, active_indices
        ).to(device)

    initial_adapter = {
        key: value.detach().cpu().clone()
        for key, value in objective.adapter.state_dict().items()
    }
    initial_logit_scale = objective.logit_scale.detach().cpu().clone()
    zero_score = selection_map(
        objective,
        val_features,
        val_labels,
        active_indices,
        args,
        device,
    )
    best_score = -math.inf
    best_epoch = None
    best_adapter = None
    best_logit_scale = None
    history = []

    for epoch in range(args.epochs):
        objective.train()
        totals = {"loss": 0.0, "class_loss": 0.0, "identity_loss": 0.0}
        sample_count = 0
        for feature_batch, label_batch, mask_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            mask_batch = mask_batch.to(device)
            logits, adapted, original = objective(feature_batch)
            class_loss = masked_bce_with_logits(
                logits[:, active],
                label_batch[:, active],
                mask_batch[:, active],
                pos_weight=pos_weight,
            )
            identity_loss = 1.0 - (adapted * original).sum(dim=-1).mean()
            loss = class_loss + args.identity_weight * identity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = feature_batch.shape[0]
            sample_count += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["class_loss"] += class_loss.item() * batch_size
            totals["identity_loss"] += identity_loss.item() * batch_size

        score = selection_map(
            objective,
            val_features,
            val_labels,
            active_indices,
            args,
            device,
        )
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / sample_count,
            "classification_loss": totals["class_loss"] / sample_count,
            "identity_loss": totals["identity_loss"] / sample_count,
            "logit_scale": float(
                objective.logit_scale.exp().clamp(max=100.0)
            ),
            "val_selection_mAP": score,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: loss={row['loss']:.6f}, "
            f"val_selection_mAP={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_adapter = {
                key: value.detach().cpu().clone()
                for key, value in objective.adapter.state_dict().items()
            }
            best_logit_scale = objective.logit_scale.detach().cpu().clone()

    objective.adapter.load_state_dict(best_adapter)
    objective.logit_scale.data.copy_(best_logit_scale.to(device))
    return {
        "best_epoch": best_epoch,
        "zero_selection_mAP": zero_score,
        "best_selection_mAP": best_score,
        "history": history,
        "sampling": sampling,
        "best_adapter": best_adapter,
        "best_logit_scale": best_logit_scale,
        "initial_adapter": initial_adapter,
        "initial_logit_scale": initial_logit_scale,
    }


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.shots_per_class is not None and args.shots_per_class <= 0:
        raise ValueError("shots_per_class must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ddp_model, ddp_checkpoint = load_frozen_ddp(
        args.ddp_checkpoint, args.clip_model_path, device
    )
    transform = build_clip_transform(args.input_size)
    train_dataset = EMOTIC(
        args.data_root,
        train=True,
        transform=transform,
        input_mode=args.emotic_input_mode,
    )
    val_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=transform,
        input_mode=args.emotic_input_mode,
    )
    test_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("test",),
        transform=transform,
        input_mode=args.emotic_input_mode,
    )
    classnames = list(train_dataset.classes)
    if classnames != list(ddp_checkpoint["classnames"]):
        raise RuntimeError("DDP checkpoint and EMOTIC class orders differ")
    if len(classnames) != args.total_classes:
        raise RuntimeError(
            f"Expected {args.total_classes} classes, found {len(classnames)}"
        )

    positive_prototypes = encode_template_ensemble(
        ddp_model, classnames, POSITIVE_TEMPLATES, device
    )
    negative_prototypes = encode_template_ensemble(
        ddp_model, classnames, NEGATIVE_TEMPLATES, device
    )
    train_features, train_labels = load_or_extract_split(
        ddp_model, train_dataset, "train", args, device
    )
    val_features, val_labels = load_or_extract_split(
        ddp_model, val_dataset, "val", args, device
    )
    test_features, test_labels = load_or_extract_split(
        ddp_model, test_dataset, "test", args, device
    )

    # The Adapter is owned by the same DDP module as the prompt-free encoder
    # and the two prompted inference routes.  No parallel model-level Adapter
    # instance is constructed.
    adapter = ddp_model.enable_feature_adapter(
        bottleneck_dim=args.adapter_dim,
        residual_scale=args.residual_scale,
        correction_mode="feature_correction",
    )
    if adapter.feature_dim != int(train_features.shape[1]):
        raise RuntimeError(
            "Prompt-free global CLS dimension does not match DDP's shared "
            "Adapter dimension"
        )
    objective = PromptFreePrototypeObjective(
        adapter,
        positive_prototypes.to(device),
        negative_prototypes.to(device),
        initial_logit_scale=args.initial_logit_scale,
    ).to(device)
    active_indices = protocol_class_indices(
        args.protocol, args.total_classes, args.base_classes
    )
    training = train_auxiliary(
        objective,
        train_features,
        train_labels,
        val_features,
        val_labels,
        active_indices,
        args,
        device,
    )

    best_metrics = {
        "val": evaluate_objective(
            objective,
            val_features,
            val_labels,
            classnames,
            args,
            device,
        ),
        "test": evaluate_objective(
            objective,
            test_features,
            test_labels,
            classnames,
            args,
            device,
        ),
    }
    objective.adapter.load_state_dict(training["initial_adapter"])
    objective.logit_scale.data.copy_(
        training["initial_logit_scale"].to(device)
    )
    zero_metrics = {
        "val": evaluate_objective(
            objective,
            val_features,
            val_labels,
            classnames,
            args,
            device,
        ),
        "test": evaluate_objective(
            objective,
            test_features,
            test_labels,
            classnames,
            args,
            device,
        ),
    }

    checkpoint_payload = {
        # Kept compatible with the existing transfer/evaluation loaders.
        "model": training["best_adapter"],
        "auxiliary_logit_scale": training["best_logit_scale"],
        "positive_prototypes": positive_prototypes,
        "negative_prototypes": negative_prototypes,
        "epoch": training["best_epoch"],
        "selection_score": training["best_selection_mAP"],
        "selection_split": "val",
        "protocol": args.protocol,
        "classnames": classnames,
        "args": vars(args),
        "sampling": training["sampling"],
        "training_source": "ddp_owned_prompt_free_auxiliary_branch",
        "transfer_target": "prompted_cls_to_ddp_pooled_feature_correction",
    }
    torch.save(checkpoint_payload, output_dir / "best_adapter.pth")
    protocol = {
        "single_clip_instance": True,
        "clip_owner": "DDP",
        "backbone_frozen": True,
        "text_encoder_frozen": True,
        "visual_prompts_during_auxiliary_training": False,
        "fixed_natural_language_prototypes": True,
        "learned_parameters": ["adapter.W1", "adapter.W2", "logit_scale"],
        "retained_for_inference": ["adapter.W1", "adapter.W2"],
        "auxiliary_branch_retained_for_inference": False,
        "inference_routes": [
            "original_ddp_token_attention_pooled",
            "prompted_cls_shared_adapter_feature_correction",
        ],
        "training_classes": active_indices.tolist(),
        "future_labels_used": args.protocol == "all26",
        "selection_split": "val",
        "test_used_for_selection": False,
    }
    summary = {
        "protocol": protocol,
        "best_epoch": training["best_epoch"],
        "zero_selection_mAP": training["zero_selection_mAP"],
        "best_selection_mAP": training["best_selection_mAP"],
        "selection_gain": (
            training["best_selection_mAP"]
            - training["zero_selection_mAP"]
        ),
        "adapter_parameters": sum(
            parameter.numel() for parameter in adapter.parameters()
        ),
        "auxiliary_only_parameters": 1,
        "sampling": training["sampling"],
        "zero_shot": zero_metrics,
        "best": best_metrics,
        "history": training["history"],
        "args": vars(args),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    with open(
        output_dir / "evaluation_summary.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_training_html(output_dir / "training_history.html", summary)
    print(
        f"Best epoch={training['best_epoch']}, "
        f"standalone test mAP={best_metrics['test']['mAP']:.4f}; "
        f"saved {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
