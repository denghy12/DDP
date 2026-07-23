"""Train one strict task-specific Adapter for the EMOTIC B5-C3 bank.

Only train and validation splits are opened.  Old/future labels remain present
in the offline EMOTIC annotation file but are explicitly masked from the loss.
"""

import argparse
import hashlib
import json
import math
import os
from html import escape
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ddp_internal_adapter import PromptFreePrototypeObjective
from emotic_multilabel_losses import LOSS_NAMES, build_asymmetric_loss
from emotic_task_adapter_bank import (
    CHECKPOINT_SCHEMA_VERSION,
    file_sha256,
    prepare_task_training_subset,
    task_class_indices,
    task_class_range,
    validate_task_adapter_checkpoint,
)
from evaluation_metrics import mAP
from prototype_adapter import NEGATIVE_TEMPLATES, POSITIVE_TEMPLATES
from prototype_fewshot import masked_bce_with_logits, masked_pos_weight
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import load_frozen_ddp, set_seed
from train_emotic_ddp_prompt_free_auxiliary import (
    build_clip_transform,
    encode_template_ensemble,
    load_or_extract_split,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one task-specific prompt-free EMOTIC Adapter"
    )
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--training_mode", choices=("full", "16shot"), required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_adapter_checkpoint")
    parser.add_argument(
        "--ddp_checkpoint",
        default="./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth",
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
        "--emotic_input_mode", choices=("full", "person_crop"), default="full"
    )
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--adapter_batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--training_residual_scale", type=float, default=0.1)
    parser.add_argument("--inference_alpha", type=float, default=0.03)
    parser.add_argument("--identity_weight", type=float, default=0.1)
    parser.add_argument("--initial_logit_scale", type=float, default=10.0)
    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument(
        "--classification_loss",
        choices=LOSS_NAMES,
        default="weighted_bce",
    )
    parser.add_argument("--class_balanced_bce", action="store_true")
    parser.add_argument("--asl_gamma_neg", type=float, default=9.8)
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--bal_weight_power", type=float, default=1.6)
    parser.add_argument("--bal_label_smoothing", type=float, default=0.1)
    parser.add_argument("--bal_smoothing_num_classes", type=int, default=26)
    parser.add_argument(
        "--checkpoint_rule",
        choices=("best_val", "last_epoch"),
        default="best_val",
    )
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def loss_configuration(args):
    configuration = {
        "name": args.classification_loss,
        "class_balanced_bce": bool(args.class_balanced_bce),
        "asl_gamma_neg": (
            float(args.asl_gamma_neg)
            if args.classification_loss != "weighted_bce"
            else None
        ),
        "asl_gamma_pos": (
            float(args.asl_gamma_pos)
            if args.classification_loss != "weighted_bce"
            else None
        ),
        "asl_clip": (
            float(args.asl_clip)
            if args.classification_loss != "weighted_bce"
            else None
        ),
        "bal_weight_power": (
            float(args.bal_weight_power)
            if args.classification_loss.startswith("bal")
            else None
        ),
        "bal_label_smoothing": (
            float(args.bal_label_smoothing)
            if args.classification_loss.startswith("bal")
            else 0.0
        ),
        "bal_smoothing_num_classes": (
            int(args.bal_smoothing_num_classes)
            if args.classification_loss.startswith("bal")
            else None
        ),
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    configuration["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return configuration


def current_validation_map(
    objective,
    features,
    labels,
    active_indices,
    batch_size,
    device,
):
    active = active_indices.tolist()
    current_labels = labels[:, active]
    current_samples = current_labels.sum(dim=1).gt(0)
    if not current_samples.any():
        raise RuntimeError("Current-task validation subset is empty")
    loader = DataLoader(
        TensorDataset(features[current_samples]),
        batch_size=batch_size,
        shuffle=False,
    )
    logits = []
    objective.eval()
    with torch.no_grad():
        for (feature_batch,) in loader:
            batch_logits, _, _ = objective(feature_batch.to(device))
            logits.append(batch_logits.cpu())
    probabilities = torch.sigmoid(torch.cat(logits))
    score, per_class_ap = mAP(
        current_labels[current_samples].numpy(), probabilities.numpy()
    )
    return float(score), [100 * float(value) for value in per_class_ap]


def train_task_adapter(
    objective,
    train_features,
    train_labels,
    val_features,
    val_labels,
    active_indices,
    supervision_mask,
    args,
    device,
):
    active = active_indices.tolist()
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
    active_labels = train_labels[:, active]
    active_mask = supervision_mask[:, active]
    asymmetric_loss = None
    loss_diagnostics = None
    if args.classification_loss == "weighted_bce" and args.class_balanced_bce:
        pos_weight = masked_pos_weight(
            train_labels, supervision_mask, active_indices
        ).to(device)
        positives = (active_labels.gt(0) & active_mask.bool()).sum(dim=0).float()
        negatives = (active_labels.eq(0) & active_mask.bool()).sum(dim=0).float()
        loss_diagnostics = {
            "loss_name": "weighted_bce",
            "visible_positives": positives.tolist(),
            "visible_negatives": negatives.tolist(),
            "pos_weight": pos_weight.cpu().tolist(),
        }
    elif args.classification_loss == "weighted_bce":
        positives = (active_labels.gt(0) & active_mask.bool()).sum(dim=0).float()
        negatives = (active_labels.eq(0) & active_mask.bool()).sum(dim=0).float()
        loss_diagnostics = {
            "loss_name": "weighted_bce",
            "visible_positives": positives.tolist(),
            "visible_negatives": negatives.tolist(),
            "pos_weight": None,
        }
    else:
        asymmetric_loss, loss_diagnostics = build_asymmetric_loss(
            args.classification_loss,
            active_labels,
            active_mask,
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
            bal_weight_power=args.bal_weight_power,
            bal_label_smoothing=args.bal_label_smoothing,
            smoothing_num_classes=args.bal_smoothing_num_classes,
        )
        asymmetric_loss = asymmetric_loss.to(device)

    initial_adapter = {
        key: value.detach().cpu().clone()
        for key, value in objective.adapter.state_dict().items()
    }
    initial_logit_scale = objective.logit_scale.detach().cpu().clone()
    zero_score, zero_per_class = current_validation_map(
        objective,
        val_features,
        val_labels,
        active_indices,
        args.adapter_batch_size,
        device,
    )
    best_score = -math.inf
    best_epoch = None
    best_adapter = None
    best_logit_scale = None
    best_per_class = None
    last_adapter = None
    last_logit_scale = None
    last_score = None
    last_per_class = None
    history = []

    for epoch in range(args.epochs):
        objective.train()
        totals = {"loss": 0.0, "classification_loss": 0.0, "identity_loss": 0.0}
        sample_count = 0
        for feature_batch, label_batch, mask_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch[:, active].to(device)
            mask_batch = mask_batch[:, active].to(device)
            logits, adapted, original = objective(feature_batch)
            if asymmetric_loss is None:
                class_loss = masked_bce_with_logits(
                    logits,
                    label_batch,
                    mask_batch,
                    pos_weight=pos_weight,
                )
            else:
                class_loss = asymmetric_loss(logits, label_batch, mask_batch)
            identity_loss = 1.0 - (adapted * original).sum(dim=-1).mean()
            loss = class_loss + args.identity_weight * identity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = feature_batch.shape[0]
            sample_count += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["classification_loss"] += class_loss.item() * batch_size
            totals["identity_loss"] += identity_loss.item() * batch_size

        selection_score, per_class_ap = current_validation_map(
            objective,
            val_features,
            val_labels,
            active_indices,
            args.adapter_batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / sample_count,
            "classification_loss": totals["classification_loss"] / sample_count,
            "identity_loss": totals["identity_loss"] / sample_count,
            "auxiliary_logit_scale": float(
                objective.logit_scale.exp().clamp(max=100.0)
            ),
            "current_val_mAP": selection_score,
        }
        history.append(row)
        print(
            f"Task {args.task_id} epoch {epoch:03d}: "
            f"loss={row['loss']:.6f}, val_mAP={selection_score:.4f}",
            flush=True,
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_adapter = {
                key: value.detach().cpu().clone()
                for key, value in objective.adapter.state_dict().items()
            }
            best_logit_scale = objective.logit_scale.detach().cpu().clone()
            best_per_class = per_class_ap
        last_adapter = {
            key: value.detach().cpu().clone()
            for key, value in objective.adapter.state_dict().items()
        }
        last_logit_scale = objective.logit_scale.detach().cpu().clone()
        last_score = selection_score
        last_per_class = per_class_ap

    if args.checkpoint_rule == "last_epoch":
        selected_epoch = args.epochs - 1
        selected_adapter = last_adapter
        selected_logit_scale = last_logit_scale
        selected_score = last_score
        selected_per_class = last_per_class
    else:
        selected_epoch = best_epoch
        selected_adapter = best_adapter
        selected_logit_scale = best_logit_scale
        selected_score = best_score
        selected_per_class = best_per_class
    objective.adapter.load_state_dict(selected_adapter)
    objective.logit_scale.data.copy_(selected_logit_scale.to(device))
    return {
        "checkpoint_rule": args.checkpoint_rule,
        "selected_epoch": selected_epoch,
        "selected_adapter": selected_adapter,
        "selected_logit_scale": selected_logit_scale,
        "selected_score": selected_score,
        "selected_per_class": selected_per_class,
        "best_epoch": best_epoch,
        "zero_selection_mAP": zero_score,
        "zero_per_class_ap": zero_per_class,
        "best_selection_mAP": best_score,
        "best_per_class_ap": best_per_class,
        "history": history,
        "best_adapter": best_adapter,
        "best_logit_scale": best_logit_scale,
        "initial_adapter": initial_adapter,
        "initial_logit_scale": initial_logit_scale,
        "loss_diagnostics": loss_diagnostics,
    }


def write_training_html(path, summary):
    rows = summary["history"]
    headers = list(rows[0]) if rows else []
    header_html = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[name]))}</td>" for name in headers)
        + "</tr>"
        for row in rows
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Task Adapter Training</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:7px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>Task {summary['task_adapter']['task_id']} Adapter Training</h1>"
        f"<pre>{escape(json.dumps(summary['protocol'], indent=2, ensure_ascii=False))}</pre>"
        f"<table><tr>{header_html}</tr>{body_html}</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    low, high = task_class_range(args.task_id)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.training_mode == "16shot" and args.shots_per_class != 16:
        raise ValueError("The locked few-shot protocol requires exactly 16 shots")
    if args.task_id == 0 and args.init_adapter_checkpoint:
        raise ValueError("Task 0 must start from the identity Adapter initialization")
    if args.task_id > 0 and not args.init_adapter_checkpoint:
        raise ValueError("Tasks 1-7 require the matching task0 Adapter checkpoint")
    if args.classification_loss != "weighted_bce" and args.class_balanced_bce:
        raise ValueError(
            "--class_balanced_bce cannot be combined with ASL/BAL because it "
            "would double-weight positive classes"
        )
    if args.classification_loss == "weighted_bce" and not args.class_balanced_bce:
        raise ValueError(
            "The locked weighted_bce baseline requires --class_balanced_bce"
        )
    locked_loss_config = loss_configuration(args)

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ddp_model, ddp_checkpoint = load_frozen_ddp(
        args.ddp_checkpoint, args.clip_model_path, device
    )
    checkpoint_classnames = list(ddp_checkpoint["classnames"])
    transform = build_clip_transform(args.input_size)
    train_dataset = EMOTIC(
        args.data_root,
        train=True,
        transform=transform,
        input_mode=args.emotic_input_mode,
        class_names=checkpoint_classnames,
    )
    val_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=transform,
        input_mode=args.emotic_input_mode,
        class_names=checkpoint_classnames,
    )
    classnames = list(train_dataset.classes)
    if classnames != checkpoint_classnames:
        raise RuntimeError("DDP checkpoint and EMOTIC class orders differ")

    current_classnames = classnames[low:high]
    positive_prototypes = encode_template_ensemble(
        ddp_model, current_classnames, POSITIVE_TEMPLATES, device
    )
    negative_prototypes = encode_template_ensemble(
        ddp_model, current_classnames, NEGATIVE_TEMPLATES, device
    )
    train_features, train_labels = load_or_extract_split(
        ddp_model, train_dataset, "train", args, device
    )
    val_features, val_labels = load_or_extract_split(
        ddp_model, val_dataset, "val", args, device
    )
    active_indices = task_class_indices(args.task_id)
    selected, supervision_mask, sampling = prepare_task_training_subset(
        train_labels,
        args.task_id,
        args.training_mode,
        args.seed,
        shots_per_class=args.shots_per_class,
    )
    selected_features = train_features[selected]
    selected_labels = train_labels[selected]

    adapter = ddp_model.enable_feature_adapter(
        bottleneck_dim=args.adapter_dim,
        residual_scale=args.training_residual_scale,
        correction_mode="linear_residual",
    )
    initialization = {"kind": "identity", "source": None, "sha256": None}
    if args.init_adapter_checkpoint:
        init_path = Path(args.init_adapter_checkpoint)
        init_checkpoint = torch.load(init_path, map_location="cpu")
        validate_task_adapter_checkpoint(
            init_checkpoint,
            task_id=0,
            training_mode=args.training_mode,
            seed=args.seed,
            classnames=classnames,
            classification_loss=args.classification_loss,
            loss_config_sha256=locked_loss_config["sha256"],
            checkpoint_rule=args.checkpoint_rule,
        )
        adapter.load_state_dict(init_checkpoint["model"], strict=True)
        initialization = {
            "kind": "task0_anchor",
            "source": os.path.abspath(init_path),
            "sha256": file_sha256(init_path),
        }

    objective = PromptFreePrototypeObjective(
        adapter,
        positive_prototypes.to(device),
        negative_prototypes.to(device),
        initial_logit_scale=args.initial_logit_scale,
    ).to(device)
    training = train_task_adapter(
        objective,
        selected_features,
        selected_labels,
        val_features,
        val_labels,
        active_indices,
        supervision_mask,
        args,
        device,
    )

    task_adapter = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_id": args.task_id,
        "class_range": [low, high],
        "class_ids": active_indices.tolist(),
        "class_names": current_classnames,
        "training_mode": args.training_mode,
        "seed": args.seed,
        "initialization": initialization,
        "classification_loss": args.classification_loss,
        "loss_config": locked_loss_config,
        "checkpoint_rule": args.checkpoint_rule,
    }
    checkpoint_payload = {
        "model": training["selected_adapter"],
        "auxiliary_logit_scale": training["selected_logit_scale"],
        "epoch": training["selected_epoch"],
        "reporting_val_score": training["selected_score"],
        "checkpoint_rule": args.checkpoint_rule,
        "selection_split": (
            "none_fixed_last_epoch"
            if args.checkpoint_rule == "last_epoch"
            else "val_current_task"
        ),
        "classnames": classnames,
        "task_adapter": task_adapter,
        "args": {
            **vars(args),
            "feature_dim": int(train_features.shape[1]),
            "adapter_dim": args.adapter_dim,
            "residual_scale": args.training_residual_scale,
            "inference_alpha": args.inference_alpha,
            "correction_mode": "linear_residual",
            "formula": "feature_difference",
        },
        "sampling": sampling,
        "training_source": "ddp_owned_prompt_free_task_adapter_branch",
        "transfer_target": "task_routed_prompted_cls_feature_difference",
        "loss_diagnostics": training["loss_diagnostics"],
    }
    checkpoint_name = (
        "last_adapter.pth"
        if args.checkpoint_rule == "last_epoch"
        else "best_adapter.pth"
    )
    checkpoint_path = output_dir / checkpoint_name
    torch.save(checkpoint_payload, checkpoint_path)
    protocol = {
        "dataset": "EMOTIC",
        "incremental_protocol": "B5-C3",
        "task_id": args.task_id,
        "training_classes": active_indices.tolist(),
        "current_labels_supervised": True,
        "old_labels_supervised": False,
        "future_labels_supervised": False,
        "future_class_prototypes_encoded": False,
        "test_dataset_constructed": False,
        "test_examples_iterated": False,
        "test_labels_used": False,
        "test_used_for_selection": False,
        "checkpoint_rule": args.checkpoint_rule,
        "selection_split": (
            "none; validation is reporting-only"
            if args.checkpoint_rule == "last_epoch"
            else "current-task validation"
        ),
        "validation_used_for_checkpoint_selection": (
            args.checkpoint_rule == "best_val"
        ),
        "single_frozen_ddp_owned_clip": True,
        "visual_prompts_during_auxiliary_training": False,
        "adapter_initialization": initialization,
        "training_residual_scale": args.training_residual_scale,
        "inference_alpha_locked": args.inference_alpha,
        "inference_formula_locked": "feature_difference",
        "class_specific_gate": False,
        "task_specific_alpha": False,
        "classification_loss": args.classification_loss,
        "loss_config": locked_loss_config,
    }
    summary = {
        "protocol": protocol,
        "task_adapter": task_adapter,
        "best_epoch": training["best_epoch"],
        "selected_epoch": training["selected_epoch"],
        "checkpoint_rule": training["checkpoint_rule"],
        "zero_selection_mAP": training["zero_selection_mAP"],
        "best_selection_mAP": training["best_selection_mAP"],
        "selection_gain": (
            training["best_selection_mAP"] - training["zero_selection_mAP"]
        ),
        "zero_per_class_ap": dict(
            zip(current_classnames, training["zero_per_class_ap"])
        ),
        "best_per_class_ap": dict(
            zip(current_classnames, training["best_per_class_ap"])
        ),
        "selected_per_class_ap": dict(
            zip(current_classnames, training["selected_per_class"])
        ),
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "sampling": sampling,
        "loss_diagnostics": training["loss_diagnostics"],
        "history": training["history"],
        "checkpoint": {
            "path": os.path.abspath(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "args": vars(args),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    loss_diagnostics = {
        **training["loss_diagnostics"],
        "class_names": current_classnames,
        "loss_config": locked_loss_config,
        "training_mode": args.training_mode,
        "task_id": args.task_id,
        "seed": args.seed,
        "source_split": "train",
        "old_and_future_entries_included": False,
    }
    with open(
        output_dir / "loss_diagnostics.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(loss_diagnostics, fp, indent=2, ensure_ascii=False)
    distribution_rows = []
    visible_positives = training["loss_diagnostics"]["visible_positives"]
    visible_negatives = training["loss_diagnostics"]["visible_negatives"]
    class_weights = training["loss_diagnostics"].get("class_weights")
    pos_weights = training["loss_diagnostics"].get("pos_weight")
    for index, name in enumerate(current_classnames):
        distribution_rows.append(
            {
                "class_id": int(active_indices[index]),
                "class_name": name,
                "visible_positives": visible_positives[index],
                "visible_negatives": visible_negatives[index],
                "bal_class_weight": (
                    None if class_weights is None else class_weights[index]
                ),
                "bce_pos_weight": (
                    None if pos_weights is None else pos_weights[index]
                ),
            }
        )
    with open(
        output_dir / "class_distribution.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(
            {
                "task_id": args.task_id,
                "training_mode": args.training_mode,
                "classification_loss": args.classification_loss,
                "train_only": True,
                "classes": distribution_rows,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    write_training_html(output_dir / "training_history.html", summary)
    print(
        json.dumps(
            {
                "task": args.task_id,
                "classes": current_classnames,
                "training_mode": args.training_mode,
                "best_epoch": training["best_epoch"],
                "selected_epoch": training["selected_epoch"],
                "checkpoint_rule": args.checkpoint_rule,
                "classification_loss": args.classification_loss,
                "best_val_mAP": training["best_selection_mAP"],
                "reporting_val_mAP": training["selected_score"],
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
