"""Train one task-routed Transformer Adapter on frozen BCE-trained DDP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from html import escape
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from build_cfg import setup_cfg
from emotic_multilabel_losses import build_asymmetric_loss
from emotic_task_adapter_bank import (
    file_sha256,
    prepare_task_training_subset,
    task_class_indices,
    task_class_range,
)
from emotic_transformer_adapter_bank import (
    P2L_CA_LAYER_INDICES,
    P2L_CA_LAYER_NUMBERS,
    TRANSFORMER_ADAPTER_SCHEMA_VERSION,
    TaskRoutedTransformerAdapterBank,
    TransformerTaskAdapter,
    validate_transformer_adapter_checkpoint,
)
from eval_emotic_threshold_sweep import checkpoint_model_args
from evaluation_metrics import mAP
from models import ddp
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--training_mode", choices=("full", "16shot"), required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ddp_checkpoint", required=True)
    parser.add_argument("--init_task0_adapter")
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument("--asl_gamma_neg", type=float, default=9.8)
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--report_val_every", type=int, default=1)
    parser.add_argument("--emotic_input_mode", default="full", choices=("full",))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.05, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(
                int((256 / 224) * image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


class TaskSupervisionDataset(Dataset):
    def __init__(self, dataset, selected_indices, supervision_mask):
        self.dataset = dataset
        self.selected_indices = [int(index) for index in selected_indices]
        self.supervision_mask = supervision_mask.bool()
        if len(self.selected_indices) != self.supervision_mask.shape[0]:
            raise ValueError("Selected indices and supervision mask differ in length")

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        image, target = self.dataset[self.selected_indices[index]]
        return image, target, self.supervision_mask[index]


def loss_configuration(args):
    config = {
        "name": "asl",
        "gamma_neg": float(args.asl_gamma_neg),
        "gamma_pos": float(args.asl_gamma_pos),
        "clip": float(args.asl_clip),
        "positive_class_weight": False,
        "label_smoothing": 0.0,
        "classification_input": "ddp_positive_minus_negative_margin",
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config


def load_frozen_task_ddp(args, expected_task, device):
    checkpoint = torch.load(args.ddp_checkpoint, map_location="cpu")
    if int(checkpoint.get("task", -1)) != int(expected_task):
        raise ValueError(
            f"Task {expected_task} Adapter requires DDP task{expected_task}, "
            f"found task {checkpoint.get('task')}"
        )
    helper_args = argparse.Namespace(clip_model_path=args.clip_model_path)
    cfg = setup_cfg(checkpoint_model_args(checkpoint, helper_args))
    model = ddp(cfg, checkpoint["classnames"])
    model = model.module if hasattr(model, "module") else model
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def current_task_val_map(model, dataset, task_id, batch_size, workers, device):
    low, high = task_class_range(task_id)
    source_indices = [
        index
        for index, labels in enumerate(dataset.targets)
        if any(low <= int(class_id) < high for class_id in labels)
    ]
    loader = DataLoader(
        Subset(dataset, source_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
    )
    scores = []
    targets = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
                logits = model(
                    images,
                    cls_id=(low, high),
                    inference=False,
                )
            scores.append(torch.softmax(logits.float(), dim=1)[:, 1, :].cpu())
            targets.append(labels[:, low:high].float())
    score, per_class = mAP(torch.cat(targets).numpy(), torch.cat(scores).numpy())
    return float(score), [100.0 * float(value) for value in per_class]


def write_history_csv(path, history):
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def write_training_html(path, summary):
    history = summary["history"]
    headers = list(history[0]) if history else []
    rows = []
    for row in history:
        rows.append(
            "<tr>"
            + "".join(f"<td>{escape(str(row[key]))}</td>" for key in headers)
            + "</tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Transformer Adapter Training</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>Task {summary['transformer_task_adapter']['task_id']} "
        "Transformer Adapter</h1>"
        f"<pre>{escape(json.dumps(summary['protocol'], indent=2, ensure_ascii=False))}</pre>"
        "<table><tr>"
        + "".join(f"<th>{escape(key)}</th>" for key in headers)
        + "</tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.epochs != 20:
        raise ValueError("Locked P2L-CA protocol requires exactly 20 epochs")
    if args.hidden_dim != 768 or args.bottleneck_dim != 128:
        raise ValueError("Locked architecture is 768 -> 128 -> 768")
    if args.training_mode == "16shot" and args.shots_per_class != 16:
        raise ValueError("Locked few-shot protocol requires exactly 16 shots")
    if args.effective_batch_size < args.train_batch_size:
        raise ValueError("effective_batch_size must be >= train_batch_size")
    if args.effective_batch_size % args.train_batch_size:
        raise ValueError("effective_batch_size must divide by train_batch_size")
    if args.task_id == 0 and args.init_task0_adapter:
        raise ValueError("Task 0 starts from zero-up identity initialization")
    if args.task_id > 0 and not args.init_task0_adapter:
        raise ValueError("Tasks 1--7 require the matching Task 0 anchor")

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    low, high = task_class_range(args.task_id)
    active_indices = task_class_indices(args.task_id)
    locked_loss = loss_configuration(args)

    model, ddp_checkpoint = load_frozen_task_ddp(args, args.task_id, device)
    classnames = list(ddp_checkpoint["classnames"])
    train_dataset = EMOTIC(
        args.data_root,
        train=True,
        transform=train_transform(),
        input_mode=args.emotic_input_mode,
        class_names=classnames,
    )
    val_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=eval_transform(),
        input_mode=args.emotic_input_mode,
        class_names=classnames,
    )
    train_labels = dense_labels(train_dataset)
    selected, supervision_mask, sampling = prepare_task_training_subset(
        train_labels,
        args.task_id,
        args.training_mode,
        args.seed,
        shots_per_class=args.shots_per_class,
    )
    selected_labels = train_labels[selected]
    training_dataset = TaskSupervisionDataset(
        train_dataset,
        selected.tolist(),
        supervision_mask,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )

    adapter = TransformerTaskAdapter(
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        layer_indices=P2L_CA_LAYER_INDICES,
    ).to(device)
    initialization = {"kind": "zero_up_identity", "source": None, "sha256": None}
    if args.init_task0_adapter:
        anchor_path = Path(args.init_task0_adapter)
        anchor = torch.load(anchor_path, map_location="cpu")
        validate_transformer_adapter_checkpoint(
            anchor,
            task_id=0,
            training_mode=args.training_mode,
            seed=args.seed,
            classnames=classnames,
            classification_loss="asl",
            loss_config_sha256=locked_loss["sha256"],
        )
        adapter.load_state_dict(anchor["model"], strict=True)
        initialization = {
            "kind": "task0_anchor",
            "source": os.path.abspath(anchor_path),
            "sha256": file_sha256(anchor_path),
        }

    router = TaskRoutedTransformerAdapterBank(
        {args.task_id: adapter},
        classification_loss="asl",
        loss_config_sha256=locked_loss["sha256"],
        require_contiguous=False,
    ).to(device)
    model.enable_transformer_adapter_bank(router, freeze=False)
    model.eval()
    router.train()

    active_labels = selected_labels[:, low:high]
    active_mask = supervision_mask[:, low:high]
    asymmetric_loss, loss_diagnostics = build_asymmetric_loss(
        "asl",
        active_labels,
        active_mask,
        gamma_neg=args.asl_gamma_neg,
        gamma_pos=args.asl_gamma_pos,
        clip=args.asl_clip,
        bal_weight_power=1.6,
        bal_label_smoothing=0.0,
        smoothing_num_classes=26,
    )
    asymmetric_loss = asymmetric_loss.to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    accumulation_steps = args.effective_batch_size // args.train_batch_size

    initial_val_mAP, initial_per_class = current_task_val_map(
        model,
        val_dataset,
        args.task_id,
        args.eval_batch_size,
        args.workers,
        device,
    )
    history = []
    last_val_mAP = initial_val_mAP
    last_per_class = initial_per_class
    for epoch in range(args.epochs):
        model.eval()
        router.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        samples = 0
        pending = 0
        for batch_id, (images, labels, mask) in enumerate(train_loader):
            images = images.to(device, non_blocking=True).float()
            labels = labels[:, low:high].to(device, non_blocking=True)
            mask = mask[:, low:high].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
                path_logits = model(
                    images,
                    cls_id=(low, high),
                    inference=False,
                )
                margins = path_logits[:, 1, :] - path_logits[:, 0, :]
                classification_loss = asymmetric_loss(margins, labels, mask)
                scaled_loss = classification_loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            pending += 1
            if pending == accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            batch_size = images.shape[0]
            total_loss += float(classification_loss.detach()) * batch_size
            samples += batch_size
        if pending:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        should_report = (
            (epoch + 1) % args.report_val_every == 0
            or epoch == args.epochs - 1
        )
        if should_report:
            last_val_mAP, last_per_class = current_task_val_map(
                model,
                val_dataset,
                args.task_id,
                args.eval_batch_size,
                args.workers,
                device,
            )
        row = {
            "epoch": epoch,
            "classification_loss": total_loss / max(samples, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "reporting_val_mAP": last_val_mAP if should_report else None,
        }
        history.append(row)
        print(
            f"Task {args.task_id} epoch {epoch:02d}: "
            f"ASL={row['classification_loss']:.6f}, "
            f"val_mAP={row['reporting_val_mAP']}",
            flush=True,
        )

    architecture = {
        "source": "P2L-CA best reported adapter depth",
        "hidden_dim": args.hidden_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "activation": "ReLU",
        "layer_numbers": list(P2L_CA_LAYER_NUMBERS),
        "layer_indices": list(P2L_CA_LAYER_INDICES),
        "adapter_count": len(P2L_CA_LAYER_INDICES),
        "adapter_location": "parallel_to_vit_mlp_after_ln2",
        "up_projection_initialization": "zeros",
        "residual_scale": 1.0,
    }
    metadata = {
        "schema_version": TRANSFORMER_ADAPTER_SCHEMA_VERSION,
        "task_id": args.task_id,
        "class_range": [low, high],
        "class_ids": active_indices.tolist(),
        "class_names": classnames[low:high],
        "training_mode": args.training_mode,
        "seed": args.seed,
        "classification_loss": "asl",
        "loss_config": locked_loss,
        "checkpoint_rule": "last_epoch",
        "architecture": architecture,
        "initialization": initialization,
        "ddp_checkpoint": {
            "path": os.path.abspath(args.ddp_checkpoint),
            "sha256": file_sha256(args.ddp_checkpoint),
            "task": int(ddp_checkpoint["task"]),
            "main_training_loss": "two_way_bce",
        },
    }
    checkpoint = {
        "model": {
            key: value.detach().cpu().clone()
            for key, value in adapter.state_dict().items()
        },
        "epoch": args.epochs - 1,
        "reporting_val_mAP": last_val_mAP,
        "reporting_per_class_ap": dict(
            zip(classnames[low:high], last_per_class)
        ),
        "classnames": classnames,
        "transformer_task_adapter": metadata,
        "sampling": sampling,
        "args": vars(args),
    }
    checkpoint_path = output_dir / "last_transformer_adapter.pth"
    torch.save(checkpoint, checkpoint_path)

    protocol = {
        "dataset": "EMOTIC",
        "incremental_protocol": "B5-C3",
        "task_id": args.task_id,
        "ddp_main_loss": "two_way_bce_frozen_checkpoint",
        "adapter_loss": "asl",
        "current_labels_supervised": True,
        "old_labels_supervised": False,
        "future_labels_supervised": False,
        "test_dataset_constructed": False,
        "test_labels_used": False,
        "validation_role": "reporting_only",
        "validation_used_for_checkpoint_selection": False,
        "checkpoint_rule": "fixed_last_epoch",
        "epochs": args.epochs,
        "optimizer": "Adam",
        "initial_learning_rate": args.lr,
        "scheduler": "CosineAnnealingLR",
        "physical_batch_size": args.train_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "architecture": architecture,
        "routing": "class_introduction_task",
    }
    summary = {
        "protocol": protocol,
        "transformer_task_adapter": metadata,
        "initial_val_mAP": initial_val_mAP,
        "final_reporting_val_mAP": last_val_mAP,
        "reporting_val_gain": last_val_mAP - initial_val_mAP,
        "adapter_parameters": adapter.parameter_count,
        "sampling": sampling,
        "loss_diagnostics": loss_diagnostics,
        "history": history,
        "checkpoint": {
            "path": os.path.abspath(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "args": vars(args),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_history_csv(output_dir / "training_history.csv", history)
    write_training_html(output_dir / "training_summary.html", summary)
    print(
        json.dumps(
            {
                "task": args.task_id,
                "training_mode": args.training_mode,
                "final_val_mAP": last_val_mAP,
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
