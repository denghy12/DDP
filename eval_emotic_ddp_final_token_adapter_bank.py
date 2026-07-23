"""Evaluate one or more Final-token Adapter Banks in a shared DDP pass.

Multiple seeds share each expensive frozen DDP token extraction.  No test or
validation labels select a scale, gate, threshold, or checkpoint: alpha=0.03,
the last training epoch, and decision threshold 0.5 are fixed in advance.
"""

from __future__ import annotations

import argparse
import json
import os
from html import escape
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from emotic_final_token_adapter import (
    FINAL_TOKEN_FORMULA,
    TaskRoutedFinalTokenAdapterBank,
    ddp_pool_final_tokens,
    file_sha256,
)
from emotic_task_adapter_bank import TASK_CLASS_RANGES, TASK_SEEN_CLASSES
from eval_emotic_ddp_internal_adapter import build_model, forgetting
from eval_emotic_prototype_fusion import score_metrics
from eval_emotic_threshold_sweep import rebuild_text_feature_cache, temperature_for_task
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform


FIXED_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EMOTIC Final-token Adapter Banks"
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--bank_manifests", nargs="+", required=True)
    parser.add_argument("--run_names", nargs="+", required=True)
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument(
        "--baseline_scores_dir",
        default="./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=2.0)
    parser.add_argument("--t_gamma", type=float, default=0.7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def grouped_metrics(targets, scores, classnames, low, high):
    if low >= high:
        return None
    return score_metrics(
        targets[:, low:high],
        scores[:, low:high],
        FIXED_THRESHOLD,
        classnames[low:high],
    )


def fixed_baseline_metrics(path, classnames):
    payload = torch.load(path, map_location="cpu")
    return score_metrics(
        payload["targets"],
        payload["scores"],
        FIXED_THRESHOLD,
        classnames,
    )


def evaluate_split_shared(
    model,
    banks,
    dataset,
    labels,
    seen_classes,
    temperature,
    args,
):
    source_indices = torch.nonzero(
        labels[:, :seen_classes].sum(dim=1).gt(0), as_tuple=False
    ).flatten()
    loader = DataLoader(
        Subset(dataset, source_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    scores = [[] for _ in banks]
    targets = labels[source_indices].float()
    model.eval()
    with torch.no_grad():
        for batch_id, (images, _) in enumerate(loader):
            images = images.to(args.device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                pooled, base_logits, text_features, token_features = (
                    model.extract_path_features(
                        images,
                        cls_id=(0, seen_classes),
                        inference=True,
                        return_token_features=True,
                    )
                )
            del pooled, base_logits
            token_features = token_features.float()
            text_features = text_features.float()
            for bank_id, bank in enumerate(banks):
                with torch.cuda.amp.autocast(
                    enabled=args.device.startswith("cuda")
                ):
                    adapted = bank.adapt_token_features(
                        token_features, seen_classes=seen_classes
                    )
                    _, path_logits, _, _ = ddp_pool_final_tokens(
                        adapted, text_features
                    )
                    logits = path_logits.reshape(
                        path_logits.shape[0], 2, seen_classes
                    )
                scores[bank_id].append(
                    torch.softmax(logits / temperature, dim=1)[:, 1, :].cpu()
                )
                del adapted, path_logits, logits
            if batch_id % 100 == 0:
                print(
                    f"[K={seen_classes}] shared final-token evaluation "
                    f"{batch_id + 1}/{len(loader)}",
                    flush=True,
                )
    return [torch.cat(parts) for parts in scores], targets


def write_html(path: Path, summary: dict):
    rows = []
    for row in summary["tasks"]:
        rows.append(
            "<tr>"
            f"<td>{row['task']}</td><td>{row['seen_classes']}</td>"
            f"<td>{row['baseline_ddp_test_mAP']:.4f}</td>"
            f"<td>{row['test']['mAP']:.4f}</td>"
            f"<td>{row['test_mAP_gain']:+.4f}</td>"
            f"<td>{row['test']['cF1']:.4f}</td>"
            f"<td>{row['test']['oF1']:.4f}</td>"
            "</tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Final-token Adapter Bank</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>{escape(summary['run_name'])}</h1>"
        "<p>197 final tokens → shared Adapter → exact original DDP pooling; "
        "fixed α=0.03 and threshold=0.5.</p>"
        f"<pre>{escape(json.dumps(summary['aggregate'], indent=2, ensure_ascii=False))}</pre>"
        "<table><tr><th>Task</th><th>Seen</th><th>DDP mAP</th>"
        "<th>Final-token mAP</th><th>Gain</th><th>cF1</th><th>oF1</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if len(args.bank_manifests) != len(args.run_names):
        raise ValueError("--bank_manifests and --run_names must have equal length")
    if len(set(args.run_names)) != len(args.run_names):
        raise ValueError("--run_names must be unique")
    checkpoint_paths = [
        Path(args.checkpoint_dir) / f"task{task_id}.pth" for task_id in range(8)
    ]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    for task_id in range(8):
        baseline_path = Path(args.baseline_scores_dir) / f"task{task_id}_scores.pt"
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)

    manifests = []
    for path in args.bank_manifests:
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("inference_formula") != FINAL_TOKEN_FORMULA:
            raise ValueError(f"Not a Final-token bank: {manifest_path}")
        if float(manifest.get("residual_scale", -1)) != 0.03:
            raise ValueError("The locked evaluation requires alpha=0.03")
        if manifest.get("selection") not in {
            "fixed_last_epoch_no_validation_selection",
            "fixed_optimizer_steps_no_validation_selection",
        }:
            raise ValueError("Final-token bank used an unlocked checkpoint rule")
        manifests.append((manifest_path, manifest))

    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    classnames = list(first_checkpoint["classnames"])
    for _, manifest in manifests:
        if list(manifest.get("classnames", [])) != classnames:
            raise ValueError("Final-token bank and DDP class orders differ")
    device = torch.device(args.device)
    model = build_model(first_checkpoint, args, device)
    val_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("val",),
        transform=eval_transform(),
        input_mode="full",
        class_names=classnames,
    )
    test_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("test",),
        transform=eval_transform(),
        input_mode="full",
        class_names=classnames,
    )
    val_labels = dense_labels(val_dataset)
    test_labels = dense_labels(test_dataset)
    rows_by_run = [[] for _ in manifests]

    output_dirs = []
    for run_name in args.run_names:
        output_dir = Path(args.output_root) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append(output_dir)

    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        ddp_checkpoint = torch.load(checkpoint_paths[task_id], map_location="cpu")
        model.feature_adapter = None
        model.feature_adapter_bank = None
        model.final_token_adapter_bank = None
        model.load_state_dict(ddp_checkpoint["model"], strict=True)
        model.text_feature_cache.clear()
        rebuild_text_feature_cache(model, seen_classes)
        model.eval()
        banks = [
            TaskRoutedFinalTokenAdapterBank.from_manifest(
                path,
                device,
                max_task=task_id,
                classnames=classnames,
            )
            for path, _ in manifests
        ]
        temperature = temperature_for_task(
            seen_classes, 26, 5, args.t_min, args.t_max, args.t_gamma
        )
        val_scores_by_run, val_targets = evaluate_split_shared(
            model,
            banks,
            val_dataset,
            val_labels,
            seen_classes,
            temperature,
            args,
        )
        test_scores_by_run, test_targets = evaluate_split_shared(
            model,
            banks,
            test_dataset,
            test_labels,
            seen_classes,
            temperature,
            args,
        )
        ddp_metrics = fixed_baseline_metrics(
            Path(args.baseline_scores_dir) / f"task{task_id}_scores.pt",
            classnames[:seen_classes],
        )
        ddp_map = ddp_metrics["mAP"]
        low, high = TASK_CLASS_RANGES[task_id]
        for run_id, (val_scores, test_scores) in enumerate(
            zip(val_scores_by_run, test_scores_by_run)
        ):
            val_metrics = score_metrics(
                val_targets[:, :seen_classes],
                val_scores,
                FIXED_THRESHOLD,
                classnames[:seen_classes],
            )
            test_metrics = score_metrics(
                test_targets[:, :seen_classes],
                test_scores,
                FIXED_THRESHOLD,
                classnames[:seen_classes],
            )
            row = {
                "task": task_id,
                "seen_classes": seen_classes,
                "adapter_tasks_loaded": list(range(task_id + 1)),
                "class_range": [low, high],
                "temperature": temperature,
                "decision_threshold": FIXED_THRESHOLD,
                "selection_split": None,
                "val": val_metrics,
                "test": test_metrics,
                "old_val": grouped_metrics(
                    val_targets, val_scores, classnames, 0, low
                ),
                "old_test": grouped_metrics(
                    test_targets, test_scores, classnames, 0, low
                ),
                "current_val": grouped_metrics(
                    val_targets, val_scores, classnames, low, high
                ),
                "current_test": grouped_metrics(
                    test_targets, test_scores, classnames, low, high
                ),
                "baseline_ddp_test_mAP": ddp_map,
                "baseline_ddp_test": ddp_metrics,
                "test_mAP_gain": test_metrics["mAP"] - ddp_map,
            }
            rows_by_run[run_id].append(row)
            torch.save(
                {
                    "scores": test_scores,
                    "targets": test_targets[:, :seen_classes],
                    "temperature": temperature,
                    "threshold": FIXED_THRESHOLD,
                    "task": task_id,
                    "bank_manifest": os.path.abspath(
                        args.bank_manifests[run_id]
                    ),
                    "bank_manifest_sha256": file_sha256(
                        args.bank_manifests[run_id]
                    ),
                    "adapter_tasks_loaded": list(range(task_id + 1)),
                    "feature_source": "final_197_tokens",
                    "pooling": "original_ddp_token_attention",
                },
                output_dirs[run_id] / f"task{task_id}_scores.pt",
            )
            print(
                f"[{args.run_names[run_id]} Task {task_id}] DDP "
                f"{ddp_map:.4f} -> {test_metrics['mAP']:.4f} "
                f"({test_metrics['mAP'] - ddp_map:+.4f})",
                flush=True,
            )
        del banks

    for run_id, rows in enumerate(rows_by_run):
        task_maps = [row["test"]["mAP"] for row in rows]
        baseline_maps = [row["baseline_ddp_test_mAP"] for row in rows]
        forgetting_metrics = forgetting(rows, classnames)
        selection = manifests[run_id][1]["selection"]
        training_protocol = manifests[run_id][1].get("training_protocol", {})
        if selection == "fixed_optimizer_steps_no_validation_selection":
            checkpoint_selection = (
                f"fixed {int(training_protocol['optimizer_steps'])} optimizer steps"
            )
        else:
            checkpoint_selection = "fixed last epoch"
        summary = {
            "run_name": args.run_names[run_id],
            "protocol": {
                "name": "EMOTIC B5-C3 Task-routed Final-token Adapter Bank",
                "training_mode": manifests[run_id][1]["training_mode"],
                "seed": manifests[run_id][1]["seed"],
                "routing": "class introduction task",
                "feature_source": "197 final projected prompted ViT tokens",
                "pooling": "exact original DDP token attention",
                "residual_scale": 0.03,
                "decision_threshold": FIXED_THRESHOLD,
                "checkpoint_selection": checkpoint_selection,
                "validation_role": "reporting only",
                "test_used_for_selection": False,
                "task_specific_alpha": False,
                "class_specific_gate": False,
                "external_score_fusion": False,
                "pooling_aware_regularization": training_protocol,
                "adapter_parameter_count_each": manifests[run_id][1][
                    "adapter_parameter_count_each"
                ],
                "adapter_bank_parameter_count": manifests[run_id][1][
                    "adapter_bank_parameter_count"
                ],
            },
            "inputs": {
                "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
                "bank_manifest": os.path.abspath(args.bank_manifests[run_id]),
                "bank_manifest_sha256": file_sha256(
                    args.bank_manifests[run_id]
                ),
                "baseline_scores_dir": os.path.abspath(args.baseline_scores_dir),
            },
            "tasks": rows,
            "aggregate": {
                "average_mAP": float(np.mean(task_maps)),
                "final_mAP": task_maps[-1],
                "baseline_average_mAP": float(np.mean(baseline_maps)),
                "baseline_final_mAP": baseline_maps[-1],
                "baseline_final_cF1": rows[-1]["baseline_ddp_test"]["cF1"],
                "baseline_final_oF1": rows[-1]["baseline_ddp_test"]["oF1"],
                "average_mAP_gain": float(
                    np.mean(task_maps) - np.mean(baseline_maps)
                ),
                "final_mAP_gain": task_maps[-1] - baseline_maps[-1],
                "final_cF1": rows[-1]["test"]["cF1"],
                "final_oF1": rows[-1]["test"]["oF1"],
                "average_current_mAP": float(
                    np.mean([row["current_test"]["mAP"] for row in rows])
                ),
                "forgetting": forgetting_metrics,
            },
            "args": vars(args),
        }
        json_path = output_dirs[run_id] / "evaluation_summary.json"
        json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_html(output_dirs[run_id] / "evaluation_summary.html", summary)
        print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
