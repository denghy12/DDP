"""Strict test-only evaluation of the EMOTIC Transformer Adapter Bank."""

import argparse
import json
import os
from html import escape
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from emotic_task_adapter_bank import TASK_CLASS_RANGES, TASK_SEEN_CLASSES, file_sha256
from emotic_transformer_adapter_bank import TaskRoutedTransformerAdapterBank
from eval_emotic_ddp_internal_adapter import baseline_map, build_model, forgetting
from eval_emotic_prototype_fusion import score_metrics
from eval_emotic_threshold_sweep import rebuild_text_feature_cache, temperature_for_task
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--bank_manifest", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--baseline_scores_dir",
        default="./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=2.0)
    parser.add_argument("--t_gamma", type=float, default=0.7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def grouped_metrics(targets, scores, threshold, classnames, low, high):
    if low >= high:
        return None
    return score_metrics(
        targets[:, low:high],
        scores[:, low:high],
        threshold,
        classnames[low:high],
    )


def evaluate_task(model, dataset, labels, seen_classes, temperature, args):
    source_indices = torch.nonzero(
        labels[:, :seen_classes].sum(dim=1).gt(0), as_tuple=False
    ).flatten()
    loader = DataLoader(
        Subset(dataset, source_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    scores = []
    model.eval()
    with torch.no_grad():
        for batch_id, (images, _) in enumerate(loader):
            images = images.to(args.device, non_blocking=True).float()
            with torch.cuda.amp.autocast(
                enabled=str(args.device).startswith("cuda")
            ):
                logits = model(
                    images,
                    cls_id=(0, seen_classes),
                    inference=True,
                )
            scores.append(
                torch.softmax(logits.float() / temperature, dim=1)[:, 1, :].cpu()
            )
            if batch_id % 100 == 0:
                print(
                    f"[K={seen_classes}] test batch {batch_id + 1}/{len(loader)}",
                    flush=True,
                )
    return (
        torch.cat(scores),
        labels.index_select(0, source_indices)[:, :seen_classes],
        source_indices,
    )


def write_html(path, summary):
    rows = []
    for row in summary["tasks"]:
        old_map = "—" if row["old_test"] is None else f"{row['old_test']['mAP']:.4f}"
        rows.append(
            "<tr>"
            f"<td>{row['task']}</td><td>{row['seen_classes']}</td>"
            f"<td>{row['test']['mAP']:.4f}</td>"
            f"<td>{row['test_mAP_gain']:+.4f}</td>"
            f"<td>{old_map}</td><td>{row['current_test']['mAP']:.4f}</td>"
            f"<td>{row['test']['cF1']:.4f}</td>"
            f"<td>{row['test']['oF1']:.4f}</td></tr>"
        )
    aggregate = summary["aggregate"]
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Transformer Adapter Bank Evaluation</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        f"<h1>{escape(summary['protocol']['name'])}</h1>"
        f"<pre>{escape(json.dumps(summary['protocol'], indent=2, ensure_ascii=False))}</pre>"
        f"<p>Average mAP: {aggregate['average_mAP']:.4f}; "
        f"Final mAP: {aggregate['final_mAP']:.4f}; "
        f"Final gain: {aggregate['final_mAP_gain']:+.4f}</p>"
        "<table><tr><th>Task</th><th>Seen</th><th>Test mAP</th>"
        "<th>vs DDP</th><th>Old mAP</th><th>Current mAP</th>"
        "<th>cF1</th><th>oF1</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if float(args.fixed_threshold) != 0.5:
        raise ValueError("Locked protocol requires one fixed threshold of 0.5")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    manifest_path = Path(args.bank_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("adapter_location") != "parallel_to_vit_mlp":
        raise ValueError("Manifest is not a Transformer-block Adapter Bank")
    if manifest.get("classification_loss") != "asl":
        raise ValueError("Locked Transformer Bank requires Adapter ASL")
    if manifest.get("checkpoint_rule") != "last_epoch":
        raise ValueError("Locked Transformer Bank requires fixed-last checkpoints")
    if manifest.get("decision_threshold") != 0.5:
        raise ValueError("Manifest decision threshold must be 0.5")

    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    classnames = list(first_checkpoint["classnames"])
    if classnames != list(manifest.get("classnames", [])):
        raise ValueError("Transformer Bank and DDP class orders differ")
    model = build_model(first_checkpoint, args, torch.device(args.device))
    test_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("test",),
        transform=eval_transform(),
        input_mode="full",
        class_names=classnames,
    )
    test_labels = dense_labels(test_dataset)
    rows = []

    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        model.disable_transformer_adapter_bank()
        model.feature_adapter = None
        model.feature_adapter_bank = None
        ddp_checkpoint = torch.load(checkpoint_paths[task_id], map_location="cpu")
        model.load_state_dict(ddp_checkpoint["model"], strict=True)
        model.text_feature_cache.clear()
        rebuild_text_feature_cache(model, seen_classes)
        bank = TaskRoutedTransformerAdapterBank.from_manifest(
            manifest_path,
            args.device,
            max_task=task_id,
            classnames=classnames,
        )
        model.enable_transformer_adapter_bank(bank, freeze=True)
        temperature = temperature_for_task(
            seen_classes, 26, 5, args.t_min, args.t_max, args.t_gamma
        )
        test_scores, test_targets, source_indices = evaluate_task(
            model, test_dataset, test_labels, seen_classes, temperature, args
        )
        test_metrics = score_metrics(
            test_targets,
            test_scores,
            args.fixed_threshold,
            classnames[:seen_classes],
        )
        low, high = TASK_CLASS_RANGES[task_id]
        old_test = grouped_metrics(
            test_targets, test_scores, args.fixed_threshold, classnames, 0, low
        )
        current_test = grouped_metrics(
            test_targets, test_scores, args.fixed_threshold, classnames, low, high
        )
        ddp_map = baseline_map(
            Path(args.baseline_scores_dir) / f"task{task_id}_scores.pt"
        )
        row = {
            "task": task_id,
            "seen_classes": seen_classes,
            "adapter_tasks_loaded": list(range(task_id + 1)),
            "class_range": [low, high],
            "temperature": temperature,
            "selection_split": "none",
            "threshold_role": "fixed_protocol",
            "selected_threshold": args.fixed_threshold,
            "test": test_metrics,
            "old_test": old_test,
            "current_test": current_test,
            "baseline_ddp_test_mAP": ddp_map,
            "test_mAP_gain": test_metrics["mAP"] - ddp_map,
        }
        rows.append(row)
        torch.save(
            {
                "scores": test_scores,
                "targets": test_targets,
                "source_indices": source_indices,
                "temperature": temperature,
                "threshold": args.fixed_threshold,
                "task": task_id,
                "bank_manifest": os.path.abspath(manifest_path),
                "bank_manifest_sha256": file_sha256(manifest_path),
                "adapter_tasks_loaded": list(range(task_id + 1)),
                "adapter_location": "parallel_to_vit_mlp",
                "classification_loss": manifest["classification_loss"],
                "checkpoint_rule": manifest["checkpoint_rule"],
                "routing_mode": manifest["routing_mode"],
            },
            output_dir / f"task{task_id}_scores.pt",
        )
        print(
            f"[Task {task_id}] DDP {ddp_map:.4f} -> Transformer Bank "
            f"{test_metrics['mAP']:.4f} "
            f"({test_metrics['mAP'] - ddp_map:+.4f})",
            flush=True,
        )

    task_maps = [row["test"]["mAP"] for row in rows]
    baseline_maps = [row["baseline_ddp_test_mAP"] for row in rows]
    summary = {
        "protocol": {
            "name": args.name,
            "dataset": "EMOTIC test only",
            "incremental_protocol": "B5-C3",
            "training_mode": manifest["training_mode"],
            "seed": manifest["seed"],
            "ddp_main_loss": "two_way_bce_frozen_checkpoint",
            "adapter_loss": "asl",
            "checkpoint_rule": "fixed_last_epoch",
            "routing": "class introduction task",
            "adapter_location": "parallel_to_vit_mlp_after_ln2",
            "layer_numbers": manifest["architecture"]["layer_numbers"],
            "bottleneck": "768-128-768",
            "decision_threshold": 0.5,
            "validation_used_for_threshold_selection": False,
            "validation_used_for_checkpoint_selection": False,
            "test_used_for_selection": False,
        },
        "inputs": {
            "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
            "bank_manifest": os.path.abspath(manifest_path),
            "bank_manifest_sha256": file_sha256(manifest_path),
            "baseline_scores_dir": os.path.abspath(args.baseline_scores_dir),
        },
        "tasks": rows,
        "aggregate": {
            "average_mAP": float(np.mean(task_maps)),
            "final_mAP": task_maps[-1],
            "baseline_average_mAP": float(np.mean(baseline_maps)),
            "baseline_final_mAP": baseline_maps[-1],
            "average_mAP_gain": float(np.mean(task_maps) - np.mean(baseline_maps)),
            "final_mAP_gain": task_maps[-1] - baseline_maps[-1],
            "final_cF1": rows[-1]["test"]["cF1"],
            "final_oF1": rows[-1]["test"]["oF1"],
            "average_current_mAP": float(
                np.mean([row["current_test"]["mAP"] for row in rows])
            ),
            "forgetting": forgetting(rows, classnames),
        },
        "args": vars(args),
    }
    json_path = output_dir / "evaluation_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(output_dir / "evaluation_summary.html", summary)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
