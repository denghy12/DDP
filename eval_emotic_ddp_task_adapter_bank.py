"""Strict all-task evaluation of an EMOTIC task-routed Adapter Bank."""

import argparse
import json
import os
from html import escape
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from emotic_task_adapter_bank import (
    TASK_CLASS_RANGES,
    TASK_SEEN_CLASSES,
    TaskRoutedAdapterBank,
    file_sha256,
)
from eval_emotic_ddp_internal_adapter import (
    baseline_map,
    build_model,
    forgetting,
    task_feature_cache,
)
from eval_emotic_prototype_fusion import score_metrics, select_threshold
from eval_emotic_threshold_sweep import rebuild_text_feature_cache, temperature_for_task
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed-alpha class-routed EMOTIC Adapter Bank"
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--bank_manifest", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--cache_dir", default="./output/emotic_ddp_cls_feature_correction_cache"
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
    parser.add_argument(
        "--fixed_threshold",
        type=float,
        default=None,
        help="Use one locked decision threshold and do not inspect validation labels.",
    )
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=2.0)
    parser.add_argument("--t_gamma", type=float, default=0.7)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def predict_from_bank(bank, payload, seen_classes, args):
    features = payload["path_features"]
    base_logits = payload["base_path_logits"]
    loader = DataLoader(
        TensorDataset(features, base_logits),
        batch_size=args.adapter_batch_size,
        shuffle=False,
    )
    text_features = payload["text_features"].to(args.device).float()
    temperature = temperature_for_task(
        seen_classes, 26, 5, args.t_min, args.t_max, args.t_gamma
    )
    scores = []
    with torch.no_grad():
        for feature_batch, base_batch in loader:
            logits = bank.logits_from_features(
                feature_batch.to(args.device).float(),
                base_batch.to(args.device).float(),
                text_features,
                seen_classes,
                logit_scale=100.0,
            )
            scores.append(
                torch.softmax(logits / temperature, dim=1)[:, 1, :].cpu()
            )
    return torch.cat(scores), temperature


def grouped_metrics(targets, scores, threshold, classnames, low, high):
    if low >= high:
        return None
    return score_metrics(
        targets[:, low:high],
        scores[:, low:high],
        threshold,
        classnames[low:high],
    )


def write_html(path, summary):
    rows = []
    for row in summary["tasks"]:
        old_map = "—" if row["old_test"] is None else f"{row['old_test']['mAP']:.4f}"
        rows.append(
            "<tr>"
            f"<td>{row['task']}</td>"
            f"<td>{row['seen_classes']}</td>"
            f"<td>{row['test']['mAP']:.4f}</td>"
            f"<td>{row['test_mAP_gain']:+.4f}</td>"
            f"<td>{old_map}</td>"
            f"<td>{row['current_test']['mAP']:.4f}</td>"
            f"<td>{row['test']['cF1']:.4f}</td>"
            f"<td>{row['test']['oF1']:.4f}</td>"
            "</tr>"
        )
    aggregate = summary["aggregate"]
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC Task Adapter Bank Evaluation</title>"
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
    args.feature_source = "cls"
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
    if manifest.get("inference_formula") != "feature_difference":
        raise ValueError("This evaluator only accepts Feature Difference banks")
    if float(manifest.get("inference_alpha", -1)) != 0.03:
        raise ValueError("The locked experiment requires inference_alpha=0.03")
    if manifest.get("routing_mode", "class_introduction_task") != "class_introduction_task":
        raise ValueError("Only class-introduction task routing is supported")
    if args.fixed_threshold is not None and not 0 < args.fixed_threshold < 1:
        raise ValueError("fixed_threshold must be in (0, 1)")
    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    classnames = list(first_checkpoint["classnames"])
    if classnames != list(manifest.get("classnames", [])):
        raise ValueError("Adapter Bank and DDP class orders differ")
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
    rows = []

    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        model.feature_adapter = None
        model.feature_adapter_bank = None
        ddp_checkpoint = torch.load(checkpoint_paths[task_id], map_location="cpu")
        model.load_state_dict(ddp_checkpoint["model"], strict=True)
        model.text_feature_cache.clear()
        rebuild_text_feature_cache(model, seen_classes)
        model.eval()
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
        bank = TaskRoutedAdapterBank.from_manifest(
            manifest_path,
            args.device,
            max_task=task_id,
            classnames=classnames,
        )
        val_scores, temperature = predict_from_bank(
            bank, val_cache, seen_classes, args
        )
        test_scores, _ = predict_from_bank(bank, test_cache, seen_classes, args)
        if args.fixed_threshold is None:
            threshold, threshold_rows = select_threshold(
                val_cache["targets"][:, :seen_classes], val_scores, args
            )
            selected_threshold = threshold["threshold"]
            threshold_role = "selected_on_validation"
        else:
            selected_threshold = float(args.fixed_threshold)
            threshold = {
                "threshold": selected_threshold,
                "source": "fixed_protocol",
            }
            threshold_rows = []
            threshold_role = "fixed_protocol"
        val_metrics = score_metrics(
            val_cache["targets"][:, :seen_classes],
            val_scores,
            selected_threshold,
            classnames[:seen_classes],
        )
        test_metrics = score_metrics(
            test_cache["targets"][:, :seen_classes],
            test_scores,
            selected_threshold,
            classnames[:seen_classes],
        )
        low, high = TASK_CLASS_RANGES[task_id]
        old_val = grouped_metrics(
            val_cache["targets"], val_scores, selected_threshold, classnames, 0, low
        )
        old_test = grouped_metrics(
            test_cache["targets"], test_scores, selected_threshold, classnames, 0, low
        )
        current_val = grouped_metrics(
            val_cache["targets"],
            val_scores,
            selected_threshold,
            classnames,
            low,
            high,
        )
        current_test = grouped_metrics(
            test_cache["targets"],
            test_scores,
            selected_threshold,
            classnames,
            low,
            high,
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
            "selection_split": (
                "none" if args.fixed_threshold is not None else "val"
            ),
            "threshold_role": threshold_role,
            "selected_threshold": threshold,
            "threshold_sweep": threshold_rows,
            "val": val_metrics,
            "test": test_metrics,
            "old_val": old_val,
            "old_test": old_test,
            "current_val": current_val,
            "current_test": current_test,
            "baseline_ddp_test_mAP": ddp_map,
            "test_mAP_gain": test_metrics["mAP"] - ddp_map,
        }
        rows.append(row)
        torch.save(
            {
                "scores": test_scores,
                "targets": test_cache["targets"][:, :seen_classes],
                "temperature": temperature,
                "threshold": selected_threshold,
                "task": task_id,
                "bank_manifest": os.path.abspath(manifest_path),
                "bank_manifest_sha256": file_sha256(manifest_path),
                "adapter_tasks_loaded": list(range(task_id + 1)),
                "correction_mode": "feature_difference",
                "inference_alpha": 0.03,
                "feature_source": "cls",
                "classification_loss": manifest.get("classification_loss"),
                "loss_config": manifest.get("loss_config"),
                "checkpoint_rule": manifest.get("checkpoint_rule"),
                "routing_mode": manifest.get(
                    "routing_mode", "class_introduction_task"
                ),
            },
            output_dir / f"task{task_id}_scores.pt",
        )
        print(
            f"[Task {task_id}] DDP {ddp_map:.4f} -> Bank "
            f"{test_metrics['mAP']:.4f} "
            f"({test_metrics['mAP'] - ddp_map:+.4f})",
            flush=True,
        )
        del bank

    task_maps = [row["test"]["mAP"] for row in rows]
    baseline_maps = [row["baseline_ddp_test_mAP"] for row in rows]
    forgetting_metrics = forgetting(rows, classnames)
    summary = {
        "protocol": {
            "name": "EMOTIC B5-C3 Task-routed Adapter Bank",
            "training_mode": manifest["training_mode"],
            "seed": manifest["seed"],
            "classification_loss": manifest.get("classification_loss", "legacy"),
            "loss_config": manifest.get("loss_config"),
            "checkpoint_rule": manifest.get("checkpoint_rule", "best_val"),
            "routing": "class introduction task",
            "feature_source": "prompted_cls",
            "correction_mode": "feature_difference",
            "inference_alpha": 0.03,
            "task_specific_alpha": False,
            "class_specific_gate": False,
            "external_score_fusion": False,
            "decision_threshold": (
                float(args.fixed_threshold)
                if args.fixed_threshold is not None
                else "selected_per_stage_on_validation"
            ),
            "validation_role": (
                "reporting only"
                if args.fixed_threshold is not None
                else "per-stage global decision threshold only"
            ),
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
            "forgetting": forgetting_metrics,
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
