"""Unified evaluation of frozen CLS Adapter weights at three DDP locations.

For each image batch, the expensive prompted DDP visual encoder runs once.
The resulting final tokens, prompted CLS features, base path logits, and text
features are then reused by four fixed inference routes:

* the original DDP baseline;
* the source CLS Feature-Difference Adapter Bank;
* the same frozen CLS weights applied point-wise to all final tokens;
* the same frozen CLS weights applied only to final token 0.

No validation or test result selects a checkpoint, scale, gate, or decision
threshold.  The source Bank is fixed before this experiment; alpha=0.03 and
threshold=0.5 are enforced by the loaders and this evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from html import escape
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from emotic_cls_to_final_token_transfer import (
    TRANSFER_RESIDUAL_SCALE,
    TaskRoutedCLSToFinalTokenTransferBank,
)
from emotic_final_token_adapter import (
    ddp_pool_final_tokens,
    final_token_drift_tensors,
)
from emotic_task_adapter_bank import (
    TASK_CLASS_RANGES,
    TASK_SEEN_CLASSES,
    TaskRoutedAdapterBank,
    file_sha256,
)
from eval_emotic_ddp_internal_adapter import build_model, forgetting
from eval_emotic_prototype_fusion import score_metrics
from eval_emotic_threshold_sweep import rebuild_text_feature_cache, temperature_for_task
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform


FIXED_THRESHOLD = 0.5
METHODS = (
    "ddp",
    "old_cls_feature_difference",
    "all_tokens",
    "cls_only",
)
METHOD_LABELS = {
    "ddp": "Original DDP",
    "old_cls_feature_difference": "Original CLS Feature Difference",
    "all_tokens": "CLS-trained Adapter on all 197 final tokens",
    "cls_only": "CLS-trained Adapter on final CLS token only",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen CLS Bank weights before DDP token pooling"
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--bank_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument(
        "--baseline_scores_dir",
        default="./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050",
        help="Existing DDP scores used only for a consistency audit.",
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


def _grouped_metrics(targets, scores, classnames, low, high):
    if low >= high:
        return None
    return score_metrics(
        targets[:, low:high],
        scores[:, low:high],
        FIXED_THRESHOLD,
        classnames[low:high],
    )


def _existing_baseline_metrics(path: Path, classnames):
    payload = torch.load(path, map_location="cpu")
    if payload["targets"].shape[1] != len(classnames):
        raise ValueError(f"Baseline class count mismatch: {path}")
    return score_metrics(
        payload["targets"].float(),
        payload["scores"].float(),
        FIXED_THRESHOLD,
        classnames,
    )


def _probability(path_logits, seen_classes, temperature):
    logits = path_logits.reshape(path_logits.shape[0], 2, seen_classes)
    return torch.softmax(logits / temperature, dim=1)[:, 1, :]


def _new_distribution_accumulator():
    return {
        "sum": 0.0,
        "square_sum": 0.0,
        "count": 0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def _update_distribution(accumulator, values):
    values = values.detach().float()
    if values.numel() == 0:
        return
    accumulator["sum"] += float(values.sum().cpu())
    accumulator["square_sum"] += float(values.square().sum().cpu())
    accumulator["count"] += int(values.numel())
    accumulator["min"] = min(
        accumulator["min"],
        float(values.min().cpu()),
    )
    accumulator["max"] = max(
        accumulator["max"],
        float(values.max().cpu()),
    )


def _new_drift_accumulator():
    return {
        "attention_kl": _new_distribution_accumulator(),
        "pooled_feature_cosine_drift": _new_distribution_accumulator(),
        "path_logit_absolute_drift": _new_distribution_accumulator(),
        "path_logit_signed_drift": _new_distribution_accumulator(),
        "all_token_delta_l2": _new_distribution_accumulator(),
        "cls_token_delta_l2": _new_distribution_accumulator(),
        "patch_token_delta_l2": _new_distribution_accumulator(),
    }


def _update_drift_accumulator(accumulator, drift):
    path_delta = drift["path_logit_delta"]
    token_delta = drift["token_delta_l2"]
    _update_distribution(
        accumulator["attention_kl"],
        drift["attention_kl_original_to_adapted"],
    )
    _update_distribution(
        accumulator["pooled_feature_cosine_drift"],
        drift["pooled_feature_cosine_drift"],
    )
    _update_distribution(
        accumulator["path_logit_absolute_drift"],
        path_delta.abs(),
    )
    _update_distribution(
        accumulator["path_logit_signed_drift"],
        path_delta,
    )
    _update_distribution(accumulator["all_token_delta_l2"], token_delta)
    _update_distribution(
        accumulator["cls_token_delta_l2"],
        token_delta[:, :, 0],
    )
    if token_delta.shape[2] > 1:
        _update_distribution(
            accumulator["patch_token_delta_l2"],
            token_delta[:, :, 1:],
        )


def _finalize_distribution(accumulator):
    count = accumulator["count"]
    if count <= 0:
        return None
    mean = accumulator["sum"] / count
    variance = max(accumulator["square_sum"] / count - mean * mean, 0.0)
    return {
        "mean": mean,
        "rms": (accumulator["square_sum"] / count) ** 0.5,
        "std": variance**0.5,
        "min": accumulator["min"],
        "max": accumulator["max"],
        "count": count,
    }


def _finalize_drift_accumulator(accumulator):
    return {
        key: _finalize_distribution(values)
        for key, values in accumulator.items()
    }


def evaluate_split_shared(
    model,
    cls_bank,
    all_token_bank,
    cls_only_bank,
    dataset,
    labels,
    seen_classes,
    temperature,
    args,
    split,
):
    """Evaluate all four routes while sharing each DDP feature extraction."""

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
    scores = {method: [] for method in METHODS}
    drift_accumulators = {
        "all_tokens": _new_drift_accumulator(),
        "cls_only": _new_drift_accumulator(),
    }
    model.eval()
    with torch.no_grad():
        for batch_id, (images, _) in enumerate(loader):
            images = images.to(args.device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                (
                    _,
                    base_path_logits,
                    text_features,
                    prompted_cls,
                    token_features,
                ) = model.extract_path_features(
                    images,
                    cls_id=(0, seen_classes),
                    inference=True,
                    return_cls_features=True,
                    return_token_features=True,
                )

            # Keep all Adapter arithmetic and the final softmax in float32.
            base_path_logits = base_path_logits.float()
            text_features = text_features.float()
            prompted_cls = prompted_cls.float()
            token_features = token_features.float()
            ddp_scores = _probability(
                base_path_logits, seen_classes, temperature
            )
            cls_logits = cls_bank.logits_from_features(
                prompted_cls,
                base_path_logits,
                text_features,
                seen_classes,
                logit_scale=100.0,
            )
            cls_scores = torch.softmax(
                cls_logits / temperature, dim=1
            )[:, 1, :]

            all_tokens = all_token_bank.adapt_token_features(
                token_features, seen_classes=seen_classes
            )
            _, all_path_logits, _, _ = ddp_pool_final_tokens(
                all_tokens, text_features
            )
            all_scores = _probability(
                all_path_logits, seen_classes, temperature
            )
            all_drift = final_token_drift_tensors(
                all_tokens.permute(0, 1, 3, 2).contiguous(),
                token_features.permute(0, 1, 3, 2).contiguous(),
                text_features,
            )
            _update_drift_accumulator(
                drift_accumulators["all_tokens"],
                all_drift,
            )

            cls_only_tokens = cls_only_bank.adapt_token_features(
                token_features, seen_classes=seen_classes
            )
            _, cls_only_path_logits, _, _ = ddp_pool_final_tokens(
                cls_only_tokens, text_features
            )
            cls_only_scores = _probability(
                cls_only_path_logits, seen_classes, temperature
            )
            cls_only_drift = final_token_drift_tensors(
                cls_only_tokens.permute(0, 1, 3, 2).contiguous(),
                token_features.permute(0, 1, 3, 2).contiguous(),
                text_features,
            )
            _update_drift_accumulator(
                drift_accumulators["cls_only"],
                cls_only_drift,
            )

            batch_scores = {
                "ddp": ddp_scores,
                "old_cls_feature_difference": cls_scores,
                "all_tokens": all_scores,
                "cls_only": cls_only_scores,
            }
            for method, values in batch_scores.items():
                scores[method].append(values.cpu())
            if batch_id % 100 == 0:
                print(
                    f"[K={seen_classes} {split}] shared evaluation "
                    f"{batch_id + 1}/{len(loader)}",
                    flush=True,
                )
            del all_tokens, cls_only_tokens

    return (
        {method: torch.cat(parts) for method, parts in scores.items()},
        labels[source_indices].float(),
        source_indices,
        {
            method: _finalize_drift_accumulator(accumulator)
            for method, accumulator in drift_accumulators.items()
        },
    )


def _method_aggregate(rows, classnames, ddp_rows):
    maps = [row["test"]["mAP"] for row in rows]
    ddp_maps = [row["test"]["mAP"] for row in ddp_rows]
    return {
        "average_mAP": float(np.mean(maps)),
        "final_mAP": maps[-1],
        "average_mAP_gain_vs_ddp": float(np.mean(maps) - np.mean(ddp_maps)),
        "final_mAP_gain_vs_ddp": maps[-1] - ddp_maps[-1],
        "final_cF1": rows[-1]["test"]["cF1"],
        "final_oF1": rows[-1]["test"]["oF1"],
        "average_current_mAP": float(
            np.mean([row["current_test"]["mAP"] for row in rows])
        ),
        "forgetting": forgetting(rows, classnames),
    }


def write_html(path: Path, summary: dict):
    header = "".join(
        f"<th>{escape(METHOD_LABELS[method])}</th>" for method in METHODS
    )
    rows = []
    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        cells = "".join(
            f"<td>{summary['methods'][method]['tasks'][task_id]['test']['mAP']:.4f}</td>"
            for method in METHODS
        )
        rows.append(
            f"<tr><td>{task_id}</td><td>{seen_classes}</td>{cells}</tr>"
        )
    aggregates = {
        method: payload["aggregate"] for method, payload in summary["methods"].items()
    }
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>EMOTIC CLS to Final-token Transfer</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}pre{white-space:pre-wrap}</style>"
        f"<h1>{escape(summary['protocol']['name'])}</h1>"
        "<p>One shared DDP visual extraction per batch; frozen existing CLS "
        "weights; fixed α=0.03 and threshold=0.5; no validation/test selection.</p>"
        f"<pre>{escape(json.dumps(aggregates, indent=2, ensure_ascii=False))}</pre>"
        f"<table><tr><th>Task</th><th>Seen</th>{header}</tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.bank_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if float(manifest.get("inference_alpha", -1)) != TRANSFER_RESIDUAL_SCALE:
        raise ValueError("The locked transfer evaluation requires alpha=0.03")
    if manifest.get("test_used_for_selection") is not False:
        raise ValueError("Source Adapter Bank used test data for selection")

    checkpoint_paths = [
        Path(args.checkpoint_dir) / f"task{task_id}.pth"
        for task_id in range(len(TASK_SEEN_CLASSES))
    ]
    baseline_paths = [
        Path(args.baseline_scores_dir) / f"task{task_id}_scores.pt"
        for task_id in range(len(TASK_SEEN_CLASSES))
    ]
    for path in checkpoint_paths + baseline_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    classnames = list(first_checkpoint["classnames"])
    if classnames != list(manifest.get("classnames", [])):
        raise ValueError("Source Adapter Bank and DDP class orders differ")
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
    rows = {method: [] for method in METHODS}
    transfer_provenance = None

    for method in METHODS:
        (output_dir / method).mkdir(parents=True, exist_ok=True)

    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        ddp_checkpoint = torch.load(checkpoint_paths[task_id], map_location="cpu")
        model.feature_adapter = None
        model.feature_adapter_bank = None
        model.final_token_adapter_bank = None
        model.load_state_dict(ddp_checkpoint["model"], strict=True)
        model.text_feature_cache.clear()
        rebuild_text_feature_cache(model, seen_classes)
        model.eval()

        cls_bank = TaskRoutedAdapterBank.from_manifest(
            manifest_path,
            device,
            max_task=task_id,
            classnames=classnames,
        )
        all_token_bank = TaskRoutedCLSToFinalTokenTransferBank.from_cls_manifest(
            manifest_path,
            application_mode="all_tokens",
            device=device,
            max_task=task_id,
            classnames=classnames,
        )
        cls_only_bank = TaskRoutedCLSToFinalTokenTransferBank.from_cls_manifest(
            manifest_path,
            application_mode="cls_only",
            device=device,
            max_task=task_id,
            classnames=classnames,
        )
        # This is refreshed at every stage so the final record contains the
        # complete task-0..7 source-checkpoint provenance rather than only the
        # task-0 subset loaded for the first stage.
        transfer_provenance = {
            "all_tokens": all_token_bank.transfer_provenance(),
            "cls_only": cls_only_bank.transfer_provenance(),
        }

        temperature = temperature_for_task(
            seen_classes, 26, 5, args.t_min, args.t_max, args.t_gamma
        )
        val_scores, val_targets, val_indices, val_diagnostics = evaluate_split_shared(
            model,
            cls_bank,
            all_token_bank,
            cls_only_bank,
            val_dataset,
            val_labels,
            seen_classes,
            temperature,
            args,
            "val",
        )
        test_scores, test_targets, test_indices, test_diagnostics = (
            evaluate_split_shared(
            model,
            cls_bank,
            all_token_bank,
            cls_only_bank,
            test_dataset,
            test_labels,
            seen_classes,
            temperature,
            args,
            "test",
            )
        )
        low, high = TASK_CLASS_RANGES[task_id]
        external_ddp = _existing_baseline_metrics(
            baseline_paths[task_id], classnames[:seen_classes]
        )
        task_metrics = {}
        for method in METHODS:
            val_metric = score_metrics(
                val_targets[:, :seen_classes],
                val_scores[method],
                FIXED_THRESHOLD,
                classnames[:seen_classes],
            )
            test_metric = score_metrics(
                test_targets[:, :seen_classes],
                test_scores[method],
                FIXED_THRESHOLD,
                classnames[:seen_classes],
            )
            row = {
                "task": task_id,
                "seen_classes": seen_classes,
                "adapter_tasks_loaded": (
                    [] if method == "ddp" else list(range(task_id + 1))
                ),
                "class_range": [low, high],
                "temperature": temperature,
                "decision_threshold": FIXED_THRESHOLD,
                "selection_split": None,
                "val": val_metric,
                "test": test_metric,
                "old_val": _grouped_metrics(
                    val_targets, val_scores[method], classnames, 0, low
                ),
                "old_test": _grouped_metrics(
                    test_targets, test_scores[method], classnames, 0, low
                ),
                "current_val": _grouped_metrics(
                    val_targets, val_scores[method], classnames, low, high
                ),
                "current_test": _grouped_metrics(
                    test_targets, test_scores[method], classnames, low, high
                ),
                "structural_diagnostics": (
                    {
                        "val": val_diagnostics[method],
                        "test": test_diagnostics[method],
                    }
                    if method in val_diagnostics
                    else None
                ),
            }
            if method == "ddp":
                row["existing_baseline_test"] = external_ddp
                row["existing_baseline_mAP_difference"] = (
                    test_metric["mAP"] - external_ddp["mAP"]
                )
            rows[method].append(row)
            task_metrics[method] = test_metric["mAP"]
            torch.save(
                {
                    "split": "test",
                    "scores": test_scores[method],
                    "targets": test_targets[:, :seen_classes],
                    "samples": int(test_targets.shape[0]),
                    "source_indices": test_indices,
                    "temperature": temperature,
                    "threshold": FIXED_THRESHOLD,
                    "task": task_id,
                    "method": method,
                    "bank_manifest": str(manifest_path),
                    "bank_manifest_sha256": file_sha256(manifest_path),
                    "adapter_tasks_loaded": row["adapter_tasks_loaded"],
                    "validation_used_for_selection": False,
                    "test_used_for_selection": False,
                },
                output_dir / method / f"task{task_id}_scores.pt",
            )
        print(
            f"[Task {task_id}] "
            + ", ".join(
                f"{method}={task_metrics[method]:.4f}" for method in METHODS
            ),
            flush=True,
        )
        del cls_bank, all_token_bank, cls_only_bank

    ddp_rows = rows["ddp"]
    method_payloads = {}
    for method in METHODS:
        method_payloads[method] = {
            "label": METHOD_LABELS[method],
            "tasks": rows[method],
            "aggregate": _method_aggregate(
                rows[method], classnames, ddp_rows
            ),
        }
    summary = {
        "protocol": {
            "name": args.name,
            "experiment": "prompt_free_cls_weights_at_final_token_pooling",
            "seed": int(manifest["seed"]),
            "source_training_mode": manifest["training_mode"],
            "routing": "class introduction task; matching +/- paths share Adapter",
            "shared_feature_extraction": True,
            "evaluation_splits": {"reporting": "val", "final": "test"},
            "residual_scale": TRANSFER_RESIDUAL_SCALE,
            "decision_threshold": FIXED_THRESHOLD,
            "checkpoint_selection": "fixed existing source CLS Bank weights",
            "validation_role": "reporting only",
            "validation_used_for_transfer_selection": False,
            "test_used_for_selection": False,
            "adapter_training": False,
            "adapter_fine_tuned": False,
            "class_specific_gate": False,
            "task_specific_alpha": False,
            "external_score_fusion": False,
        },
        "inputs": {
            "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
            "source_bank_manifest": str(manifest_path),
            "source_bank_manifest_sha256": file_sha256(manifest_path),
            "baseline_scores_dir": os.path.abspath(args.baseline_scores_dir),
        },
        "transfer_provenance": transfer_provenance,
        "methods": method_payloads,
        "args": vars(args),
    }
    (output_dir / "transfer_manifest.json").write_text(
        json.dumps(
            {
                "protocol": summary["protocol"],
                "inputs": summary["inputs"],
                "transfer_provenance": summary["transfer_provenance"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    json_path = output_dir / "evaluation_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html(output_dir / "evaluation_summary.html", summary)
    print(
        json.dumps(
            {
                method: payload["aggregate"]["final_mAP"]
                for method, payload in method_payloads.items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
