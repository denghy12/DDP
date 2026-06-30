import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_emotic_prototype_fusion import (
    fit_calibrator,
    load_feature_cache,
    load_prototype_model,
    prototype_logits,
    select_beta,
    select_threshold,
    split_metrics,
)
from evaluation_metrics import mAP


TASK_SEEN_CLASSES = (5, 8, 11, 14, 17, 20, 23, 26)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate val-selected Prototype/DDP fusion over all EMOTIC "
            "B5-C3 tasks without retraining DDP"
        )
    )
    parser.add_argument(
        "--ddp_scores_dir",
        default="./output/emotic_b5c3_ddp_semantic_tau2_threshold050",
    )
    parser.add_argument(
        "--prototype_checkpoint",
        default=(
            "./output/emotic_prototype_adapter_base5_balanced/"
            "best_adapter.pth"
        ),
    )
    parser.add_argument(
        "--val_cache",
        default="./output/emotic_clip_feature_cache/val_full_224_vitb16.pt",
    )
    parser.add_argument(
        "--test_cache",
        default="./output/emotic_clip_feature_cache/test_full_224_vitb16.pt",
    )
    parser.add_argument(
        "--output_dir", default="./output/emotic_prototype_fusion_all_tasks"
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--beta_step", type=float, default=0.02)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--calibration_max_iter", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def seen_sample_mask(labels, seen_classes):
    """Match CODE_DDP's `any(class_id < high_range)` evaluation filter."""
    return labels[:, :seen_classes].bool().any(dim=1)


def select_binary_class_gates(
    targets,
    ddp_scores,
    prototype_scores,
    global_beta,
    classnames=None,
):
    """Choose either DDP-only or the global mixture for each class on val."""
    if not (targets.shape == ddp_scores.shape == prototype_scores.shape):
        raise ValueError("targets and score tensors must have identical shapes")
    candidates = sorted(set([0.0, float(global_beta)]))
    selected = torch.zeros(targets.shape[1], dtype=ddp_scores.dtype)
    rows = []
    for class_id in range(targets.shape[1]):
        best = None
        candidate_rows = []
        for beta in candidates:
            scores = (
                (1.0 - beta) * ddp_scores[:, class_id : class_id + 1]
                + beta * prototype_scores[:, class_id : class_id + 1]
            )
            ap, _ = mAP(
                targets[:, class_id : class_id + 1].numpy(), scores.numpy()
            )
            row = {"beta": beta, "val_ap": float(ap)}
            candidate_rows.append(row)
            # beta=0 is first, so exact AP ties keep the simpler DDP-only path.
            if best is None or row["val_ap"] > best["val_ap"]:
                best = row
        selected[class_id] = best["beta"]
        rows.append(
            {
                "class_id": class_id,
                "class_name": (
                    classnames[class_id]
                    if classnames is not None
                    else str(class_id)
                ),
                "selected_beta": best["beta"],
                "selected_val_ap": best["val_ap"],
                "candidates": candidate_rows,
            }
        )
    return selected, rows


def fuse_classwise(ddp_scores, prototype_scores, class_betas):
    class_betas = class_betas.to(dtype=ddp_scores.dtype).view(1, -1)
    return (1.0 - class_betas) * ddp_scores + class_betas * prototype_scores


def introduction_tasks(total_classes):
    tasks = []
    for class_id in range(total_classes):
        if class_id < TASK_SEEN_CLASSES[0]:
            tasks.append(0)
        else:
            tasks.append(1 + (class_id - TASK_SEEN_CLASSES[0]) // 3)
    return tasks


def forgetting_summary(task_summaries, method, split, classnames):
    """Compute peak-to-final AP forgetting after each class is introduced."""
    intro_tasks = introduction_tasks(len(classnames))
    final_task = len(task_summaries) - 1
    rows = []
    for class_id, class_name in enumerate(classnames):
        intro = intro_tasks[class_id]
        history = [
            task_summaries[task_id]["results"][method][split][
                "per_class_ap"
            ][class_name]
            for task_id in range(intro, final_task + 1)
        ]
        peak_offset = int(np.argmax(history))
        peak_task = intro + peak_offset
        peak_ap = float(history[peak_offset])
        final_ap = float(history[-1])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "introduction_task": intro,
                "introduction_ap": float(history[0]),
                "peak_task": peak_task,
                "peak_ap": peak_ap,
                "final_ap": final_ap,
                "forgetting": peak_ap - final_ap,
                "backward_transfer": final_ap - float(history[0]),
            }
        )
    old_rows = [row for row in rows if row["introduction_task"] < final_task]
    return {
        "definition": "max AP from class introduction through task7 minus task7 AP",
        "evaluated_split": split,
        "old_class_count": len(old_rows),
        "average_forgetting_old_classes": float(
            np.mean([row["forgetting"] for row in old_rows])
        ),
        "average_backward_transfer_old_classes": float(
            np.mean([row["backward_transfer"] for row in old_rows])
        ),
        "per_class": rows,
    }


def aggregate_results(task_summaries, methods, classnames):
    aggregate = {"methods": {}}
    for method in methods:
        method_summary = {}
        for split in ("test", "val_test"):
            task_map = [
                task["results"][method][split]["mAP"]
                for task in task_summaries
            ]
            method_summary[split] = {
                "task_mAP": task_map,
                "average_mAP": float(np.mean(task_map)),
                "final_mAP": float(task_map[-1]),
                "forgetting": forgetting_summary(
                    task_summaries, method, split, classnames
                ),
            }
        aggregate["methods"][method] = method_summary

    for split in ("test", "val_test"):
        ddp = aggregate["methods"]["ddp"][split]
        gated = aggregate["methods"]["binary_gate"][split]
        aggregate[f"binary_gate_gain_over_ddp_{split}"] = {
            "average_mAP": gated["average_mAP"] - ddp["average_mAP"],
            "final_mAP": gated["final_mAP"] - ddp["final_mAP"],
            "old_class_forgetting_reduction": (
                ddp["forgetting"]["average_forgetting_old_classes"]
                - gated["forgetting"]["average_forgetting_old_classes"]
            ),
        }
    return aggregate


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    val_features, val_labels, val_metadata = load_feature_cache(args.val_cache)
    test_features, test_labels, test_metadata = load_feature_cache(args.test_cache)
    if val_metadata.get("classnames") != test_metadata.get("classnames"):
        raise RuntimeError("Validation and test cache class orders differ")

    model, classnames, checkpoint = load_prototype_model(
        args.prototype_checkpoint, device
    )
    if classnames != val_metadata.get("classnames"):
        raise RuntimeError("Prototype checkpoint and cache class orders differ")
    if len(classnames) != TASK_SEEN_CLASSES[-1]:
        raise RuntimeError(
            f"Expected {TASK_SEEN_CLASSES[-1]} classes, got {len(classnames)}"
        )

    all_features = torch.cat([val_features, test_features], dim=0)
    raw_logits = prototype_logits(model, all_features, args.batch_size, device)
    raw_val_logits = raw_logits[: val_features.shape[0]]
    raw_test_logits = raw_logits[val_features.shape[0] :]
    del all_features, raw_logits

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_summaries = []
    methods = ("ddp", "prototype", "global_fusion", "binary_gate")

    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        score_path = Path(args.ddp_scores_dir) / f"task{task_id}_scores.pt"
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        ddp_payload = torch.load(score_path, map_location="cpu")
        ddp_scores = ddp_payload["scores"].float()
        ddp_targets = ddp_payload["targets"].float()

        val_mask = seen_sample_mask(val_labels, seen_classes)
        test_mask = seen_sample_mask(test_labels, seen_classes)
        targets = torch.cat(
            [
                val_labels[val_mask, :seen_classes],
                test_labels[test_mask, :seen_classes],
            ],
            dim=0,
        )
        task_raw_logits = torch.cat(
            [
                raw_val_logits[val_mask, :seen_classes],
                raw_test_logits[test_mask, :seen_classes],
            ],
            dim=0,
        )
        val_count = int(val_mask.sum().item())
        test_count = int(test_mask.sum().item())
        expected_shape = (val_count + test_count, seen_classes)
        if tuple(ddp_scores.shape) != expected_shape:
            raise RuntimeError(
                f"Task {task_id} DDP shape {tuple(ddp_scores.shape)} does not "
                f"match cache-filtered shape {expected_shape}"
            )
        if not torch.equal(ddp_targets, targets):
            mismatch = int(ddp_targets.ne(targets).sum().item())
            raise RuntimeError(
                f"Task {task_id} DDP/cache target order differs at "
                f"{mismatch} entries"
            )

        calibration = fit_calibrator(
            task_raw_logits[:val_count],
            targets[:val_count],
            seen_classes,
            args.calibration_max_iter,
            device,
        )
        calibrator = calibration.pop("model")
        calibrator.eval()
        with torch.no_grad():
            prototype_scores = torch.sigmoid(
                calibrator(task_raw_logits.to(device))
            ).cpu()

        best_beta, beta_rows = select_beta(
            targets[:val_count],
            ddp_scores[:val_count],
            prototype_scores[:val_count],
            args.beta_step,
        )
        global_beta = float(best_beta["beta"])
        global_fusion = (
            (1.0 - global_beta) * ddp_scores + global_beta * prototype_scores
        )
        class_betas, gate_rows = select_binary_class_gates(
            targets[:val_count],
            ddp_scores[:val_count],
            prototype_scores[:val_count],
            global_beta,
            classnames[:seen_classes],
        )
        binary_gate = fuse_classwise(ddp_scores, prototype_scores, class_betas)
        method_scores = {
            "ddp": ddp_scores,
            "prototype": prototype_scores,
            "global_fusion": global_fusion,
            "binary_gate": binary_gate,
        }

        thresholds = {}
        threshold_sweeps = {}
        results = {}
        for method, scores in method_scores.items():
            threshold, rows = select_threshold(
                targets[:val_count], scores[:val_count], args
            )
            thresholds[method] = threshold
            threshold_sweeps[method] = rows
            results[method] = split_metrics(
                scores,
                targets,
                val_count,
                threshold["threshold"],
                classnames[:seen_classes],
            )

        task_summary = {
            "task": task_id,
            "seen_classes": seen_classes,
            "classnames": classnames[:seen_classes],
            "alignment": {
                "val_samples": val_count,
                "test_samples": test_count,
                "total_samples": val_count + test_count,
                "targets_equal": True,
            },
            "ddp_temperature": ddp_payload.get("temperature"),
            "prototype_calibration": calibration,
            "selection": {
                "split": "val",
                "global_beta": best_beta,
                "binary_class_betas": {
                    classnames[index]: float(class_betas[index])
                    for index in range(seen_classes)
                },
                "binary_gate_details": gate_rows,
                "thresholds": thresholds,
            },
            "results": results,
            "beta_sweep": beta_rows,
            "threshold_sweeps": threshold_sweeps,
        }
        task_summaries.append(task_summary)
        torch.save(
            {
                "ddp_scores": ddp_scores,
                "prototype_scores": prototype_scores,
                "global_fusion_scores": global_fusion,
                "binary_gate_scores": binary_gate,
                "targets": targets,
                "val_count": val_count,
                "global_beta": global_beta,
                "class_betas": class_betas,
                "thresholds": {
                    name: row["threshold"] for name, row in thresholds.items()
                },
            },
            output_dir / f"task{task_id}_fusion_scores.pt",
        )

        ddp_test = results["ddp"]["test"]
        gate_test = results["binary_gate"]["test"]
        print(
            f"[Task {task_id}] seen={seen_classes:2d} "
            f"samples={val_count}+{test_count} beta={global_beta:.2f} "
            f"gated_classes={int(class_betas.gt(0).sum().item()):2d} "
            f"test mAP {ddp_test['mAP']:.4f} -> {gate_test['mAP']:.4f} "
            f"cF1 {ddp_test['cF1']:.4f} -> {gate_test['cF1']:.4f} "
            f"oF1 {ddp_test['oF1']:.4f} -> {gate_test['oF1']:.4f}",
            flush=True,
        )

    aggregate = aggregate_results(task_summaries, methods, classnames)
    summary = {
        "protocol": {
            "name": "EMOTIC B5-C3 offline Prototype/DDP fusion",
            "selection_split": "val",
            "test_used_for_selection": False,
            "ddp_retrained": False,
            "prototype_training": "Base5-balanced checkpoint, frozen for all tasks",
            "main_fusion": "per-class binary gate choosing beta in {0, global_beta}",
        },
        "inputs": {
            "ddp_scores_dir": args.ddp_scores_dir,
            "prototype_checkpoint": args.prototype_checkpoint,
            "prototype_checkpoint_epoch": checkpoint.get("epoch"),
            "val_cache": args.val_cache,
            "test_cache": args.test_cache,
        },
        "tasks": task_summaries,
        "aggregate": aggregate,
        "args": vars(args),
    }
    summary_path = output_dir / "fusion_all_tasks_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\nAggregate results (main binary gate versus DDP):")
    for split in ("test", "val_test"):
        ddp = aggregate["methods"]["ddp"][split]
        gate = aggregate["methods"]["binary_gate"][split]
        print(
            f"{split:8s} average mAP {ddp['average_mAP']:.4f} -> "
            f"{gate['average_mAP']:.4f}; final mAP "
            f"{ddp['final_mAP']:.4f} -> {gate['final_mAP']:.4f}; "
            f"old-class forgetting "
            f"{ddp['forgetting']['average_forgetting_old_classes']:.4f} -> "
            f"{gate['forgetting']['average_forgetting_old_classes']:.4f}"
        )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
