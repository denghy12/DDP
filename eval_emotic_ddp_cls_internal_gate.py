import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_emotic_ddp_internal_adapter import (
    TASK_SEEN_CLASSES,
    build_model,
    class_mask,
    load_task_model,
    predict_from_cache,
    task_feature_cache,
)
from eval_emotic_prototype_fusion import score_metrics, select_threshold
from eval_emotic_prototype_fusion_all_tasks import forgetting_summary
from evaluation_metrics import mAP
from src.helper_functions.detail_report import DetailReport
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import dense_labels, eval_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict task-alpha and class-gated internal CLS residual"
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--alpha_candidates",
        nargs="+",
        type=float,
        default=(0.0, 0.001, 0.003, 0.01, 0.03),
    )
    parser.add_argument("--task_val_margin", type=float, default=0.1)
    parser.add_argument("--class_val_margin", type=float, default=0.1)
    parser.add_argument("--feature_batch_size", type=int, default=4)
    parser.add_argument("--adapter_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=2.0)
    parser.add_argument("--t_gamma", type=float, default=0.7)
    parser.add_argument("--force_recache", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def single_class_ap(targets, scores, class_id):
    value, _ = mAP(
        targets[:, class_id : class_id + 1].numpy(),
        scores[:, class_id : class_id + 1].numpy(),
    )
    return float(value)


def select_task_alpha(
    targets,
    scores_by_alpha,
    alpha_candidates,
    minimum_gain,
):
    rows = []
    for alpha in alpha_candidates:
        value, _ = mAP(targets.numpy(), scores_by_alpha[alpha].numpy())
        rows.append({"alpha": float(alpha), "val_mAP": float(value)})
    baseline = next(row for row in rows if row["alpha"] == 0.0)
    unconstrained = max(rows, key=lambda row: row["val_mAP"])
    if unconstrained["val_mAP"] > baseline["val_mAP"] + minimum_gain:
        selected = unconstrained
    else:
        selected = baseline
    return selected, baseline, rows


def select_class_gates(
    targets,
    baseline_scores,
    adapted_scores,
    classnames,
    minimum_gain,
):
    gates = torch.zeros(targets.shape[1], dtype=torch.bool)
    rows = []
    for class_id, class_name in enumerate(classnames):
        baseline_ap = single_class_ap(targets, baseline_scores, class_id)
        adapted_ap = single_class_ap(targets, adapted_scores, class_id)
        gain = adapted_ap - baseline_ap
        enabled = gain > minimum_gain
        gates[class_id] = enabled
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "baseline_val_ap": baseline_ap,
                "adapted_val_ap": adapted_ap,
                "val_ap_gain": gain,
                "gate": int(enabled),
            }
        )
    return gates, rows


def apply_class_gates(baseline_scores, adapted_scores, gates):
    return torch.where(gates.view(1, -1), adapted_scores, baseline_scores)


def method_aggregate(task_summaries, method, classnames):
    task_maps = [
        task["results"][method]["test"]["mAP"] for task in task_summaries
    ]
    return {
        "task_mAP": task_maps,
        "average_mAP": float(np.mean(task_maps)),
        "final_mAP": float(task_maps[-1]),
        "forgetting": forgetting_summary(
            task_summaries, method, "test", classnames
        ),
    }


def main():
    args = parse_args()
    args.feature_source = "cls"
    if 0.0 not in args.alpha_candidates:
        raise ValueError("alpha_candidates must include 0.0 identity baseline")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [
        Path(args.checkpoint_dir) / f"task{task_id}.pth" for task_id in range(8)
    ]
    adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    classnames = list(first_checkpoint["classnames"])
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
    report = DetailReport(
        str(output_dir), args.name, classnames, class_mask(), test_dataset.targets
    )
    methods = ("ddp", "task_alpha", "class_gate")
    task_summaries = []

    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        checkpoint = torch.load(checkpoint_paths[task_id], map_location="cpu")
        load_task_model(model, checkpoint, adapter_checkpoint, seen_classes)
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
        val_targets = val_cache["targets"][:, :seen_classes]
        test_targets = test_cache["targets"][:, :seen_classes]

        val_scores_by_alpha = {}
        temperature = None
        for alpha in args.alpha_candidates:
            model.feature_adapter.residual_scale = float(alpha)
            scores, temperature = predict_from_cache(
                model, val_cache, seen_classes, args
            )
            val_scores_by_alpha[float(alpha)] = scores
        selected, baseline_row, alpha_rows = select_task_alpha(
            val_targets,
            val_scores_by_alpha,
            [float(alpha) for alpha in args.alpha_candidates],
            args.task_val_margin,
        )
        selected_alpha = float(selected["alpha"])
        val_ddp = val_scores_by_alpha[0.0]
        val_task_alpha = val_scores_by_alpha[selected_alpha]
        gates, gate_rows = select_class_gates(
            val_targets,
            val_ddp,
            val_task_alpha,
            classnames[:seen_classes],
            args.class_val_margin,
        )
        val_gate = apply_class_gates(val_ddp, val_task_alpha, gates)

        model.feature_adapter.residual_scale = 0.0
        test_ddp, _ = predict_from_cache(model, test_cache, seen_classes, args)
        model.feature_adapter.residual_scale = selected_alpha
        test_task_alpha, _ = predict_from_cache(
            model, test_cache, seen_classes, args
        )
        test_gate = apply_class_gates(test_ddp, test_task_alpha, gates)
        val_method_scores = {
            "ddp": val_ddp,
            "task_alpha": val_task_alpha,
            "class_gate": val_gate,
        }
        test_method_scores = {
            "ddp": test_ddp,
            "task_alpha": test_task_alpha,
            "class_gate": test_gate,
        }
        thresholds = {}
        results = {}
        for method in methods:
            threshold, sweep = select_threshold(
                val_targets, val_method_scores[method], args
            )
            thresholds[method] = {"selected": threshold, "sweep": sweep}
            results[method] = {
                "val": score_metrics(
                    val_targets,
                    val_method_scores[method],
                    threshold["threshold"],
                    classnames[:seen_classes],
                ),
                "test": score_metrics(
                    test_targets,
                    test_method_scores[method],
                    threshold["threshold"],
                    classnames[:seen_classes],
                ),
            }

        full_scores = torch.zeros(test_gate.shape[0], 26)
        full_scores[:, :seen_classes] = test_gate
        loss = F.binary_cross_entropy(
            test_gate.clamp(1e-6, 1 - 1e-6), test_targets
        ).item()
        report_row = report.update(
            task_id,
            full_scores,
            test_cache["targets"],
            thresholds["class_gate"]["selected"]["threshold"],
            loss,
        )
        task_summary = {
            "task": task_id,
            "seen_classes": seen_classes,
            "temperature": temperature,
            "selection": {
                "split": "val",
                "task_alpha": selected,
                "identity": baseline_row,
                "alpha_sweep": alpha_rows,
                "class_gate_margin": args.class_val_margin,
                "enabled_classes": int(gates.sum().item()),
                "class_gates": gate_rows,
                "thresholds": thresholds,
            },
            "results": results,
            "test": results["class_gate"]["test"],
            "report": report_row,
        }
        task_summaries.append(task_summary)
        torch.save(
            {
                "val_targets": val_targets,
                "test_targets": test_targets,
                "val_ddp_scores": val_ddp,
                "test_ddp_scores": test_ddp,
                "val_task_alpha_scores": val_task_alpha,
                "test_task_alpha_scores": test_task_alpha,
                "val_class_gate_scores": val_gate,
                "test_class_gate_scores": test_gate,
                "selected_alpha": selected_alpha,
                "class_gates": gates,
            },
            output_dir / f"task{task_id}_scores.pt",
        )
        print(
            f"[Task {task_id}] alpha={selected_alpha:.3f} "
            f"gates={int(gates.sum())}/{seen_classes} test mAP "
            f"{results['ddp']['test']['mAP']:.4f} -> "
            f"{results['task_alpha']['test']['mAP']:.4f} -> "
            f"{results['class_gate']['test']['mAP']:.4f}",
            flush=True,
        )

    aggregate_methods = {
        method: method_aggregate(task_summaries, method, classnames)
        for method in methods
    }
    baseline = aggregate_methods["ddp"]
    main_result = aggregate_methods["class_gate"]
    aggregate = {
        "methods": aggregate_methods,
        "average_mAP": main_result["average_mAP"],
        "final_mAP": main_result["final_mAP"],
        "baseline_average_mAP": baseline["average_mAP"],
        "baseline_final_mAP": baseline["final_mAP"],
        "average_mAP_gain": (
            main_result["average_mAP"] - baseline["average_mAP"]
        ),
        "final_mAP_gain": main_result["final_mAP"] - baseline["final_mAP"],
        "forgetting": main_result["forgetting"],
    }
    summary = {
        "protocol": {
            "name": "EMOTIC DDP internal CLS task-alpha and class gate",
            "selection_split": "val",
            "test_used_for_selection": False,
            "external_clip_branch": False,
            "adapter_retrained": False,
            "main_method": "class_gate",
        },
        "inputs": {
            "checkpoint_dir": args.checkpoint_dir,
            "adapter_checkpoint": args.adapter_checkpoint,
            "cache_dir": args.cache_dir,
        },
        "tasks": task_summaries,
        "aggregate": aggregate,
        "args": vars(args),
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Saved {output_dir / 'evaluation_summary.json'}", flush=True)
    print(f"Saved {report.path}", flush=True)


if __name__ == "__main__":
    main()
