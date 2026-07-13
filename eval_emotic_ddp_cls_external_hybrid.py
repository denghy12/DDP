import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_emotic_prototype_fusion import (
    score_metrics,
    select_beta,
    select_threshold,
)
from eval_emotic_prototype_fusion_all_tasks import (
    forgetting_summary,
    fuse_classwise,
    select_binary_class_gates,
)
from src.helper_functions.detail_report import DetailReport
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import eval_transform


TASK_SEEN_CLASSES = (5, 8, 11, 14, 17, 20, 23, 26)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse internal CLS-gated DDP with external Prototype scores"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ddp_checkpoint_dir", required=True)
    parser.add_argument("--internal_dir", required=True)
    parser.add_argument("--external_dir", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--beta_step", type=float, default=0.02)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    return parser.parse_args()


def class_mask():
    return [list(range(5))] + [
        list(range(low, min(low + 3, 26))) for low in range(5, 26, 3)
    ]


def require_aligned_targets(task_id, internal_val, internal_test, external, val_count):
    external_val = external[:val_count]
    external_test = external[val_count:]
    if not torch.equal(internal_val, external_val):
        raise RuntimeError(f"Task {task_id} val target alignment failed")
    if not torch.equal(internal_test, external_test):
        raise RuntimeError(f"Task {task_id} test target alignment failed")


def aggregate_method(tasks, method, classnames):
    maps = [task["results"][method]["test"]["mAP"] for task in tasks]
    return {
        "task_mAP": maps,
        "average_mAP": float(np.mean(maps)),
        "final_mAP": float(maps[-1]),
        "forgetting": forgetting_summary(tasks, method, "test", classnames),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_checkpoint = torch.load(
        Path(args.ddp_checkpoint_dir) / "task0.pth", map_location="cpu"
    )
    classnames = list(first_checkpoint["classnames"])
    test_dataset = EMOTIC(
        args.data_root,
        train=False,
        eval_splits=("test",),
        transform=eval_transform(),
        input_mode="full",
    )
    report = DetailReport(
        str(output_dir), args.name, classnames, class_mask(), test_dataset.targets
    )
    methods = ("ddp", "internal_gate", "global_hybrid", "binary_hybrid")
    task_summaries = []

    for task_id, seen_classes in enumerate(TASK_SEEN_CLASSES):
        internal = torch.load(
            Path(args.internal_dir) / f"task{task_id}_scores.pt",
            map_location="cpu",
        )
        external = torch.load(
            Path(args.external_dir) / f"task{task_id}_fusion_scores.pt",
            map_location="cpu",
        )
        required = (
            "val_targets",
            "test_targets",
            "val_ddp_scores",
            "test_ddp_scores",
            "val_class_gate_scores",
            "test_class_gate_scores",
        )
        missing = [key for key in required if key not in internal]
        if missing:
            raise RuntimeError(
                f"Internal task{task_id} scores need re-export; missing {missing}"
            )
        val_count = int(external["val_count"])
        ext_targets = external["targets"][:, :seen_classes].float()
        val_targets = internal["val_targets"].float()
        test_targets = internal["test_targets"].float()
        require_aligned_targets(
            task_id, val_targets, test_targets, ext_targets, val_count
        )

        prototype = external["prototype_scores"][:, :seen_classes].float()
        proto_val = prototype[:val_count]
        proto_test = prototype[val_count:]
        val_ddp = internal["val_ddp_scores"].float()
        test_ddp = internal["test_ddp_scores"].float()
        val_internal = internal["val_class_gate_scores"].float()
        test_internal = internal["test_class_gate_scores"].float()

        best_beta, beta_rows = select_beta(
            val_targets,
            val_internal,
            proto_val,
            args.beta_step,
        )
        beta = float(best_beta["beta"])
        val_global = (1.0 - beta) * val_internal + beta * proto_val
        test_global = (1.0 - beta) * test_internal + beta * proto_test
        class_betas, gate_rows = select_binary_class_gates(
            val_targets,
            val_internal,
            proto_val,
            beta,
            classnames[:seen_classes],
        )
        val_binary = fuse_classwise(val_internal, proto_val, class_betas)
        test_binary = fuse_classwise(test_internal, proto_test, class_betas)
        val_scores = {
            "ddp": val_ddp,
            "internal_gate": val_internal,
            "global_hybrid": val_global,
            "binary_hybrid": val_binary,
        }
        test_scores = {
            "ddp": test_ddp,
            "internal_gate": test_internal,
            "global_hybrid": test_global,
            "binary_hybrid": test_binary,
        }
        thresholds = {}
        results = {}
        for method in methods:
            threshold, sweep = select_threshold(
                val_targets, val_scores[method], args
            )
            thresholds[method] = {"selected": threshold, "sweep": sweep}
            results[method] = {
                "val": score_metrics(
                    val_targets,
                    val_scores[method],
                    threshold["threshold"],
                    classnames[:seen_classes],
                ),
                "test": score_metrics(
                    test_targets,
                    test_scores[method],
                    threshold["threshold"],
                    classnames[:seen_classes],
                ),
            }

        full_scores = torch.zeros(test_binary.shape[0], 26)
        full_scores[:, :seen_classes] = test_binary
        full_targets = torch.zeros(test_targets.shape[0], 26)
        full_targets[:, :seen_classes] = test_targets
        loss = torch.nn.functional.binary_cross_entropy(
            test_binary.clamp(1e-6, 1 - 1e-6), test_targets
        ).item()
        report_row = report.update(
            task_id,
            full_scores,
            full_targets,
            thresholds["binary_hybrid"]["selected"]["threshold"],
            loss,
        )
        task_summary = {
            "task": task_id,
            "seen_classes": seen_classes,
            "selection": {
                "split": "val",
                "global_beta": best_beta,
                "beta_sweep": beta_rows,
                "enabled_external_classes": int(class_betas.gt(0).sum()),
                "class_gates": gate_rows,
                "thresholds": thresholds,
            },
            "results": results,
            "test": results["binary_hybrid"]["test"],
            "report": report_row,
        }
        task_summaries.append(task_summary)
        torch.save(
            {
                "targets": test_targets,
                "ddp_scores": test_ddp,
                "internal_gate_scores": test_internal,
                "prototype_scores": proto_test,
                "global_hybrid_scores": test_global,
                "binary_hybrid_scores": test_binary,
                "beta": beta,
                "class_betas": class_betas,
            },
            output_dir / f"task{task_id}_scores.pt",
        )
        print(
            f"[Task {task_id}] beta={beta:.2f} external_classes="
            f"{int(class_betas.gt(0).sum())}/{seen_classes} test mAP "
            f"{results['ddp']['test']['mAP']:.4f} -> "
            f"{results['internal_gate']['test']['mAP']:.4f} -> "
            f"{results['binary_hybrid']['test']['mAP']:.4f}",
            flush=True,
        )

    aggregates = {
        method: aggregate_method(task_summaries, method, classnames)
        for method in methods
    }
    baseline = aggregates["ddp"]
    main_result = aggregates["binary_hybrid"]
    aggregate = {
        "methods": aggregates,
        "average_mAP": main_result["average_mAP"],
        "final_mAP": main_result["final_mAP"],
        "baseline_average_mAP": baseline["average_mAP"],
        "baseline_final_mAP": baseline["final_mAP"],
        "average_mAP_gain": main_result["average_mAP"] - baseline["average_mAP"],
        "final_mAP_gain": main_result["final_mAP"] - baseline["final_mAP"],
        "forgetting": main_result["forgetting"],
    }
    summary = {
        "protocol": {
            "name": "Internal CLS gate plus external 16-shot Prototype upper",
            "selection_split": "val",
            "test_used_for_selection": False,
            "main_method": "binary_hybrid",
            "external_clip_branch": True,
        },
        "seed": args.seed,
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
