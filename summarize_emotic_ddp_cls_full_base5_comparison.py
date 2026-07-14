import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np


METHODS = (
    (
        "feature_difference",
        "Feature difference",
        "emotic_ddp_cls_full_base5_feature_difference_seed",
        "emotic_ddp_cls_full_base5_feature_difference_screen",
    ),
    (
        "cosine_difference",
        "Cosine difference",
        "emotic_ddp_cls_full_base5_cosine_difference_seed",
        "emotic_ddp_cls_full_base5_cosine_difference_screen",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Full Base5 external-to-internal CLS transfer"
    )
    parser.add_argument("--output_root", default="./output")
    parser.add_argument(
        "--baseline_run", default="emotic_ddp_cls_full_base5_ddp_baseline"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_cls_full_base5_comparison",
    )
    return parser.parse_args()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def metric_summary(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def baseline_summary(data):
    aggregate = data["aggregate"]
    final = data["tasks"][-1]["test"]
    return {
        "display_name": "Original DDP",
        "run_count": 1,
        "average_mAP": {
            "mean": aggregate["average_mAP"],
            "std": 0.0,
        },
        "final_mAP": {"mean": aggregate["final_mAP"], "std": 0.0},
        "average_mAP_gain": {"mean": 0.0, "std": 0.0},
        "final_mAP_gain": {"mean": 0.0, "std": 0.0},
        "final_cF1": {"mean": final["cF1"], "std": 0.0},
        "final_oF1": {"mean": final["oF1"], "std": 0.0},
        "forgetting": {
            "mean": aggregate["forgetting"]["average_forgetting_old_classes"],
            "std": 0.0,
        },
        "tasks": [
            {
                "task": row["task"],
                "seen_classes": row["seen_classes"],
                "test_mAP": {"mean": row["test"]["mAP"], "std": 0.0},
                "test_mAP_gain": {"mean": 0.0, "std": 0.0},
            }
            for row in data["tasks"]
        ],
    }


def adapter_summary(output_root, seeds, key, display_name, run_prefix, screen_dir):
    runs = []
    task_values = [[] for _ in range(8)]
    task_gains = [[] for _ in range(8)]
    sources = []
    for seed in seeds:
        data = read_json(
            output_root
            / f"{run_prefix}{seed}"
            / "evaluation_summary.json"
        )
        expected_mode = "linear_residual" if key == "feature_difference" else key
        actual_mode = data["protocol"]["correction_mode"]
        if actual_mode != expected_mode:
            raise RuntimeError(
                f"{run_prefix}{seed} uses {actual_mode}, expected {expected_mode}"
            )
        aggregate = data["aggregate"]
        final = data["tasks"][-1]["test"]
        runs.append(
            {
                "seed": seed,
                "average_mAP": aggregate["average_mAP"],
                "final_mAP": aggregate["final_mAP"],
                "average_mAP_gain": aggregate["average_mAP_gain"],
                "final_mAP_gain": aggregate["final_mAP_gain"],
                "final_cF1": final["cF1"],
                "final_oF1": final["oF1"],
                "forgetting": aggregate["forgetting"][
                    "average_forgetting_old_classes"
                ],
            }
        )
        source = data["inputs"]["transfer_selection"]["source"]
        sources.append(source)
        for task_id, task in enumerate(data["tasks"]):
            task_values[task_id].append(task["test"]["mAP"])
            task_gains[task_id].append(task["test_mAP_gain"])

    screen = read_json(output_root / screen_dir / "transfer_screen_summary.json")
    if not screen["passes_val_gate"]:
        raise RuntimeError(f"{display_name} did not pass its validation gate")
    metrics = [key for key in runs[0] if key != "seed"]
    result = {
        "display_name": display_name,
        "run_count": len(runs),
        "source_checkpoints": sources,
        "selection": {
            "split": screen["selection_split"],
            "test_used": screen["test_used"],
            "global_alpha": screen["best"]["residual_scale"],
            "identity_val_mAP": screen["identity_val_mAP"],
            "selected_val_mAP": screen["best"]["mean_val_mAP"],
            "selected_val_gain": screen["best"]["mean_val_gain"],
            "minimum_seed_gain": screen["best"]["minimum_seed_gain"],
            "all_seeds_positive": screen["best"]["all_seeds_positive"],
        },
        "runs": runs,
        "tasks": [],
    }
    result.update(
        {
            metric: metric_summary([row[metric] for row in runs])
            for metric in metrics
        }
    )
    for task_id, values in enumerate(task_values):
        result["tasks"].append(
            {
                "task": task_id,
                "seen_classes": (5, 8, 11, 14, 17, 20, 23, 26)[task_id],
                "test_mAP": metric_summary(values),
                "test_mAP_gain": metric_summary(task_gains[task_id]),
            }
        )
    return result


def fmt_metric(metric):
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def write_html(path, summary):
    method_order = ("ddp", "feature_difference", "cosine_difference")
    metric_keys = (
        "average_mAP",
        "final_mAP",
        "final_cF1",
        "final_oF1",
        "forgetting",
    )
    main_rows = []
    for key in method_order:
        method = summary["methods"][key]
        cells = "".join(
            f"<td>{escape(fmt_metric(method[metric]))}</td>"
            for metric in metric_keys
        )
        main_rows.append(
            f"<tr><th>{escape(method['display_name'])}</th>{cells}</tr>"
        )
    selection_rows = []
    for key in ("feature_difference", "cosine_difference"):
        method = summary["methods"][key]
        selected = method["selection"]
        selection_rows.append(
            "<tr>"
            f"<th>{escape(method['display_name'])}</th>"
            f"<td>{selected['global_alpha']:.4f}</td>"
            f"<td>{selected['identity_val_mAP']:.4f}</td>"
            f"<td>{selected['selected_val_mAP']:.4f}</td>"
            f"<td>{selected['selected_val_gain']:+.4f}</td>"
            f"<td>{selected['minimum_seed_gain']:+.4f}</td>"
            "</tr>"
        )
    task_rows = []
    for task in summary["task_comparison"]:
        task_rows.append(
            "<tr>"
            f"<td>{task['task']}</td><td>{task['seen_classes']}</td>"
            f"<td>{task['ddp_mAP']:.4f}</td>"
            f"<td>{escape(fmt_metric(task['feature_difference_mAP']))}</td>"
            f"<td>{escape(fmt_metric(task['cosine_difference_mAP']))}</td>"
            "</tr>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Full Base5 CLS Transfer Comparison</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;margin:16px 0 30px;width:100%}"
        "th,td{border:1px solid #ccd3df;padding:8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "h1,h2{color:#172554}.note{background:#f3f6fb;padding:12px}"
        "</style><h1>Full Base5 External Adapter → Internal CLS</h1>"
        "<p class='note'>One global α is selected on task0 validation for each "
        "correction formula. No class gate, task-specific α, external score "
        "fusion, or test-set selection is used.</p>"
        "<h2>Main results</h2><table><tr><th>Method</th>"
        "<th>Average mAP</th><th>Final mAP</th><th>Final cF1</th>"
        "<th>Final oF1</th><th>Forgetting ↓</th></tr>"
        + "".join(main_rows)
        + "</table><h2>Pure-validation selection</h2><table><tr>"
        "<th>Method</th><th>Global α</th><th>Identity Val mAP</th>"
        "<th>Selected Val mAP</th><th>Mean gain</th><th>Minimum seed gain</th>"
        "</tr>"
        + "".join(selection_rows)
        + "</table><h2>Per-task test mAP</h2><table><tr><th>Task</th>"
        "<th>Seen classes</th><th>Original DDP</th><th>Feature difference</th>"
        "<th>Cosine difference</th></tr>"
        + "".join(task_rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    baseline_data = read_json(
        output_root / args.baseline_run / "evaluation_summary.json"
    )
    if not baseline_data["protocol"].get("ddp_only"):
        raise RuntimeError("Baseline summary was not produced with --ddp_only")
    methods = {"ddp": baseline_summary(baseline_data)}
    for key, display_name, run_prefix, screen_dir in METHODS:
        methods[key] = adapter_summary(
            output_root,
            args.seeds,
            key,
            display_name,
            run_prefix,
            screen_dir,
        )
    task_comparison = []
    for task_id in range(8):
        task_comparison.append(
            {
                "task": task_id,
                "seen_classes": methods["ddp"]["tasks"][task_id]["seen_classes"],
                "ddp_mAP": methods["ddp"]["tasks"][task_id]["test_mAP"][
                    "mean"
                ],
                "feature_difference_mAP": methods["feature_difference"]["tasks"]
                [task_id]["test_mAP"],
                "cosine_difference_mAP": methods["cosine_difference"]["tasks"]
                [task_id]["test_mAP"],
            }
        )
    summary = {
        "protocol": {
            "adapter_source": "Full Base5 class-balanced external Adapter",
            "adapter_source_seeds": list(args.seeds),
            "feature_source": "class-specific CLS",
            "selection_split": "task0 val",
            "test_used_for_selection": False,
            "global_alpha": True,
            "task_specific_alpha": False,
            "class_specific_gate": False,
            "external_score_fusion": False,
        },
        "methods": methods,
        "task_comparison": task_comparison,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    csv_rows = []
    for key in ("ddp", "feature_difference", "cosine_difference"):
        method = methods[key]
        csv_rows.append(
            {
                "method": method["display_name"],
                "average_mAP_mean": method["average_mAP"]["mean"],
                "average_mAP_std": method["average_mAP"]["std"],
                "final_mAP_mean": method["final_mAP"]["mean"],
                "final_mAP_std": method["final_mAP"]["std"],
                "final_cF1_mean": method["final_cF1"]["mean"],
                "final_oF1_mean": method["final_oF1"]["mean"],
                "forgetting_mean": method["forgetting"]["mean"],
            }
        )
    with open(
        output_dir / "comparison_summary.csv", "w", newline="", encoding="utf-8"
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_html(output_dir / "comparison_summary.html", summary)
    for key in ("ddp", "feature_difference", "cosine_difference"):
        method = methods[key]
        print(
            f"{method['display_name']:20s} "
            f"average={fmt_metric(method['average_mAP'])} "
            f"final={fmt_metric(method['final_mAP'])}"
        )
    print(f"Saved {output_dir / 'comparison_summary.json'}")
    print(f"Saved {output_dir / 'comparison_summary.html'}")


if __name__ == "__main__":
    main()
