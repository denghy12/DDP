"""Summarize the frozen CLS-Adapter to Final-token transfer experiment.

Each seed is evaluated in one shared DDP feature pass.  Its
``evaluation_summary.json`` contains the DDP reference, the original
CLS feature-difference use of the bank, and the two transfer placements:
all final tokens or the CLS token only.
"""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path
from statistics import fmean, pstdev


METHODS = (
    "ddp",
    "old_cls_feature_difference",
    "all_tokens",
    "cls_only",
)
METHOD_LABELS = {
    "ddp": "DDP baseline",
    "old_cls_feature_difference": "Original CLS Feature Difference",
    "all_tokens": "CLS-trained Adapter → all 197 tokens",
    "cls_only": "CLS-trained Adapter → CLS token only",
    "pooling_aware_final_token": "Pooling-aware Final-token Bank",
}
METHOD_ALIASES = {
    "ddp": ("ddp", "ddp_baseline", "baseline_ddp"),
    "old_cls_feature_difference": (
        "old_cls_feature_difference",
        "cls_feature_difference",
        "original_cls_feature_difference",
    ),
    "all_tokens": ("all_tokens", "all_token_transfer"),
    "cls_only": ("cls_only", "cls_token_only"),
}
METRICS = (
    "average_mAP",
    "final_mAP",
    "final_cF1",
    "final_oF1",
    "average_current_mAP",
    "forgetting",
)
DIAGNOSTIC_METHODS = ("all_tokens", "cls_only")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize frozen CLS-to-Final-token transfer"
    )
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default=(
            "./output/"
            "emotic_ddp_cls_to_final_token_transfer_comparison"
        ),
    )
    return parser.parse_args()


def evaluation_path(output_root: Path, seed: int) -> Path:
    return (
        output_root
        / f"emotic_ddp_cls_to_final_token_transfer_seed{seed}"
        / "evaluation_summary.json"
    )


def _method_payload(data: dict, method: str, path: Path) -> dict:
    container = data.get("methods", data.get("results", {}))
    for alias in METHOD_ALIASES[method]:
        if alias in container:
            payload = container[alias]
            if not isinstance(payload, dict):
                raise TypeError(f"{path}: method {alias!r} is not an object")
            return payload
    raise KeyError(
        f"{path}: missing method {method!r}; available={sorted(container)}"
    )


def _locked_float(protocol: dict, names, expected: float, path: Path):
    for name in names:
        if name in protocol:
            value = float(protocol[name])
            if value != expected:
                raise ValueError(
                    f"{path}: unlocked {name}={value}, expected {expected}"
                )
            return
    raise KeyError(f"{path}: protocol is missing one of {tuple(names)}")


def validate_run(data: dict, seed: int, path: Path):
    protocol = data.get("protocol")
    if not isinstance(protocol, dict):
        raise KeyError(f"{path}: missing protocol")
    if int(protocol.get("seed", -1)) != int(seed):
        raise ValueError(f"{path}: seed mismatch")
    _locked_float(
        protocol,
        ("residual_scale", "inference_alpha", "adapter_residual_scale"),
        0.03,
        path,
    )
    _locked_float(protocol, ("decision_threshold", "threshold"), 0.5, path)
    if protocol.get("test_used_for_selection") is not False:
        raise ValueError(f"{path}: test_used_for_selection must be false")
    if protocol.get("adapter_fine_tuned", False) is not False:
        raise ValueError(f"{path}: transferred Adapter must remain frozen")
    if protocol.get("adapter_training", False) is not False:
        raise ValueError(f"{path}: this experiment must not train an Adapter")
    if protocol.get("class_specific_gate", False) is not False:
        raise ValueError(f"{path}: class-specific gates are forbidden")
    if protocol.get("task_specific_alpha", False) is not False:
        raise ValueError(f"{path}: task-specific alpha is forbidden")
    if protocol.get("external_score_fusion", False) is not False:
        raise ValueError(f"{path}: external score fusion is forbidden")
    for method in METHODS:
        payload = _method_payload(data, method, path)
        if not isinstance(payload.get("aggregate"), dict):
            raise KeyError(f"{path}: {method} is missing aggregate")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 8:
            raise ValueError(f"{path}: {method} must contain eight tasks")


def _forgetting_value(aggregate: dict) -> float:
    value = aggregate.get("forgetting", 0.0)
    if isinstance(value, dict):
        value = value.get("average_forgetting_old_classes", 0.0)
    return float(value)


def _aggregate_metric(aggregate: dict, name: str) -> float:
    if name == "forgetting":
        return _forgetting_value(aggregate)
    if name in aggregate:
        return float(aggregate[name])
    if name == "final_cF1":
        return float(aggregate.get("baseline_final_cF1", 0.0))
    if name == "final_oF1":
        return float(aggregate.get("baseline_final_oF1", 0.0))
    if name == "average_current_mAP":
        return float(aggregate.get("average_mAP", 0.0))
    raise KeyError(name)


def _task_metric(row: dict, name: str) -> float:
    metrics = row.get("test", row)
    if name == "mAP":
        return float(metrics.get("mAP", row.get("test_mAP")))
    return float(metrics[name])


def collect_transfer_runs(output_root: Path, seeds):
    runs = []
    method_task_sets = {method: [] for method in METHODS}
    source_manifests = []
    source_sha256 = []
    for seed in seeds:
        path = evaluation_path(output_root, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_run(data, seed, path)
        inputs = data.get("inputs", {})
        manifest = inputs.get(
            "source_bank_manifest", inputs.get("bank_manifest")
        )
        if manifest is not None:
            source_manifests.append(str(manifest))
        digest = inputs.get(
            "source_bank_manifest_sha256",
            inputs.get("bank_manifest_sha256"),
        )
        if digest is not None:
            source_sha256.append(str(digest))
        for method in METHODS:
            payload = _method_payload(data, method, path)
            aggregate = payload["aggregate"]
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": int(seed),
            }
            row.update(
                {
                    metric: _aggregate_metric(aggregate, metric)
                    for metric in METRICS
                }
            )
            runs.append(row)
            method_task_sets[method].append(payload["tasks"])
    aggregate_by_method = {}
    for method in METHODS:
        method_runs = [row for row in runs if row["method"] == method]
        aggregate_by_method[method] = {
            metric: {
                "mean": float(fmean([row[metric] for row in method_runs])),
                "std": float(pstdev([row[metric] for row in method_runs])),
            }
            for metric in METRICS
        }
    return (
        runs,
        aggregate_by_method,
        method_task_sets,
        sorted(set(source_manifests)),
        sorted(set(source_sha256)),
    )


def collect_per_task(method_task_sets: dict):
    rows = []
    for method in METHODS:
        task_sets = method_task_sets[method]
        for task_id in range(8):
            task_rows = [tasks[task_id] for tasks in task_sets]
            seen_classes = int(
                task_rows[0].get("seen_classes", 5 + 3 * task_id)
            )
            values = [_task_metric(row, "mAP") for row in task_rows]
            cf1 = [_task_metric(row, "cF1") for row in task_rows]
            of1 = [_task_metric(row, "oF1") for row in task_rows]
            current = []
            for row in task_rows:
                current_metrics = row.get("current_test")
                current.append(
                    float(
                        current_metrics["mAP"]
                        if isinstance(current_metrics, dict)
                        else row.get("current_mAP", values[len(current)])
                    )
                )
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "task": task_id,
                    "seen_classes": seen_classes,
                    "mAP_mean": float(fmean(values)),
                    "mAP_std": float(pstdev(values)),
                    "current_mAP_mean": float(fmean(current)),
                    "current_mAP_std": float(pstdev(current)),
                    "cF1_mean": float(fmean(cf1)),
                    "oF1_mean": float(fmean(of1)),
                }
            )
    reference = {
        row["task"]: row
        for row in rows
        if row["method"] == "ddp"
    }
    old_cls = {
        row["task"]: row
        for row in rows
        if row["method"] == "old_cls_feature_difference"
    }
    for row in rows:
        task = row["task"]
        row["mAP_vs_ddp"] = (
            row["mAP_mean"] - reference[task]["mAP_mean"]
        )
        row["mAP_vs_old_cls"] = (
            row["mAP_mean"] - old_cls[task]["mAP_mean"]
        )
    return rows


def collect_per_class(method_task_sets: dict):
    rows = []
    for method in METHODS:
        final_rows = [tasks[-1] for tasks in method_task_sets[method]]
        per_class_sets = [
            row.get("test", row).get("per_class_ap", {})
            for row in final_rows
        ]
        if not all(per_class_sets):
            continue
        classnames = list(per_class_sets[0])
        if any(list(values) != classnames for values in per_class_sets[1:]):
            raise ValueError(f"{method}: final per-class AP order differs")
        for class_name in classnames:
            values = [
                float(per_class[class_name]) for per_class in per_class_sets
            ]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "class_name": class_name,
                    "final_ap_mean": float(fmean(values)),
                    "final_ap_std": float(pstdev(values)),
                }
            )
    reference = {
        row["class_name"]: row
        for row in rows
        if row["method"] == "ddp"
    }
    old_cls = {
        row["class_name"]: row
        for row in rows
        if row["method"] == "old_cls_feature_difference"
    }
    for row in rows:
        class_name = row["class_name"]
        row["final_ap_vs_ddp"] = (
            row["final_ap_mean"] - reference[class_name]["final_ap_mean"]
        )
        row["final_ap_vs_old_cls"] = (
            row["final_ap_mean"] - old_cls[class_name]["final_ap_mean"]
        )
    return rows


def collect_structural_diagnostics(method_task_sets: dict):
    """Aggregate test-split DDP-structure drift across the three source seeds."""

    rows = []
    for method in DIAGNOSTIC_METHODS:
        task_sets = method_task_sets[method]
        for task_id in range(8):
            task_rows = [tasks[task_id] for tasks in task_sets]
            diagnostic_sets = [
                row.get("structural_diagnostics", {}).get("test", {})
                for row in task_rows
            ]
            if not all(diagnostic_sets):
                raise ValueError(
                    f"{method} task{task_id}: missing structural diagnostics"
                )
            metric_names = list(diagnostic_sets[0])
            if any(list(item) != metric_names for item in diagnostic_sets[1:]):
                raise ValueError(
                    f"{method} task{task_id}: diagnostic keys differ by seed"
                )
            for metric_name in metric_names:
                metrics = [item[metric_name] for item in diagnostic_sets]
                if not all(isinstance(item, dict) for item in metrics):
                    raise ValueError(
                        f"{method} task{task_id}: invalid {metric_name}"
                    )
                means = [float(item["mean"]) for item in metrics]
                rms_values = [float(item["rms"]) for item in metrics]
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "task": task_id,
                        "metric": metric_name,
                        "mean_across_seeds": float(fmean(means)),
                        "std_across_seeds": float(pstdev(means)),
                        "rms_across_seeds": float(fmean(rms_values)),
                        "minimum_across_seeds": float(
                            min(float(item["min"]) for item in metrics)
                        ),
                        "maximum_across_seeds": float(
                            max(float(item["max"]) for item in metrics)
                        ),
                    }
                )
    return rows


def collect_pooling_aware_reference(output_root: Path, seeds):
    runs = []
    missing = []
    for seed in seeds:
        path = (
            output_root
            / f"emotic_ddp_final_token_pooling_aware_bank_full_seed{seed}"
            / "evaluation_summary.json"
        )
        if not path.is_file():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        aggregate = data["aggregate"]
        runs.append(
            {
                metric: _aggregate_metric(aggregate, metric)
                for metric in METRICS
            }
        )
    if missing:
        return {
            "status": "missing",
            "missing_paths": missing,
            "note": "Optional historical reference; transfer summary is complete.",
        }
    return {
        "status": "complete",
        "aggregate": {
            metric: {
                "mean": float(fmean([row[metric] for row in runs])),
                "std": float(pstdev([row[metric] for row in runs])),
            }
            for metric in METRICS
        },
    }


def _fmt(metric: dict) -> str:
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def write_html(path: Path, summary: dict):
    aggregate = summary["methods"]
    ddp_final = aggregate["ddp"]["final_mAP"]["mean"]
    old_final = aggregate["old_cls_feature_difference"]["final_mAP"]["mean"]
    aggregate_rows = []
    for method in METHODS:
        metrics = aggregate[method]
        aggregate_rows.append(
            "<tr>"
            f"<td>{escape(METHOD_LABELS[method])}</td>"
            f"<td>{_fmt(metrics['average_mAP'])}</td>"
            f"<td>{_fmt(metrics['final_mAP'])}</td>"
            f"<td>{metrics['final_mAP']['mean'] - ddp_final:+.4f}</td>"
            f"<td>{metrics['final_mAP']['mean'] - old_final:+.4f}</td>"
            f"<td>{_fmt(metrics['final_cF1'])}</td>"
            f"<td>{_fmt(metrics['final_oF1'])}</td>"
            f"<td>{_fmt(metrics['forgetting'])}</td>"
            "</tr>"
        )
    task_rows = []
    by_method_task = {
        (row["method"], row["task"]): row
        for row in summary["per_task"]
    }
    for task_id in range(8):
        ddp = by_method_task[("ddp", task_id)]
        old = by_method_task[("old_cls_feature_difference", task_id)]
        all_tokens = by_method_task[("all_tokens", task_id)]
        cls_only = by_method_task[("cls_only", task_id)]
        task_rows.append(
            "<tr>"
            f"<td>{task_id}</td><td>{ddp['seen_classes']}</td>"
            f"<td>{ddp['mAP_mean']:.4f}</td>"
            f"<td>{old['mAP_mean']:.4f} ± {old['mAP_std']:.4f}</td>"
            f"<td>{all_tokens['mAP_mean']:.4f} ± {all_tokens['mAP_std']:.4f}</td>"
            f"<td>{all_tokens['mAP_vs_ddp']:+.4f}</td>"
            f"<td>{cls_only['mAP_mean']:.4f} ± {cls_only['mAP_std']:.4f}</td>"
            f"<td>{cls_only['mAP_vs_ddp']:+.4f}</td>"
            "</tr>"
        )
    optional = summary["references"]["pooling_aware_final_token"]
    optional_html = (
        "<p>Pooling-aware Final-token historical reference: "
        + (
            escape(_fmt(optional["aggregate"]["final_mAP"]))
            if optional["status"] == "complete"
            else "not present (optional)"
        )
        + "</p>"
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>CLS Adapter to Final-token Transfer</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-bottom:28px}"
        "th,td{border:1px solid #d9deea;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}td:first-child{text-align:left}"
        ".note{background:#f4f6fb;padding:12px;border-left:4px solid #416fbd}"
        "</style><h1>EMOTIC CLS → Final-token Transfer</h1>"
        "<p class='note'>Frozen prompt-free CLS-trained Task Adapter Bank · "
        "class-introduction-task routing · α=0.03 · fixed threshold=0.5 · "
        "no fine-tuning, gates, score fusion, validation selection, or test "
        "selection. Each seed's four methods share one DDP feature pass.</p>"
        + optional_html
        + "<h2>Three-seed aggregate</h2><table><tr><th>Method</th>"
        "<th>Average mAP</th><th>Final mAP</th><th>vs DDP</th>"
        "<th>vs Original CLS</th><th>Final cF1</th><th>Final oF1</th>"
        "<th>Forgetting</th></tr>"
        + "".join(aggregate_rows)
        + "</table><h2>Per-task comparison</h2><table><tr><th>Task</th>"
        "<th>Seen</th><th>DDP</th><th>Original CLS</th>"
        "<th>All-token</th><th>All vs DDP</th><th>CLS-only</th>"
        "<th>CLS-only vs DDP</th></tr>"
        + "".join(task_rows)
        + "</table>",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    (
        runs,
        aggregate_by_method,
        method_task_sets,
        source_manifests,
        source_sha256,
    ) = collect_transfer_runs(output_root, args.seeds)
    per_task = collect_per_task(method_task_sets)
    per_class = collect_per_class(method_task_sets)
    structural_diagnostics = collect_structural_diagnostics(method_task_sets)
    summary = {
        "protocol": {
            "experiment": "emotic_ddp_cls_to_final_token_transfer",
            "seeds": list(args.seeds),
            "source_training_features": "prompt-free global CLIP CLS",
            "source_training_mode": "full",
            "routing": "class introduction task",
            "transfer_application_modes": ["all_tokens", "cls_only"],
            "residual_scale": 0.03,
            "decision_threshold": 0.5,
            "adapter_fine_tuned": False,
            "validation_used_for_transfer_selection": False,
            "test_used_for_selection": False,
        },
        "inputs": {
            "source_bank_manifests": source_manifests,
            "source_bank_manifest_sha256": source_sha256,
        },
        "methods": aggregate_by_method,
        "runs": runs,
        "per_task": per_task,
        "per_class": per_class,
        "structural_diagnostics": structural_diagnostics,
        "references": {
            "pooling_aware_final_token": collect_pooling_aware_reference(
                output_root, args.seeds
            )
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "per_task_comparison.json").write_text(
        json.dumps(per_task, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "per_class_comparison.json").write_text(
        json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "structural_diagnostics.json").write_text(
        json.dumps(structural_diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "comparison_summary.csv", runs)
    _write_csv(output_dir / "per_task_comparison.csv", per_task)
    _write_csv(output_dir / "per_class_comparison.csv", per_class)
    _write_csv(
        output_dir / "structural_diagnostics.csv",
        structural_diagnostics,
    )
    write_html(output_dir / "comparison_summary.html", summary)
    print(
        json.dumps(
            {
                method: aggregate_by_method[method]["final_mAP"]
                for method in METHODS
            },
            indent=2,
        )
    )
    print(f"Saved {output_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
