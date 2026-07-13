import csv
import glob
import json
from html import escape
from pathlib import Path

import numpy as np


def load(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def method_from_runs(pattern, method):
    runs = [load(path) for path in sorted(glob.glob(pattern))]
    if len(runs) != 3:
        raise RuntimeError(f"Expected 3 runs for {pattern}, found {len(runs)}")
    values = {
        "average_mAP": [
            run["aggregate"]["methods"][method]["average_mAP"] for run in runs
        ],
        "final_mAP": [
            run["aggregate"]["methods"][method]["final_mAP"] for run in runs
        ],
        "final_cF1": [run["tasks"][-1]["results"][method]["test"]["cF1"] for run in runs],
        "final_oF1": [run["tasks"][-1]["results"][method]["test"]["oF1"] for run in runs],
        "forgetting": [
            run["aggregate"]["methods"][method]["forgetting"][
                "average_forgetting_old_classes"
            ]
            for run in runs
        ],
    }
    return values


def method_from_generic_summary(path):
    data = load(path)
    return {
        key: [run[key] for run in data["runs"]]
        for key in (
            "average_mAP",
            "final_mAP",
            "final_cF1",
            "final_oF1",
            "forgetting",
        )
    }


def add_row(rows, method, family, values, parameters, extra_encoder):
    row = {
        "method": method,
        "family": family,
        "runs": len(values["final_mAP"]),
        "additional_adapter_parameters": parameters,
        "extra_image_encoder": extra_encoder,
    }
    for metric, metric_values in values.items():
        mean, std = mean_std(metric_values)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    rows.append(row)


def external_condition(condition):
    data = load(
        "output/emotic_prototype_fusion_strict_summary/fusion_sweep_summary.json"
    )
    selected = [run for run in data["runs"] if run["condition"] == condition]
    if not selected:
        raise RuntimeError(f"No external fusion runs for {condition}")
    return {
        "average_mAP": [run["average_task_mAP"] for run in selected],
        "final_mAP": [run["final_mAP"] for run in selected],
        "final_cF1": [run["final_cF1"] for run in selected],
        "final_oF1": [run["final_oF1"] for run in selected],
        "forgetting": [run["forgetting"] for run in selected],
    }


def write_html(path, rows):
    headers = "".join(f"<th>{escape(key)}</th>" for key in rows[0])
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in row)
        + "</tr>"
        for row in rows
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Adapter Ablation</title>"
        "<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:right}</style>"
        f"<h1>EMOTIC Adapter Ablation</h1><table><tr>{headers}</tr>{body}</table>",
        encoding="utf-8",
    )


def main():
    rows = []
    gate_pattern = "output/emotic_ddp_cls_internal_gate_seed*/evaluation_summary.json"
    add_row(rows, "DDP", "baseline", method_from_runs(gate_pattern, "ddp"), 0, 0)
    add_row(
        rows,
        "Pooled transfer",
        "internal",
        method_from_generic_summary(
            "output/emotic_ddp_internal_transfer_16shot_summary/summary.json"
        ),
        131072,
        0,
    )
    add_row(
        rows,
        "CLS fixed alpha",
        "internal",
        method_from_generic_summary(
            "output/emotic_ddp_cls_internal_transfer_16shot_summary/summary.json"
        ),
        131072,
        0,
    )
    add_row(
        rows,
        "CLS task alpha",
        "internal",
        method_from_runs(gate_pattern, "task_alpha"),
        131072,
        0,
    )
    add_row(
        rows,
        "CLS class gate",
        "internal",
        method_from_runs(gate_pattern, "class_gate"),
        131072,
        0,
    )
    add_row(
        rows,
        "External 16-shot",
        "external",
        external_condition("16_shot"),
        131073,
        1,
    )
    add_row(
        rows,
        "External Full",
        "external",
        external_condition("full"),
        131073,
        1,
    )
    hybrid_pattern = "output/emotic_ddp_cls_external_hybrid_seed*/evaluation_summary.json"
    if len(glob.glob(hybrid_pattern)) == 3:
        add_row(
            rows,
            "CLS gate + External 16-shot",
            "hybrid",
            method_from_runs(hybrid_pattern, "binary_hybrid"),
            262145,
            1,
        )
    output_dir = Path("output/emotic_adapter_final_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ablation.json", "w", encoding="utf-8") as fp:
        json.dump({"rows": rows}, fp, indent=2, ensure_ascii=False)
    with open(output_dir / "ablation.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_html(output_dir / "ablation.html", rows)
    for row in rows:
        print(
            f"{row['method']:30s} final={row['final_mAP_mean']:.4f}±"
            f"{row['final_mAP_std']:.4f} avg={row['average_mAP_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
