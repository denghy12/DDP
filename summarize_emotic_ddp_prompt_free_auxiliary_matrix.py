import csv
import json
from html import escape
from pathlib import Path


OUTPUT_ROOT = Path("./output")
OUTPUT_DIR = OUTPUT_ROOT / "emotic_ddp_prompt_free_auxiliary_matrix_comparison"

FORMULAS = (
    ("feature_difference", "Feature difference"),
    ("cosine_difference", "Cosine difference"),
    ("feature_correction", "Feature Correction"),
)

METHODS = {
    ("legacy_external", "full_base5", "feature_difference"): (
        "emotic_ddp_cls_full_base5_feature_difference_summary",
        "emotic_ddp_cls_full_base5_feature_difference_screen",
    ),
    ("legacy_external", "full_base5", "cosine_difference"): (
        "emotic_ddp_cls_full_base5_cosine_difference_summary",
        "emotic_ddp_cls_full_base5_cosine_difference_screen",
    ),
    ("legacy_external", "full_base5", "feature_correction"): (
        "emotic_ddp_cls_full_base5_feature_correction_summary",
        "emotic_ddp_cls_full_base5_feature_correction_screen",
    ),
    ("legacy_external", "16shot", "feature_difference"): (
        "emotic_ddp_cls_internal_transfer_16shot_summary",
        "emotic_ddp_cls_internal_transfer_screen",
    ),
    ("legacy_external", "16shot", "cosine_difference"): (
        "emotic_ddp_cls_cosine_difference_summary",
        "emotic_ddp_cls_cosine_difference_screen",
    ),
    ("legacy_external", "16shot", "feature_correction"): (
        "emotic_ddp_cls_feature_correction_summary",
        "emotic_ddp_cls_feature_correction_screen",
    ),
    ("ddp_owned_auxiliary", "full_base5", "feature_difference"): (
        "emotic_ddp_prompt_free_auxiliary_cls_feature_difference_summary",
        "emotic_ddp_prompt_free_auxiliary_cls_feature_difference_screen",
    ),
    ("ddp_owned_auxiliary", "full_base5", "cosine_difference"): (
        "emotic_ddp_prompt_free_auxiliary_cls_cosine_difference_summary",
        "emotic_ddp_prompt_free_auxiliary_cls_cosine_difference_screen",
    ),
    ("ddp_owned_auxiliary", "full_base5", "feature_correction"): (
        "emotic_ddp_prompt_free_auxiliary_cls_feature_correction_summary",
        "emotic_ddp_prompt_free_auxiliary_cls_feature_correction_screen",
    ),
    ("ddp_owned_auxiliary", "16shot", "feature_difference"): (
        "emotic_ddp_prompt_free_auxiliary_16shot_cls_feature_difference_summary",
        "emotic_ddp_prompt_free_auxiliary_16shot_cls_feature_difference_screen",
    ),
    ("ddp_owned_auxiliary", "16shot", "cosine_difference"): (
        "emotic_ddp_prompt_free_auxiliary_16shot_cls_cosine_difference_summary",
        "emotic_ddp_prompt_free_auxiliary_16shot_cls_cosine_difference_screen",
    ),
    ("ddp_owned_auxiliary", "16shot", "feature_correction"): (
        "emotic_ddp_prompt_free_auxiliary_16shot_cls_feature_correction_summary",
        "emotic_ddp_prompt_free_auxiliary_16shot_cls_feature_correction_screen",
    ),
}


def read_json(path, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.load(open(path, encoding="utf-8"))


def metric(summary, key):
    value = summary["aggregate"][key]
    return {"mean": float(value["mean"]), "std": float(value["std"])}


def method_row(source, training_data, formula, summary_dir, screen_dir):
    screen_path = OUTPUT_ROOT / screen_dir / "transfer_screen_summary.json"
    screen = read_json(screen_path, required=False)
    if screen is None:
        return {
            "source": source,
            "training_data": training_data,
            "formula": formula,
            "status": "missing_screen",
        }
    selection = {
        "split": screen.get("selection_split"),
        "test_used": screen.get("test_used"),
        "passes_val_gate": bool(screen.get("passes_val_gate")),
        "global_alpha": float(screen["best"]["residual_scale"]),
        "identity_val_mAP": float(screen["identity_val_mAP"]),
        "selected_val_mAP": float(screen["best"]["mean_val_mAP"]),
        "selected_val_gain": float(screen["best"]["mean_val_gain"]),
        "minimum_seed_gain": float(screen["best"]["minimum_seed_gain"]),
    }
    if not selection["passes_val_gate"]:
        return {
            "source": source,
            "training_data": training_data,
            "formula": formula,
            "status": "failed_val_gate",
            "selection": selection,
        }
    summary_path = OUTPUT_ROOT / summary_dir / "summary.json"
    summary = read_json(summary_path, required=False)
    if summary is None:
        return {
            "source": source,
            "training_data": training_data,
            "formula": formula,
            "status": "missing_test_summary",
            "selection": selection,
        }
    return {
        "source": source,
        "training_data": training_data,
        "formula": formula,
        "status": "complete",
        "selection": selection,
        "run_count": len(summary.get("runs", [])),
        "average_mAP": metric(summary, "average_mAP"),
        "average_mAP_gain": metric(summary, "average_mAP_gain"),
        "final_mAP": metric(summary, "final_mAP"),
        "final_mAP_gain": metric(summary, "final_mAP_gain"),
        "final_cF1": metric(summary, "final_cF1"),
        "final_oF1": metric(summary, "final_oF1"),
        "forgetting": metric(summary, "forgetting"),
        "runs": summary.get("runs", []),
        "inputs": {
            "summary": str(summary_path),
            "screen": str(screen_path),
        },
    }


def baseline_row():
    path = (
        OUTPUT_ROOT
        / "emotic_ddp_cls_full_base5_ddp_baseline"
        / "evaluation_summary.json"
    )
    data = read_json(path)
    aggregate = data["aggregate"]
    final = data["tasks"][-1]["test"]
    # Use the official saved DDP score baseline for mAP, matching every Adapter
    # summary's gain calculation. cF1/oF1 come from the explicit DDP-only run.
    return {
        "source": "ddp_baseline",
        "training_data": "none",
        "formula": "none",
        "status": "complete",
        "average_mAP": {
            "mean": float(aggregate["baseline_average_mAP"]),
            "std": 0.0,
        },
        "average_mAP_gain": {"mean": 0.0, "std": 0.0},
        "final_mAP": {
            "mean": float(aggregate["baseline_final_mAP"]),
            "std": 0.0,
        },
        "final_mAP_gain": {"mean": 0.0, "std": 0.0},
        "final_cF1": {"mean": float(final["cF1"]), "std": 0.0},
        "final_oF1": {"mean": float(final["oF1"]), "std": 0.0},
        "forgetting": {
            "mean": float(
                aggregate["forgetting"]["average_forgetting_old_classes"]
            ),
            "std": 0.0,
        },
        "inputs": {"summary": str(path)},
    }


def fmt(value):
    if value is None:
        return "—"
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def write_html(path, comparison):
    source_names = {
        "legacy_external": "旧外部 vanilla-CLIP Adapter",
        "ddp_owned_auxiliary": "新 DDP 内置无 Prompt Auxiliary",
    }
    training_names = {"full_base5": "Full Base5", "16shot": "Base5 16-shot"}
    formula_names = dict(FORMULAS)
    rows = []
    for row in comparison["rows"]:
        selection = row.get("selection", {})
        if row["status"] == "complete":
            metric_cells = "".join(
                f"<td>{escape(fmt(row[key]))}</td>"
                for key in (
                    "average_mAP",
                    "final_mAP",
                    "final_mAP_gain",
                    "final_cF1",
                    "final_oF1",
                    "forgetting",
                )
            )
        else:
            metric_cells = f"<td colspan='6'>{escape(row['status'])}</td>"
        rows.append(
            "<tr>"
            f"<td>{escape(source_names[row['source']])}</td>"
            f"<td>{escape(training_names[row['training_data']])}</td>"
            f"<td>{escape(formula_names[row['formula']])}</td>"
            f"<td>{selection.get('global_alpha', '—')}</td>"
            f"<td>{selection.get('selected_val_gain', '—')}</td>"
            + metric_cells
            + "</tr>"
        )
    baseline = comparison["baseline"]
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>DDP Auxiliary Adapter 2x3 Matrix</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin:18px 0}"
        "th,td{border:1px solid #ccd3df;padding:8px;text-align:right}"
        "th{background:#416fbd;color:white}td:first-child,th:first-child{text-align:left}"
        ".note{background:#f3f6fb;padding:12px}</style>"
        "<h1>DDP 内置 Auxiliary 与旧外部 Adapter：2×3 对照</h1>"
        "<p class='note'>每种方法仅在 task0 validation 上选择一个全局 α；"
        "无逐类 gate、无 task-specific α、无 test 选择。DDP baseline: "
        f"Final mAP {baseline['final_mAP']['mean']:.4f}, "
        f"cF1 {baseline['final_cF1']['mean']:.4f}, "
        f"oF1 {baseline['final_oF1']['mean']:.4f}.</p>"
        "<table><tr><th>Adapter 来源</th><th>训练数据</th><th>公式</th>"
        "<th>全局 α</th><th>Val gain</th><th>Average mAP</th>"
        "<th>Final mAP</th><th>相对 DDP</th><th>cF1</th><th>oF1</th>"
        "<th>Forgetting ↓</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def main():
    rows = [
        method_row(source, training_data, formula, *paths)
        for (source, training_data, formula), paths in METHODS.items()
    ]
    comparison = {
        "protocol": {
            "matrix": "2 training sets x 3 correction formulas x 2 Adapter sources",
            "training_sets": ["Base5 16-shot", "Full Base5"],
            "formulas": [name for name, _ in FORMULAS],
            "adapter_sources": ["legacy_external", "ddp_owned_auxiliary"],
            "selection_split": "task0 validation",
            "test_used_for_selection": False,
            "class_specific_gate": False,
            "task_specific_alpha": False,
            "external_score_fusion": False,
        },
        "baseline": baseline_row(),
        "rows": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "comparison_summary.json"
    csv_path = OUTPUT_DIR / "comparison_summary.csv"
    html_path = OUTPUT_DIR / "comparison_summary.html"
    json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "source": row["source"],
                "training_data": row["training_data"],
                "formula": row["formula"],
                "status": row["status"],
                "global_alpha": row.get("selection", {}).get("global_alpha"),
                "val_gain": row.get("selection", {}).get("selected_val_gain"),
                "average_mAP_mean": row.get("average_mAP", {}).get("mean"),
                "average_mAP_std": row.get("average_mAP", {}).get("std"),
                "final_mAP_mean": row.get("final_mAP", {}).get("mean"),
                "final_mAP_std": row.get("final_mAP", {}).get("std"),
                "final_mAP_gain": row.get("final_mAP_gain", {}).get("mean"),
                "final_cF1": row.get("final_cF1", {}).get("mean"),
                "final_oF1": row.get("final_oF1", {}).get("mean"),
                "forgetting": row.get("forgetting", {}).get("mean"),
            }
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_html(html_path, comparison)
    for row in rows:
        print(
            f"{row['source']:19s} {row['training_data']:10s} "
            f"{row['formula']:19s} {row['status']:16s} "
            f"final={fmt(row.get('final_mAP'))}"
        )
    print(f"Saved {json_path}")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()
