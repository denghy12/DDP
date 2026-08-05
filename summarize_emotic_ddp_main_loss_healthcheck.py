"""Summarize the Task-0 BCE/ASL DDP main-loss health check."""

import json
from html import escape
from pathlib import Path


def load_run(loss_name):
    run_name = f"emotic_ddp_main_healthcheck_{loss_name}_task0_seed0_v2"
    root = Path("output") / run_name
    diagnostics_path = root / "training_diagnostics.json"
    detail_path = root / "detail" / f"{run_name}_per_class_task_table.json"
    checkpoint_path = root / "checkpoints" / "task0.pth"
    for path in (diagnostics_path, detail_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    overall = detail["overall_rows"][-1]
    first = diagnostics["first_batch_health"][0]
    history = [
        row for row in diagnostics["epoch_history"] if int(row["task"]) == 0
    ]
    if len(history) != 30:
        raise RuntimeError(f"Expected 30 Task-0 epochs in {diagnostics_path}")
    return {
        "loss": loss_name,
        "task0_val_mAP": float(overall["mAP"]),
        "task0_val_cF1": float(overall["cF1"]),
        "task0_val_oF1": float(overall["oF1"]),
        "initial_raw_loss_per_label": float(first["raw_loss_per_label"]),
        "initial_prompt_gradient_norm": float(first["prompt_gradient_norm"]),
        "initial_margin_min": float(first["margin_min"]),
        "initial_margin_max": float(first["margin_max"]),
        "initial_easy_negative_fraction": float(
            first["easy_negative_fraction"]
        ),
        "epoch0_raw_loss_per_label": float(history[0]["raw_loss_per_label"]),
        "epoch29_raw_loss_per_label": float(history[-1]["raw_loss_per_label"]),
        "protocol": diagnostics["protocol"],
        "checkpoint": str(checkpoint_path),
    }


def main():
    output = Path("output/emotic_ddp_main_loss_healthcheck_comparison_v2")
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(loss) for loss in ("two_way_bce", "asl")]
    bce, asl = runs
    summary = {
        "protocol": {
            "task": 0,
            "seed": 0,
            "epochs": 30,
            "checkpoint_rule": "fixed_last_epoch",
            "training_time_eval_splits": ["val"],
            "test_used": False,
        },
        "runs": runs,
        "asl_minus_bce": {
            "task0_val_mAP": asl["task0_val_mAP"] - bce["task0_val_mAP"],
            "task0_val_cF1": asl["task0_val_cF1"] - bce["task0_val_cF1"],
            "task0_val_oF1": asl["task0_val_oF1"] - bce["task0_val_oF1"],
        },
    }
    (output / "healthcheck_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{escape(run['loss'])}</td>"
            f"<td>{run['task0_val_mAP']:.4f}</td>"
            f"<td>{run['task0_val_cF1']:.4f}</td>"
            f"<td>{run['task0_val_oF1']:.4f}</td>"
            f"<td>{run['initial_raw_loss_per_label']:.6f}</td>"
            f"<td>{run['initial_prompt_gradient_norm']:.6f}</td>"
            f"<td>{run['initial_easy_negative_fraction']:.4f}</td>"
            "</tr>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>DDP ASL Health Check</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:9px;text-align:right}"
        "th{background:#416fbd;color:white}</style>"
        "<h1>DDP main-loss Task-0 health check</h1>"
        "<p>30 fixed epochs · seed 0 · val only · no test access.</p>"
        "<table><tr><th>Loss</th><th>Val mAP</th><th>Val cF1</th>"
        "<th>Val oF1</th><th>Initial loss/label</th>"
        "<th>Initial gradient norm</th><th>Easy-negative fraction</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    (output / "healthcheck_summary.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary["asl_minus_bce"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
