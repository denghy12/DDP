import csv
import json
from html import escape
from pathlib import Path


LRS = ("1e-5", "3e-5", "1e-4")
LOSSES = ("balanced", "unweighted")
MINIMUM_VAL_GAIN = 0.1


def main():
    output_root = Path("./output")
    rows = []
    for loss_balance in LOSSES:
        for lr in LRS:
            run_name = (
                "emotic_ddp_internal_screen_seed0_dim16_scale001_id1_"
                f"lr{lr}_{loss_balance}"
            )
            path = output_root / run_name / "training_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
            diagnostics = data["best_diagnostics"]
            rows.append(
                {
                    "run_name": run_name,
                    "lr": lr,
                    "loss_balance": loss_balance,
                    "best_epoch": data["best_epoch"],
                    "best_step": data["best_step"],
                    "identity_val_mAP": data["zero_shot_val_mAP"],
                    "best_val_mAP": data["best_val_mAP"],
                    "best_val_gain": data["best_val_gain"],
                    "adaptation_selected": data["adaptation_selected"],
                    "residual_ratio_mean": diagnostics["residual_ratio_mean"],
                    "residual_ratio_max": diagnostics["residual_ratio_max"],
                    "passes_val_gate": data["best_val_gain"] > MINIMUM_VAL_GAIN,
                }
            )
    rows.sort(key=lambda row: row["best_val_gain"], reverse=True)
    output_dir = output_root / "emotic_ddp_internal_adapter_screen_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_split": "val",
        "test_used": False,
        "minimum_val_gain": MINIMUM_VAL_GAIN,
        "best_run": rows[0],
        "any_run_passes": any(row["passes_val_gate"] for row in rows),
        "rows": rows,
    }
    with open(output_dir / "screen_summary.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    with open(
        output_dir / "screen_summary.csv", "w", newline="", encoding="utf-8"
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = "".join(f"<th>{escape(key)}</th>" for key in rows[0])
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in row)
        + "</tr>"
        for row in rows
    )
    (output_dir / "screen_summary.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Internal Adapter Screen</title>"
        "<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:right}</style>"
        f"<h1>Seed0 Validation-only Screen</h1><table><tr>{headers}</tr>"
        f"{body}</table>",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
