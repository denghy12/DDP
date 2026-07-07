import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize strict zero/few/full-shot Prototype-DDP fusion"
    )
    parser.add_argument("--output_root", default="./output")
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_prototype_fusion_strict_summary",
    )
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def load_run(path):
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    method = data["aggregate"]["methods"]["binary_gate"]["test"]
    final = data["tasks"][-1]["results"]["binary_gate"]["test"]
    ddp = data["aggregate"]["methods"]["ddp"]["test"]
    return {
        "average_task_mAP": method["average_mAP"],
        "final_mAP": method["final_mAP"],
        "final_cF1": final["cF1"],
        "final_oF1": final["oF1"],
        "forgetting": method["forgetting"][
            "average_forgetting_old_classes"
        ],
        "ddp_average_task_mAP": ddp["average_mAP"],
        "ddp_final_mAP": ddp["final_mAP"],
    }


def mean_std(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def main():
    args = parse_args()
    root = Path(args.output_root)
    specs = [("zero_shot", None, None, "emotic_prototype_fusion_strict_zero_shot")]
    for shots in (1, 2, 4, 8, 16):
        for seed in (0, 1, 2):
            specs.append(
                (
                    f"{shots}_shot",
                    shots,
                    seed,
                    f"emotic_prototype_fusion_strict_base5_{shots}shot_seed{seed}",
                )
            )
    for seed in (0, 1, 2):
        specs.append(("full", None, seed, f"emotic_prototype_fusion_strict_full_seed{seed}"))

    runs = []
    missing = []
    for condition, shots, seed, run_name in specs:
        path = root / run_name / "fusion_all_tasks_summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        runs.append(
            {
                "condition": condition,
                "shots": shots,
                "seed": seed,
                "run_name": run_name,
                **load_run(path),
            }
        )
    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing runs:\n" + "\n".join(missing))

    conditions = []
    for name in ("zero_shot", "1_shot", "2_shot", "4_shot", "8_shot", "16_shot", "full"):
        selected = [row for row in runs if row["condition"] == name]
        if not selected:
            continue
        summary = {"condition": name, "runs": len(selected)}
        for key in (
            "average_task_mAP",
            "final_mAP",
            "final_cF1",
            "final_oF1",
            "forgetting",
        ):
            mean, std = mean_std(selected, key)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        conditions.append(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"runs": runs, "conditions": conditions, "missing": missing}
    with open(output_dir / "fusion_sweep_summary.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    if conditions:
        with open(output_dir / "fusion_sweep_summary.csv", "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(conditions[0]))
            writer.writeheader()
            writer.writerows(conditions)
    for row in conditions:
        print(
            f"{row['condition']:10s} n={row['runs']} "
            f"final mAP={row['final_mAP_mean']:.4f}±{row['final_mAP_std']:.4f} "
            f"avg mAP={row['average_task_mAP_mean']:.4f}±{row['average_task_mAP_std']:.4f}"
        )
    if missing:
        print(f"Missing {len(missing)} runs")


if __name__ == "__main__":
    main()
