import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize internal Adapter seeds")
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--run_prefix", default="emotic_ddp_internal_adapter_16shot_seed"
    )
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_internal_adapter_16shot_summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for seed in args.seeds:
        run_name = f"{args.run_prefix}{seed}"
        path = Path(args.output_root) / run_name / "evaluation_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.load(open(path, encoding="utf-8"))
        aggregate = data["aggregate"]
        final = data["tasks"][-1]["test"]
        rows.append(
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
    metrics = [key for key in rows[0] if key != "seed"]
    summary = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in metrics
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump({"runs": rows, "aggregate": summary}, fp, indent=2)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = "".join(f"<th>{escape(key)}</th>" for key in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{row[key]:.6f}</td>" for key in row) + "</tr>"
        for row in rows
    )
    aggregate_rows = "".join(
        f"<tr><td>{escape(key)}</td><td>{value['mean']:.6f}</td>"
        f"<td>{value['std']:.6f}</td></tr>"
        for key, value in summary.items()
    )
    (output_dir / "summary.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Internal Adapter</title>"
        "<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse;"
        "margin-bottom:24px}th,td{border:1px solid #ddd;padding:6px;"
        "text-align:right}</style><h1>16-shot Internal Adapter</h1>"
        f"<table><tr>{headers}</tr>{body}</table>"
        "<h2>Mean ± std</h2><table><tr><th>metric</th><th>mean</th>"
        f"<th>std</th></tr>{aggregate_rows}</table>",
        encoding="utf-8",
    )
    for key, value in summary.items():
        print(f"{key}: {value['mean']:.4f}±{value['std']:.4f}")


if __name__ == "__main__":
    main()
