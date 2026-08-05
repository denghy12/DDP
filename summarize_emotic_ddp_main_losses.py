"""Summarize fixed-last DDP main BCE versus ASL experiments."""

import argparse
import csv
import json
import statistics
from html import escape
from pathlib import Path


LOSSES = ("two_way_bce", "asl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_main_loss_comparison",
    )
    return parser.parse_args()


def mean_std(values):
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }


def eval_dir(output_root, loss_name, seed, split):
    name = f"emotic_b5c3_ddp_main_{loss_name}_seed{seed}_{split}_threshold050"
    return Path(output_root) / name


def detail_path(directory):
    name = directory.name
    return directory / "detail" / f"{name}_per_class_task_table.json"


def forgetting_from_detail(table_rows):
    values = []
    for row in table_rows:
        introduction = int(row["class_task"])
        if introduction >= 7:
            continue
        history = [
            float(row[f"task{task}_ap"])
            for task in range(introduction, 8)
            if row.get(f"task{task}_ap") is not None
        ]
        if len(history) > 1:
            values.append(100.0 * (max(history) - history[-1]))
    return float(statistics.mean(values)) if values else 0.0


def load_split(output_root, loss_name, seed, split):
    directory = eval_dir(output_root, loss_name, seed, split)
    summary_path = directory / "evaluation_summary.json"
    table_path = detail_path(directory)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(table_path.read_text(encoding="utf-8"))
    rows = summary["rows"]
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 task rows in {summary_path}")
    protocol = summary.get("training_protocol", {})
    if protocol.get("ddp_main_classification_loss") != loss_name:
        raise RuntimeError(f"Loss protocol mismatch in {summary_path}")
    if int(protocol.get("seed", -1)) != seed:
        raise RuntimeError(f"Seed protocol mismatch in {summary_path}")
    class_rows = detail["table_rows"]
    return {
        "split": split,
        "loss": loss_name,
        "seed": seed,
        "average_mAP": float(statistics.mean(row["mAP"] for row in rows)),
        "final_mAP": float(rows[-1]["mAP"]),
        "final_cF1": float(rows[-1]["cF1"]),
        "final_oF1": float(rows[-1]["oF1"]),
        "forgetting": forgetting_from_detail(class_rows),
        "tasks": rows,
        "classes": class_rows,
        "protocol": protocol,
        "summary_path": str(summary_path),
    }


def aggregate_runs(runs):
    metrics = (
        "average_mAP",
        "final_mAP",
        "final_cF1",
        "final_oF1",
        "forgetting",
    )
    return {
        metric: mean_std([run[metric] for run in runs]) for metric in metrics
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_frequency_groups(protocol):
    counts = protocol.get("train_positive_counts") or {}
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    sizes = (9, 9, max(0, len(ordered) - 18))
    groups = {}
    cursor = 0
    for name, size in zip(("head", "middle", "tail"), sizes):
        groups[name] = ordered[cursor : cursor + size]
        cursor += size
    return groups


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_runs = {
        split: {
            loss: [
                load_split(args.output_root, loss, seed, split)
                for seed in args.seeds
            ]
            for loss in LOSSES
        }
        for split in ("val", "test")
    }

    methods = {
        split: {
            loss: aggregate_runs(all_runs[split][loss]) for loss in LOSSES
        }
        for split in ("val", "test")
    }
    paired = {}
    for split in ("val", "test"):
        paired[split] = {}
        for metric in (
            "average_mAP",
            "final_mAP",
            "final_cF1",
            "final_oF1",
            "forgetting",
        ):
            differences = [
                asl[metric] - bce[metric]
                for bce, asl in zip(
                    all_runs[split]["two_way_bce"],
                    all_runs[split]["asl"],
                )
            ]
            paired[split][metric] = {
                **mean_std(differences),
                "per_seed": {
                    str(seed): float(value)
                    for seed, value in zip(args.seeds, differences)
                },
            }

    task_rows = []
    class_rows = []
    frequency_rows = []
    for split in ("val", "test"):
        for loss in LOSSES:
            runs = all_runs[split][loss]
            for task in range(8):
                task_rows.append(
                    {
                        "split": split,
                        "loss": loss,
                        "task": task,
                        "seen_classes": runs[0]["tasks"][task]["seen_classes"],
                        "mAP_mean": mean_std(
                            [run["tasks"][task]["mAP"] for run in runs]
                        )["mean"],
                        "mAP_std": mean_std(
                            [run["tasks"][task]["mAP"] for run in runs]
                        )["std"],
                        "cF1_mean": mean_std(
                            [run["tasks"][task]["cF1"] for run in runs]
                        )["mean"],
                        "oF1_mean": mean_std(
                            [run["tasks"][task]["oF1"] for run in runs]
                        )["mean"],
                    }
                )

            names = [row["class_name"] for row in runs[0]["classes"]]
            final_ap = {}
            for class_id, class_name in enumerate(names):
                values = [
                    100.0 * float(run["classes"][class_id]["task7_ap"])
                    for run in runs
                ]
                final_ap[class_name] = values
                stats = mean_std(values)
                class_rows.append(
                    {
                        "split": split,
                        "loss": loss,
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_task": runs[0]["classes"][class_id]["class_task"],
                        "train_positive_count": runs[0]["protocol"].get(
                            "train_positive_counts", {}
                        ).get(class_name),
                        "final_ap_mean": stats["mean"],
                        "final_ap_std": stats["std"],
                    }
                )

            groups = build_frequency_groups(runs[0]["protocol"])
            for group, class_names in groups.items():
                seed_values = [
                    statistics.mean(final_ap[name][seed_index] for name in class_names)
                    for seed_index in range(len(args.seeds))
                ]
                stats = mean_std(seed_values)
                frequency_rows.append(
                    {
                        "split": split,
                        "loss": loss,
                        "group": group,
                        "classes": " | ".join(class_names),
                        "mAP_mean": stats["mean"],
                        "mAP_std": stats["std"],
                    }
                )

    summary = {
        "protocol": {
            "comparison": "DDP main two-way BCE versus ASL",
            "checkpoint_rule": "fixed_last_epoch",
            "validation_role": "reporting_only",
            "primary_split": "test",
            "threshold": 0.5,
            "temperature": "tau2 schedule: 1.0 to 2.0, gamma 0.7",
            "seeds": list(args.seeds),
            "loss_w": 0.03,
            "reduction": "sum",
            "asl": {
                "gamma_neg": 9.8,
                "gamma_pos": 0.0,
                "clip": 0.05,
            },
        },
        "methods": methods,
        "paired_asl_minus_bce": paired,
        "runs": {
            split: {
                loss: [
                    {
                        key: run[key]
                        for key in (
                            "seed",
                            "average_mAP",
                            "final_mAP",
                            "final_cF1",
                            "final_oF1",
                            "forgetting",
                            "summary_path",
                        )
                    }
                    for run in all_runs[split][loss]
                ]
                for loss in LOSSES
            }
            for split in ("val", "test")
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    comparison_rows = []
    for split in ("val", "test"):
        for loss in LOSSES:
            row = {"split": split, "loss": loss}
            for metric, stats in methods[split][loss].items():
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            comparison_rows.append(row)
    write_csv(output_dir / "comparison_summary.csv", comparison_rows)
    write_csv(output_dir / "per_task_results.csv", task_rows)
    write_csv(output_dir / "per_class_results.csv", class_rows)
    write_csv(output_dir / "frequency_group_results.csv", frequency_rows)

    html_rows = []
    for row in comparison_rows:
        html_rows.append(
            "<tr>"
            f"<td>{escape(row['split'])}</td>"
            f"<td>{escape(row['loss'])}</td>"
            f"<td>{row['average_mAP_mean']:.4f} ± {row['average_mAP_std']:.4f}</td>"
            f"<td>{row['final_mAP_mean']:.4f} ± {row['final_mAP_std']:.4f}</td>"
            f"<td>{row['final_cF1_mean']:.4f} ± {row['final_cF1_std']:.4f}</td>"
            f"<td>{row['final_oF1_mean']:.4f} ± {row['final_oF1_std']:.4f}</td>"
            f"<td>{row['forgetting_mean']:.4f} ± {row['forgetting_std']:.4f}</td>"
            "</tr>"
        )
    gain = paired["test"]["final_mAP"]
    html = (
        "<!doctype html><meta charset='utf-8'><title>DDP Main ASL</title>"
        "<style>body{font-family:Arial;margin:28px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d9deea;padding:9px;text-align:right}"
        "th{background:#416fbd;color:#fff}</style>"
        "<h1>EMOTIC B5-C3: DDP Main BCE vs ASL</h1>"
        "<p>Fixed-last checkpoints · validation reporting only · threshold 0.50 · "
        "same optimizer and loss scale.</p>"
        f"<p><b>Paired test final-mAP change:</b> {gain['mean']:+.4f} ± "
        f"{gain['std']:.4f}</p>"
        "<table><tr><th>Split</th><th>Main loss</th><th>Average mAP</th>"
        "<th>Final mAP</th><th>Final cF1</th><th>Final oF1</th>"
        "<th>Forgetting</th></tr>"
        + "".join(html_rows)
        + "</table>"
    )
    (output_dir / "comparison_summary.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary["methods"]["test"], ensure_ascii=False, indent=2))
    print(f"Saved {output_dir}")


if __name__ == "__main__":
    main()
