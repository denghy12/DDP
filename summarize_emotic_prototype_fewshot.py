import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "mAP",
    "base_mAP",
    "base_seen_mAP",
    "novel_mAP",
    "cF1",
    "oF1",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate EMOTIC Prototype Adapter few-shot seeds"
    )
    parser.add_argument("--protocol", choices=("all26", "base5"), required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output_root", default="./output")
    parser.add_argument(
        "--summary_dir", default="./output/emotic_prototype_fewshot_summary"
    )
    return parser.parse_args()


def mean_std(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def load_run(output_root, protocol, shots, seed):
    name = f"emotic_prototype_adapter_{protocol}_{shots}shot_seed{seed}"
    path = output_root / name / "evaluation_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fp:
        payload = json.load(fp)
    fewshot = payload.get("fewshot")
    if fewshot is None or int(fewshot["shots_per_class"]) != shots:
        raise RuntimeError(f"Invalid few-shot metadata in {path}")
    return name, payload


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []
    details = []
    for shots in args.shots:
        runs = [
            load_run(output_root, args.protocol, shots, seed)
            for seed in args.seeds
        ]
        row = {
            "protocol": args.protocol,
            "shots_per_class": shots,
            "seeds": ",".join(str(seed) for seed in args.seeds),
            "runs": len(runs),
        }
        sample_counts = [
            payload["fewshot"]["unique_training_samples"]
            for _, payload in runs
        ]
        row["unique_samples_mean"], row["unique_samples_std"] = mean_std(
            sample_counts
        )
        row["best_epoch_mean"], row["best_epoch_std"] = mean_std(
            [payload["best_epoch"] for _, payload in runs]
        )
        for metric in METRICS:
            mean, std = mean_std(
                [payload["best"]["test"].get(metric) for _, payload in runs]
            )
            row[f"test_{metric}_mean"] = mean
            row[f"test_{metric}_std"] = std
        rows.append(row)
        details.extend(
            {
                "run": name,
                "shots_per_class": shots,
                "seed": seed,
                "fewshot": payload["fewshot"],
                "best_epoch": payload["best_epoch"],
                "val": payload["best"]["val"],
                "test": payload["best"]["test"],
            }
            for seed, (name, payload) in zip(args.seeds, runs)
        )

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.protocol}_fewshot_summary"
    json_path = summary_dir / f"{stem}.json"
    csv_path = summary_dir / f"{stem}.csv"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {"protocol": args.protocol, "aggregate": rows, "runs": details},
            fp,
            indent=2,
            ensure_ascii=False,
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{args.protocol} few-shot held-out test results")
    for row in rows:
        print(
            f"K={row['shots_per_class']:2d} "
            f"samples={row['unique_samples_mean']:.1f} "
            f"mAP={row['test_mAP_mean']:.4f}±{row['test_mAP_std']:.4f} "
            f"cF1={row['test_cF1_mean']:.4f}±{row['test_cF1_std']:.4f} "
            f"oF1={row['test_oF1_mean']:.4f}±{row['test_oF1_std']:.4f}"
        )
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
