import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ddp_internal_adapter import (
    CORRECTION_MODES,
    SharedResidualFeatureAdapter,
    feature_logit_correction,
)
from evaluation_metrics import mAP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-only transfer of external Prototype Adapter weights"
    )
    parser.add_argument(
        "--external_pattern",
        default=(
            "./output/emotic_prototype_adapter_base5_16shot_seed{seed}/"
            "best_adapter.pth"
        ),
    )
    parser.add_argument(
        "--val_cache",
        default=(
            "./output/emotic_ddp_internal_adapter_16shot_seed0/"
            "task0_val_path_features.pt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./output/emotic_ddp_internal_transfer_screen",
    )
    parser.add_argument(
        "--feature_key",
        choices=("pooled_features", "path_features", "cls_features"),
        default="pooled_features",
    )
    parser.add_argument(
        "--correction_mode",
        choices=CORRECTION_MODES,
        default="linear_residual",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--residual_scales",
        nargs="+",
        type=float,
        default=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1),
    )
    parser.add_argument("--minimum_val_gain", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_external(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"]
    down = state["down.weight"]
    up = state["up.weight"]
    if down.shape[1] != up.shape[0] or down.shape[0] != up.shape[1]:
        raise RuntimeError(f"Incompatible external Adapter shapes in {path}")
    adapter = SharedResidualFeatureAdapter(
        feature_dim=down.shape[1],
        bottleneck_dim=down.shape[0],
        residual_scale=0.0,
    ).to(device)
    adapter.load_state_dict({"down.weight": down, "up.weight": up}, strict=True)
    adapter.eval()
    return checkpoint, adapter


def validation_map(
    adapter,
    payload,
    feature_key,
    scale,
    batch_size,
    device,
    correction_mode="linear_residual",
):
    if feature_key not in payload:
        raise KeyError(f"Feature cache does not contain '{feature_key}'")
    adapter.residual_scale = float(scale)
    loader = DataLoader(
        TensorDataset(
            payload[feature_key], payload["base_path_logits"]
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    text_features = payload["text_features"].to(device).float()
    scores = []
    ratios = []
    with torch.no_grad():
        for pooled, base_logits in loader:
            pooled = pooled.to(device).float()
            base_logits = base_logits.to(device).float()
            adapted, original = adapter(pooled)
            correction = feature_logit_correction(
                adapted,
                original,
                text_features,
                mode=correction_mode,
                logit_scale=100.0,
            )
            logits = (base_logits + correction).reshape(
                pooled.shape[0], 2, pooled.shape[1] // 2
            )
            scores.append(torch.softmax(logits, dim=1)[:, 1, :].cpu())
            ratios.append(
                (
                    (adapted - original).norm(dim=-1)
                    / original.norm(dim=-1).clamp_min(1e-12)
                ).cpu()
            )
    scores = torch.cat(scores)
    targets = payload.get("labels")
    if targets is None:
        targets = payload.get("targets")
    if targets is None:
        raise KeyError("Feature cache contains neither 'labels' nor 'targets'")
    targets = targets[:, : scores.shape[1]]
    score, _ = mAP(targets.numpy(), scores.numpy())
    ratios = torch.cat(ratios)
    return float(score), float(ratios.mean()), float(ratios.max())


def write_html(path, rows, correction_mode="linear_residual"):
    headers = "".join(f"<th>{escape(key)}</th>" for key in rows[0])
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row[key]))}</td>" for key in row)
        + "</tr>"
        for row in rows
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Adapter Transfer</title>"
        "<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:right}</style>"
        "<h1>Prototype-to-DDP Validation Screen</h1>"
        f"<p>Correction mode: {escape(correction_mode)}</p>"
        f"<table><tr>{headers}</tr>"
        f"{body}</table>",
        encoding="utf-8",
    )


def select_stable_scale(aggregate, minimum_val_gain):
    eligible = [
        row
        for row in aggregate
        if row["residual_scale"] > 0.0
        and row["mean_val_gain"] > minimum_val_gain
        and row["all_seeds_positive"]
    ]
    unconstrained_best = max(
        aggregate, key=lambda row: row["mean_val_mAP"]
    )
    if eligible:
        return (
            max(eligible, key=lambda row: row["mean_val_mAP"]),
            unconstrained_best,
            True,
        )
    identity = next(row for row in aggregate if row["residual_scale"] == 0.0)
    return identity, unconstrained_best, False


def main():
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.val_cache, map_location="cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    adapters = {}
    rows = []
    for seed in args.seeds:
        path = args.external_pattern.format(seed=seed)
        checkpoint, adapter = load_external(path, device)
        checkpoints[seed] = checkpoint
        adapters[seed] = adapter
        identity_map = None
        for scale in args.residual_scales:
            score, ratio_mean, ratio_max = validation_map(
                adapter,
                payload,
                args.feature_key,
                scale,
                args.batch_size,
                device,
                correction_mode=args.correction_mode,
            )
            if scale == 0.0:
                identity_map = score
            rows.append(
                {
                    "seed": seed,
                    "residual_scale": scale,
                    "val_mAP": score,
                    "val_gain": None if identity_map is None else score - identity_map,
                    "residual_ratio_mean": ratio_mean,
                    "residual_ratio_max": ratio_max,
                }
            )

    identity = np.mean(
        [row["val_mAP"] for row in rows if row["residual_scale"] == 0.0]
    )
    aggregate = []
    for scale in args.residual_scales:
        scale_rows = [row for row in rows if row["residual_scale"] == scale]
        values = [row["val_mAP"] for row in scale_rows]
        gains = [value - identity for value in values]
        aggregate.append(
            {
                "residual_scale": scale,
                "mean_val_mAP": float(np.mean(values)),
                "std_val_mAP": float(np.std(values)),
                "mean_val_gain": float(np.mean(gains)),
                "minimum_seed_gain": float(np.min(gains)),
                "all_seeds_positive": bool(np.min(gains) > 0.0),
            }
        )
    best, unconstrained_best, passes = select_stable_scale(
        aggregate, args.minimum_val_gain
    )
    selected_scale = float(best["residual_scale"])
    for seed in args.seeds:
        adapter = adapters[seed]
        adapter.residual_scale = selected_scale
        external = checkpoints[seed]
        torch.save(
            {
                "model": adapter.state_dict(),
                "epoch": external.get("epoch"),
                "classnames": external["classnames"],
                "args": {
                    "adapter_dim": adapter.bottleneck_dim,
                    "residual_scale": selected_scale,
                    "correction_mode": args.correction_mode,
                },
                "transfer": {
                    "source": args.external_pattern.format(seed=seed),
                    "selection_split": "val",
                    "selected_scale": selected_scale,
                    "passes_val_gate": passes,
                    "correction_mode": args.correction_mode,
                },
            },
            output_dir / f"transferred_adapter_seed{seed}.pth",
        )

    summary = {
        "selection_split": "val",
        "test_used": False,
        "correction_mode": args.correction_mode,
        "minimum_val_gain": args.minimum_val_gain,
        "identity_val_mAP": float(identity),
        "best": best,
        "unconstrained_best": unconstrained_best,
        "passes_val_gate": passes,
        "aggregate": aggregate,
        "rows": rows,
        "args": vars(args),
    }
    with open(output_dir / "transfer_screen_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    with open(
        output_dir / "transfer_screen_summary.csv", "w", newline="", encoding="utf-8"
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    write_html(
        output_dir / "transfer_screen_summary.html",
        aggregate,
        correction_mode=args.correction_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
