import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluation_metrics import mAP
from prototype_adapter import ResidualPrototypeAdapter


class GlobalLogitCalibrator(nn.Module):
    """One temperature and one bias, fitted on validation labels only."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        temperature = self.log_temperature.clamp(-5.0, 5.0).exp()
        return logits / temperature + self.bias

    @property
    def temperature(self):
        return self.log_temperature.detach().clamp(-5.0, 5.0).exp().item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate and fuse task7 DDP and Prototype Adapter scores"
    )
    parser.add_argument(
        "--ddp_scores",
        default=(
            "./output/emotic_b5c3_ddp_semantic_tau2_threshold050/"
            "task7_scores.pt"
        ),
    )
    parser.add_argument(
        "--prototype_checkpoint",
        default=(
            "./output/emotic_prototype_adapter_base5_balanced/"
            "best_adapter.pth"
        ),
    )
    parser.add_argument(
        "--zero_shot_prototype",
        action="store_true",
        help=(
            "Use the checkpoint's fixed text prototypes with an identity "
            "adapter; learned adapter weights are ignored"
        ),
    )
    parser.add_argument(
        "--val_cache",
        default="./output/emotic_clip_feature_cache/val_full_224_vitb16.pt",
    )
    parser.add_argument(
        "--test_cache",
        default="./output/emotic_clip_feature_cache/test_full_224_vitb16.pt",
    )
    parser.add_argument(
        "--output_dir", default="./output/emotic_prototype_fusion_task7"
    )
    parser.add_argument("--seen_classes", type=int, default=26)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--beta_step", type=float, default=0.02)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--calibration_max_iter", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_feature_cache(path):
    payload = torch.load(path, map_location="cpu")
    for key in ("features", "labels", "metadata"):
        if key not in payload:
            raise RuntimeError(f"Missing '{key}' in feature cache: {path}")
    features = payload["features"].float()
    labels = payload["labels"].float()
    if features.shape[0] != labels.shape[0]:
        raise RuntimeError(f"Feature/label count mismatch in {path}")
    return features, labels, payload["metadata"]


def load_prototype_model(path, device, zero_shot=False):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    positive = state["positive_prototypes"]
    negative = state["negative_prototypes"]
    adapter_dim = state["down.weight"].shape[0]
    checkpoint_args = checkpoint.get("args", {})
    model = ResidualPrototypeAdapter(
        positive,
        negative,
        bottleneck_dim=adapter_dim,
        residual_scale=float(checkpoint_args.get("residual_scale", 0.1)),
        initial_logit_scale=float(checkpoint_args.get("initial_logit_scale", 10.0)),
    )
    if not zero_shot:
        model.load_state_dict(state)
    model.to(device).eval()
    classnames = checkpoint.get("classnames")
    if classnames is None:
        raise RuntimeError("Prototype checkpoint does not contain classnames")
    return model, list(classnames), checkpoint


def prototype_logits(model, features, batch_size, device):
    loader = DataLoader(
        TensorDataset(features),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    outputs = []
    with torch.no_grad():
        for (feature_batch,) in loader:
            logits, _, _ = model(feature_batch.to(device))
            outputs.append(logits.cpu())
    return torch.cat(outputs, dim=0)


def fit_calibrator(logits, targets, seen_classes, max_iter, device):
    calibrator = GlobalLogitCalibrator().to(device)
    logits = logits[:, :seen_classes].to(device)
    targets = targets[:, :seen_classes].to(device)
    optimizer = torch.optim.LBFGS(
        calibrator.parameters(),
        lr=0.5,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(calibrator(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        loss = F.binary_cross_entropy_with_logits(calibrator(logits), targets)
    return {
        "model": calibrator,
        "temperature": calibrator.temperature,
        "bias": calibrator.bias.detach().item(),
        "val_bce": loss.item(),
    }


def binary_metrics(targets, scores, threshold):
    targets = targets.bool()
    predictions = scores.ge(threshold)
    tp = (predictions & targets).sum(dim=0).float()
    fp = (predictions & ~targets).sum(dim=0).float()
    fn = (~predictions & targets).sum(dim=0).float()
    class_precision = tp / (tp + fp).clamp_min(1.0)
    class_recall = tp / (tp + fn).clamp_min(1.0)
    class_f1 = (
        2 * class_precision * class_recall
        / (class_precision + class_recall).clamp_min(1e-12)
    )
    overall_precision = tp.sum() / (tp + fp).sum().clamp_min(1.0)
    overall_recall = tp.sum() / (tp + fn).sum().clamp_min(1.0)
    overall_f1 = (
        2
        * overall_precision
        * overall_recall
        / (overall_precision + overall_recall).clamp_min(1e-12)
    )
    return {
        "cPrecision": 100 * class_precision.mean().item(),
        "cRecall": 100 * class_recall.mean().item(),
        "cF1": 100 * class_f1.mean().item(),
        "oPrecision": 100 * overall_precision.item(),
        "oRecall": 100 * overall_recall.item(),
        "oF1": 100 * overall_f1.item(),
    }


def score_metrics(targets, scores, threshold, classnames):
    map_score, per_class_ap = mAP(targets.numpy(), scores.numpy())
    return {
        "samples": int(targets.shape[0]),
        "mAP": float(map_score),
        "threshold": float(threshold),
        **binary_metrics(targets, scores, threshold),
        "per_class_ap": {
            name: 100 * float(per_class_ap[index])
            for index, name in enumerate(classnames)
        },
    }


def threshold_values(args):
    count = int(
        round((args.threshold_max - args.threshold_min) / args.threshold_step)
    )
    return [
        args.threshold_min + index * args.threshold_step
        for index in range(count + 1)
    ]


def select_threshold(targets, scores, args):
    best = None
    rows = []
    for threshold in threshold_values(args):
        metrics = binary_metrics(targets, scores, threshold)
        objective = 0.5 * (metrics["cF1"] + metrics["oF1"])
        row = {"threshold": threshold, "objective": objective, **metrics}
        rows.append(row)
        if best is None or objective > best["objective"]:
            best = row
    return best, rows


def beta_values(step):
    if not 0 < step <= 1:
        raise ValueError("beta_step must be in (0, 1]")
    count = int(round(1.0 / step))
    values = [min(1.0, index * step) for index in range(count + 1)]
    if values[-1] != 1.0:
        values.append(1.0)
    return sorted(set(values))


def select_beta(targets, ddp_scores, prototype_scores, step):
    best = None
    rows = []
    for beta in beta_values(step):
        fused = (1 - beta) * ddp_scores + beta * prototype_scores
        map_score, _ = mAP(targets.numpy(), fused.numpy())
        row = {"beta": beta, "val_mAP": float(map_score)}
        rows.append(row)
        if best is None or row["val_mAP"] > best["val_mAP"]:
            best = row
    return best, rows


def split_metrics(
    scores,
    targets,
    val_count,
    threshold,
    classnames,
):
    splits = {
        "val": (slice(0, val_count)),
        "test": (slice(val_count, None)),
    }
    return {
        name: score_metrics(targets[index], scores[index], threshold, classnames)
        for name, index in splits.items()
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    ddp_payload = torch.load(args.ddp_scores, map_location="cpu")
    ddp_scores = ddp_payload["scores"].float()
    ddp_targets = ddp_payload["targets"].float()
    val_features, val_labels, val_metadata = load_feature_cache(args.val_cache)
    test_features, test_labels, test_metadata = load_feature_cache(args.test_cache)
    if val_metadata.get("classnames") != test_metadata.get("classnames"):
        raise RuntimeError("Validation and test cache class orders differ")

    features = torch.cat([val_features, test_features], dim=0)
    cache_targets = torch.cat([val_labels, test_labels], dim=0)
    val_count = val_features.shape[0]
    if ddp_scores.shape != cache_targets.shape:
        raise RuntimeError(
            f"DDP score shape {tuple(ddp_scores.shape)} does not match "
            f"cache targets {tuple(cache_targets.shape)}"
        )
    if not torch.equal(ddp_targets, cache_targets):
        mismatch = ddp_targets.ne(cache_targets).sum().item()
        raise RuntimeError(
            f"DDP targets and Prototype cache order differ at {mismatch} entries"
        )

    model, classnames, checkpoint = load_prototype_model(
        args.prototype_checkpoint, device, args.zero_shot_prototype
    )
    if classnames != val_metadata.get("classnames"):
        raise RuntimeError("Prototype checkpoint and cache class orders differ")
    if args.seen_classes != len(classnames):
        raise ValueError(
            f"Task7 expects {len(classnames)} seen classes, got {args.seen_classes}"
        )
    raw_prototype_logits = prototype_logits(
        model, features, args.batch_size, device
    )

    calibration = fit_calibrator(
        raw_prototype_logits[:val_count],
        cache_targets[:val_count],
        args.seen_classes,
        args.calibration_max_iter,
        device,
    )
    calibrator = calibration.pop("model")
    calibrator.eval()
    with torch.no_grad():
        calibrated_prototype_scores = torch.sigmoid(
            calibrator(raw_prototype_logits.to(device))
        ).cpu()

    best_beta, beta_rows = select_beta(
        cache_targets[:val_count],
        ddp_scores[:val_count],
        calibrated_prototype_scores[:val_count],
        args.beta_step,
    )
    beta = best_beta["beta"]
    fused_scores = (1 - beta) * ddp_scores + beta * calibrated_prototype_scores

    ddp_threshold, ddp_threshold_rows = select_threshold(
        cache_targets[:val_count], ddp_scores[:val_count], args
    )
    prototype_threshold, prototype_threshold_rows = select_threshold(
        cache_targets[:val_count],
        calibrated_prototype_scores[:val_count],
        args,
    )
    fusion_threshold, fusion_threshold_rows = select_threshold(
        cache_targets[:val_count], fused_scores[:val_count], args
    )

    results = {
        "ddp": split_metrics(
            ddp_scores,
            cache_targets,
            val_count,
            ddp_threshold["threshold"],
            classnames,
        ),
        "prototype": split_metrics(
            calibrated_prototype_scores,
            cache_targets,
            val_count,
            prototype_threshold["threshold"],
            classnames,
        ),
        "fusion": split_metrics(
            fused_scores,
            cache_targets,
            val_count,
            fusion_threshold["threshold"],
            classnames,
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "inputs": {
            "ddp_scores": args.ddp_scores,
            "prototype_checkpoint": args.prototype_checkpoint,
            "prototype_checkpoint_epoch": checkpoint.get("epoch"),
            "zero_shot_prototype": args.zero_shot_prototype,
            "val_cache": args.val_cache,
            "test_cache": args.test_cache,
        },
        "alignment": {
            "val_samples": int(val_count),
            "test_samples": int(test_features.shape[0]),
            "total_samples": int(features.shape[0]),
            "targets_equal": True,
        },
        "calibration": calibration,
        "selection": {
            "split": "val",
            "beta": best_beta,
            "thresholds": {
                "ddp": ddp_threshold,
                "prototype": prototype_threshold,
                "fusion": fusion_threshold,
            },
        },
        "results": results,
        "beta_sweep": beta_rows,
        "threshold_sweeps": {
            "ddp": ddp_threshold_rows,
            "prototype": prototype_threshold_rows,
            "fusion": fusion_threshold_rows,
        },
        "args": vars(args),
    }
    with open(output_dir / "fusion_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    torch.save(
        {
            "scores": fused_scores,
            "ddp_scores": ddp_scores,
            "prototype_scores": calibrated_prototype_scores,
            "targets": cache_targets,
            "val_count": val_count,
            "beta": beta,
            "threshold": fusion_threshold["threshold"],
        },
        output_dir / "fusion_scores.pt",
    )

    print(
        f"Aligned samples: val={val_count}, test={test_features.shape[0]}, "
        f"targets_equal=True"
    )
    print(
        f"Prototype calibration: T={calibration['temperature']:.6f}, "
        f"bias={calibration['bias']:.6f}, val_bce={calibration['val_bce']:.6f}"
    )
    print(
        f"Selected on val: beta={beta:.2f}, "
        f"fusion_threshold={fusion_threshold['threshold']:.2f}, "
        f"val_mAP={best_beta['val_mAP']:.4f}"
    )
    for name in ("ddp", "prototype", "fusion"):
        test = results[name]["test"]
        print(
            f"{name:10s} test: mAP={test['mAP']:.4f}, "
            f"cF1={test['cF1']:.4f}, oF1={test['oF1']:.4f}"
        )


if __name__ == "__main__":
    main()
