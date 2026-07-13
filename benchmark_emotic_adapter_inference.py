import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_emotic_ddp_internal_adapter import build_model
from eval_emotic_prototype_fusion import load_prototype_model
from eval_emotic_threshold_sweep import rebuild_text_feature_cache
from src.helper_functions.emotic_loader import EMOTIC
from train_emotic_ddp_internal_adapter import eval_transform
from train_emotic_prototype_adapter import build_clip_transform, load_frozen_clip


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark EMOTIC Adapter inference")
    parser.add_argument(
        "--method", choices=("ddp", "internal", "external", "hybrid"), required=True
    )
    parser.add_argument("--ddp_checkpoint", required=True)
    parser.add_argument("--internal_adapter", required=True)
    parser.add_argument("--internal_gate_summary", required=True)
    parser.add_argument("--external_adapter", required=True)
    parser.add_argument("--data_root", default="./datasets/EMOTIC")
    parser.add_argument(
        "--clip_model_path", default="./pretrained/clip/ViT-B-16.pt"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("GPU benchmark requires CUDA")
    # Both reported external methods are DDP fusion systems, not the standalone
    # Prototype branch, so they include the original DDP forward cost.
    need_ddp = True
    need_external = args.method in ("external", "hybrid")
    modules = []
    ddp_model = None
    prototype_model = None
    clip_model = None
    ddp_image = None
    external_image = None
    internal_gates = None

    if need_ddp:
        checkpoint = torch.load(args.ddp_checkpoint, map_location="cpu")
        helper_args = argparse.Namespace(clip_model_path=args.clip_model_path)
        ddp_model = build_model(checkpoint, helper_args, device)
        ddp_model.load_state_dict(checkpoint["model"], strict=True)
        rebuild_text_feature_cache(ddp_model, 26)
        ddp_model.eval()
        modules.append(ddp_model)
        dataset = EMOTIC(
            args.data_root,
            train=False,
            eval_splits=("val",),
            transform=eval_transform(),
            input_mode="full",
        )
        ddp_image = dataset[0][0].unsqueeze(0).to(device)
        if args.method in ("internal", "hybrid"):
            internal = torch.load(args.internal_adapter, map_location="cpu")
            adapter_args = internal["args"]
            ddp_model.enable_feature_adapter(
                int(adapter_args["adapter_dim"]),
                float(adapter_args["residual_scale"]),
            )
            ddp_model.feature_adapter.load_state_dict(internal["model"], strict=True)
            with open(args.internal_gate_summary, encoding="utf-8") as fp:
                gate_summary = json.load(fp)
            task = gate_summary["tasks"][-1]
            ddp_model.feature_adapter.residual_scale = float(
                task["selection"]["task_alpha"]["alpha"]
            )
            internal_gates = torch.tensor(
                [row["gate"] for row in task["selection"]["class_gates"]],
                dtype=torch.bool,
                device=device,
            ).view(1, 1, -1)

    if need_external:
        clip_model = load_frozen_clip(args.clip_model_path, device)
        prototype_model, _, _ = load_prototype_model(
            args.external_adapter, device
        )
        modules.extend([clip_model, prototype_model])
        dataset = EMOTIC(
            args.data_root,
            train=False,
            eval_splits=("val",),
            transform=build_clip_transform(224),
            input_mode="full",
        )
        external_image = dataset[0][0].unsqueeze(0).to(device)

    def forward():
        outputs = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            if need_ddp:
                if args.method in ("ddp", "external"):
                    outputs.append(
                        ddp_model(ddp_image, cls_id=(0, 26), inference=True)
                    )
                else:
                    pooled, base, text, cls_features = (
                        ddp_model.extract_path_features(
                            ddp_image,
                            cls_id=(0, 26),
                            inference=True,
                            return_cls_features=True,
                        )
                    )
                    baseline_logits = base.reshape(base.shape[0], 2, 26)
                    adapted_logits = ddp_model.logits_from_path_features(
                        cls_features, base, text
                    )
                    outputs.append(
                        torch.where(
                            internal_gates, adapted_logits, baseline_logits
                        )
                    )
            if need_external:
                features = clip_model.encode_image(external_image)
                outputs.append(prototype_model(features)[0])
        return outputs

    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        forward()
        torch.cuda.synchronize()
        latencies.append(1000.0 * (time.perf_counter() - start))

    loaded_parameters = sum(parameter_count(module) for module in modules)
    adapter_parameters = 0
    if ddp_model is not None and ddp_model.feature_adapter is not None:
        adapter_parameters = parameter_count(ddp_model.feature_adapter)
    if prototype_model is not None:
        adapter_parameters += parameter_count(prototype_model)
    result = {
        "method": args.method,
        "batch_size": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        "latency_ms_min": float(np.min(latencies)),
        "latency_ms_max": float(np.max(latencies)),
        "peak_memory_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_memory_mib": float(
            torch.cuda.max_memory_reserved() / 2**20
        ),
        "loaded_parameters": int(loaded_parameters),
        "adapter_parameters_loaded": int(adapter_parameters),
        "internal_class_gate": bool(args.method in ("internal", "hybrid")),
        "extra_image_encoder": int(need_external),
        "timing_scope": "GPU forward only; preprocessing and host-to-device excluded",
        "device": torch.cuda.get_device_name(device),
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
