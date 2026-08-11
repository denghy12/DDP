#!/usr/bin/env python3
"""Generate sample-aligned, sample-preserving CocoER head geometry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = (
    ROOT / "benchmarks" / "emotic_mlcil" / "methods" / "cocoer_ft"
    / "head_geometry.py"
)
GEOMETRY_SPEC = importlib.util.spec_from_file_location(
    "cocoer_head_geometry", GEOMETRY_PATH
)
GEOMETRY = importlib.util.module_from_spec(GEOMETRY_SPEC)
GEOMETRY_SPEC.loader.exec_module(GEOMETRY)
CONVERSION_NAME = GEOMETRY.CONVERSION_NAME
FALLBACK_SOURCE = GEOMETRY.FALLBACK_SOURCE
NATIVE_SOURCE = GEOMETRY.NATIVE_SOURCE
clip_box = GEOMETRY.clip_box
median_relative_geometry = GEOMETRY.median_relative_geometry
project_relative_geometry = GEOMETRY.project_relative_geometry
relative_geometry = GEOMETRY.relative_geometry
source_match = GEOMETRY.source_match


UPSTREAM_COMMIT = "dac8fc139e61b87f1bf0b27c581798df2a5a9d38"
BUFFALO_L_TREE_SHA256 = "50fa1383e97d137f2902b53de7b7305ffbd35eb4ae32135d95d1e25d5a9d9d3d"


def _tree_sha256(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"InsightFace model tree is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _actual_providers(model):
    session = getattr(model, "session", None)
    if session is None or not hasattr(session, "get_providers"):
        raise RuntimeError("InsightFace model does not expose its ONNX session")
    providers = list(session.get_providers())
    if not providers:
        raise RuntimeError("InsightFace ONNX session has no execution provider")
    return providers


def _require_device_provider(device: str, actual_providers):
    expected = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if not actual_providers or actual_providers[0] != expected:
        raise RuntimeError(
            f"CocoER requested {device}, but the actual ONNX providers are "
            f"{actual_providers}"
        )


def _scrfd_faces(detector, image):
    bboxes, _ = detector.detect(image, max_num=0, metric="default")
    height, width = image.shape[:2]
    return [
        candidate
        for candidate in (clip_box(width, height, row) for row in bboxes)
        if candidate is not None
    ]


def _faceanalysis_faces(app, image):
    height, width = image.shape[:2]
    return [
        candidate
        for candidate in (
            clip_box(width, height, face.bbox) for face in app.get(image)
        )
        if candidate is not None
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--insightface-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument(
        "--faceanalysis-equivalence-samples",
        type=int,
        default=8,
        help="Unique images used to verify SCRFD-only boxes against FaceAnalysis",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Diagnostic-only limit; output is marked partial and cannot be frozen",
    )
    args = parser.parse_args()
    if args.det_size != 640:
        raise ValueError("The frozen CocoER detector size is 640")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.faceanalysis_equivalence_samples <= 0:
        raise ValueError("--faceanalysis-equivalence-samples must be positive")

    # Import torch first so its CUDA/cuDNN libraries are visible to ORT.
    import torch
    import cv2
    import insightface
    import numpy as np
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    from src.helper_functions.emotic_loader import EMOTIC

    if insightface.__version__ != "0.7.3":
        raise RuntimeError(
            f"CocoER freezes insightface 0.7.3, found {insightface.__version__}"
        )
    insightface_root = args.insightface_root.expanduser().resolve()
    model_tree = insightface_root / "models" / "buffalo_l"
    model_tree_hash = _tree_sha256(model_tree)
    if model_tree_hash != BUFFALO_L_TREE_SHA256:
        raise ValueError(
            "InsightFace buffalo_l model tree differs from the fixed v0.7 asset"
        )

    requested_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    detector_path = model_tree / "det_10g.onnx"
    detector = insightface.model_zoo.get_model(
        str(detector_path), providers=requested_providers
    )
    detector.prepare(
        ctx_id=0 if args.device == "cuda" else -1,
        input_size=(args.det_size, args.det_size),
    )
    actual_providers = _actual_providers(detector)
    _require_device_provider(args.device, actual_providers)

    # FaceAnalysis uses the same SCRFD session before running its other models.
    # A small exact comparison proves the faster detector-only path preserves
    # the released bbox order and integer clipping used by the port.
    app = FaceAnalysis(
        name="buffalo_l", root=str(insightface_root), providers=requested_providers
    )
    app.prepare(
        ctx_id=0 if args.device == "cuda" else -1,
        det_size=(args.det_size, args.det_size),
    )
    faceanalysis_detector_providers = _actual_providers(app.det_model)
    _require_device_provider(args.device, faceanalysis_detector_providers)

    native_entries = {}
    native_unresolved = []
    image_faces = {}
    sample_geometry = {}
    native_resolved_by_split = Counter()
    native_unresolved_by_split = Counter()
    processed_by_split = Counter()
    processed = 0
    partial = False
    equivalence_checked = 0
    equivalence_max_abs_error = 0.0

    for split in ("train", "val", "test"):
        dataset = EMOTIC(
            str(args.data_root),
            train=split == "train",
            eval_splits=(split,),
            transform=lambda image: image,
            input_mode="full",
        )
        for sample_key, image_path, raw_body in zip(
            dataset.sample_keys, dataset.file_paths, dataset.body_bboxes
        ):
            if args.max_samples is not None and processed >= args.max_samples:
                partial = True
                break
            processed += 1
            processed_by_split[split] += 1
            image_path = str(Path(image_path).resolve())
            if image_path not in image_faces:
                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Cannot read EMOTIC image: {image_path}")
                height, width = image.shape[:2]
                faces = _scrfd_faces(detector, image)
                if equivalence_checked < args.faceanalysis_equivalence_samples:
                    reference = _faceanalysis_faces(app, image)
                    if len(faces) != len(reference):
                        raise RuntimeError(
                            "SCRFD-only and FaceAnalysis face counts differ"
                        )
                    if faces:
                        error = float(
                            np.max(
                                np.abs(
                                    np.asarray(faces, dtype=np.float64)
                                    - np.asarray(reference, dtype=np.float64)
                                )
                            )
                        )
                        equivalence_max_abs_error = max(
                            equivalence_max_abs_error, error
                        )
                    if faces != reference:
                        raise RuntimeError(
                            "SCRFD-only and FaceAnalysis integer boxes differ"
                        )
                    equivalence_checked += 1
                image_faces[image_path] = (width, height, faces)
                if len(image_faces) % 500 == 0:
                    print(
                        json.dumps(
                            {
                                "unique_images": len(image_faces),
                                "processed_samples": processed,
                                "native_resolved_samples": len(native_entries),
                                "native_unresolved_samples": len(native_unresolved),
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            width, height, faces = image_faces[image_path]
            body = clip_box(width, height, raw_body)
            if body is None:
                raise ValueError(f"Invalid EMOTIC body box: {sample_key}")
            sample_geometry[sample_key] = (split, width, height, body)
            candidates = [face for face in faces if source_match(body, face)]
            if not candidates:
                native_unresolved.append(
                    {
                        "sample_id": sample_key,
                        "split": split,
                        "image_path": image_path,
                        "body_box": body,
                        "faces_in_image": len(faces),
                        "reason": "no_source_matching_buffalo_l_face",
                    }
                )
                native_unresolved_by_split[split] += 1
            else:
                native_entries[sample_key] = candidates[0]
                native_resolved_by_split[split] += 1
        if partial:
            break

    train_relative = [
        relative_geometry(sample_geometry[key][3], face)
        for key, face in native_entries.items()
        if sample_geometry[key][0] == "train"
    ]
    median_geometry = median_relative_geometry(train_relative)
    entries = dict(native_entries)
    fallback_sample_ids = []
    fallback_by_split = Counter()
    for item in native_unresolved:
        key = item["sample_id"]
        split, width, height, body = sample_geometry[key]
        entries[key] = project_relative_geometry(
            width, height, body, median_geometry
        )
        fallback_sample_ids.append(key)
        fallback_by_split[split] += 1

    unresolved = sorted(set(sample_geometry).difference(entries))
    payload = {
        "schema_version": 2,
        "upstream_repository": "https://github.com/bisno/CocoER",
        "upstream_commit": UPSTREAM_COMMIT,
        "detector": {
            "library": "insightface",
            "version": insightface.__version__,
            "model": "buffalo_l",
            "detector_file": "det_10g.onnx",
            "model_tree_sha256": model_tree_hash,
            "implementation": "insightface_scrfd_only",
            "requested_device": args.device,
            "requested_providers": requested_providers,
            "actual_providers": actual_providers,
            "det_size": [args.det_size, args.det_size],
            "matching_rule": (
                "first face with x-center and full x-extent inside EMOTIC body bbox"
            ),
            "source_file": "inference.py",
            "runtime": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "onnxruntime": ort.__version__,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
            "faceanalysis_equivalence": {
                "samples": equivalence_checked,
                "integer_boxes_exact_match": True,
                "max_abs_error_after_integer_clipping": equivalence_max_abs_error,
                "faceanalysis_detector_actual_providers": (
                    faceanalysis_detector_providers
                ),
            },
        },
        "conversion": {
            "name": CONVERSION_NAME,
            "sample_preserving": True,
            "native_source": NATIVE_SOURCE,
            "fallback_source": FALLBACK_SOURCE,
            "fallback_calibration_split": "train",
            "fallback_uses_labels": False,
            "fallback_uses_val_or_test_statistics": False,
            "fallback_statistic": "componentwise_median",
            "fallback_relative_to": "annotated_body_xyxy",
            "fallback_rounding": "clip_to_image_then_round_half_up",
            "train_native_calibration_samples": len(train_relative),
            "median_relative_head_box": median_geometry,
        },
        "partial": partial,
        "processed_samples": processed,
        "native_resolved_samples": len(native_entries),
        "native_unresolved_samples": len(native_unresolved),
        "fallback_samples": len(fallback_sample_ids),
        "unresolved_samples": len(unresolved),
        "processed_by_split": dict(sorted(processed_by_split.items())),
        "native_resolved_by_split": dict(sorted(native_resolved_by_split.items())),
        "native_unresolved_by_split": dict(
            sorted(native_unresolved_by_split.items())
        ),
        "fallback_by_split": dict(sorted(fallback_by_split.items())),
        "entries": dict(sorted(entries.items())),
        "fallback_sample_ids": sorted(fallback_sample_ids),
        "native_unresolved": native_unresolved,
        "unresolved": unresolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "partial": partial,
                "processed_samples": processed,
                "native_resolved_samples": len(native_entries),
                "fallback_samples": len(fallback_sample_ids),
                "unresolved_samples": len(unresolved),
                "train_native_calibration_samples": len(train_relative),
                "median_relative_head_box": median_geometry,
                "actual_providers": actual_providers,
                "faceanalysis_equivalence_samples": equivalence_checked,
                "model_tree_sha256": model_tree_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if partial or unresolved:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
