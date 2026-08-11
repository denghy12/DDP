#!/usr/bin/env python3
"""Generate sample-aligned CocoER head detections with its released detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        file_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(file_digest)
    return digest.hexdigest()


def _box(image_width, image_height, raw):
    values = [float(value) for value in raw[:4]]
    x1 = int(max(0.0, min(float(image_width), values[0])))
    y1 = int(max(0.0, min(float(image_height), values[1])))
    x2 = int(max(0.0, min(float(image_width), values[2])))
    y2 = int(max(0.0, min(float(image_height), values[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _source_match(body, face):
    """Exact x-axis containment test used by the released inference.py."""

    middle_x = (face[0] + face[2]) // 2
    return (
        body[0] <= middle_x <= body[2]
        and face[0] >= body[0]
        and face[2] <= body[2]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--insightface-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
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

    import cv2
    import insightface
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
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name="buffalo_l", root=str(insightface_root), providers=providers)
    app.prepare(
        ctx_id=0 if args.device == "cuda" else -1,
        det_size=(args.det_size, args.det_size),
    )

    entries = {}
    unresolved = []
    image_faces = {}
    resolved_by_split = Counter()
    unresolved_by_split = Counter()
    processed_by_split = Counter()
    processed = 0
    partial = False
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
                detected = app.get(image)
                faces = [
                    candidate
                    for candidate in (
                        _box(width, height, face.bbox) for face in detected
                    )
                    if candidate is not None
                ]
                image_faces[image_path] = (width, height, faces)
            width, height, faces = image_faces[image_path]
            body = _box(width, height, raw_body)
            if body is None:
                raise ValueError(f"Invalid EMOTIC body box: {sample_key}")
            candidates = [
                face for face in faces if _source_match(body, face)
            ]
            if not candidates:
                unresolved.append(
                    {
                        "sample_id": sample_key,
                        "image_path": image_path,
                        "body_box": body,
                        "faces_in_image": len(faces),
                        "reason": "no_source_matching_buffalo_l_face",
                    }
                )
                unresolved_by_split[split] += 1
            else:
                # Preserve the released inference loop: take the first face in
                # FaceAnalysis output order that satisfies its containment test.
                entries[sample_key] = candidates[0]
                resolved_by_split[split] += 1
        if partial:
            break

    payload = {
        "schema_version": 1,
        "upstream_repository": "https://github.com/bisno/CocoER",
        "upstream_commit": UPSTREAM_COMMIT,
        "detector": {
            "library": "insightface",
            "version": insightface.__version__,
            "model": "buffalo_l",
            "model_tree_sha256": model_tree_hash,
            "providers": providers,
            "det_size": [args.det_size, args.det_size],
            "matching_rule": "first face with x-center and full x-extent inside EMOTIC body bbox",
            "source_file": "inference.py",
        },
        "partial": partial,
        "processed_samples": processed,
        "resolved_samples": len(entries),
        "unresolved_samples": len(unresolved),
        "processed_by_split": dict(sorted(processed_by_split.items())),
        "resolved_by_split": dict(sorted(resolved_by_split.items())),
        "unresolved_by_split": dict(sorted(unresolved_by_split.items())),
        "entries": entries,
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
                "resolved_samples": len(entries),
                "unresolved_samples": len(unresolved),
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
