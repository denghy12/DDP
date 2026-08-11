"""Registered sample-preserving head geometry for the CocoER Track-B port."""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence


CONVERSION_NAME = "buffalo_l_strict_then_train_median_v0.1"
NATIVE_SOURCE = "buffalo_l_strict"
FALLBACK_SOURCE = "train_median_relative_geometry"


def clip_box(image_width: int, image_height: int, raw: Sequence[float]):
    """Clip an xyxy box using the detector port's integer convention."""

    values = [float(value) for value in raw[:4]]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    x1 = int(max(0.0, min(float(image_width), values[0])))
    y1 = int(max(0.0, min(float(image_height), values[1])))
    x2 = int(max(0.0, min(float(image_width), values[2])))
    y2 = int(max(0.0, min(float(image_height), values[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def source_match(body: Sequence[float], face: Sequence[float]) -> bool:
    """Exact x-containment rule in the released CocoER ``inference.py``."""

    middle_x = (face[0] + face[2]) // 2
    return (
        body[0] <= middle_x <= body[2]
        and face[0] >= body[0]
        and face[2] <= body[2]
    )


def relative_geometry(body: Sequence[float], face: Sequence[float]):
    """Represent a detected head box relative to its annotated body box."""

    width = float(body[2]) - float(body[0])
    height = float(body[3]) - float(body[1])
    if width <= 0 or height <= 0:
        raise ValueError("CocoER body box must have positive area")
    values = [
        (float(face[0]) - float(body[0])) / width,
        (float(face[1]) - float(body[1])) / height,
        (float(face[2]) - float(body[0])) / width,
        (float(face[3]) - float(body[1])) / height,
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("CocoER relative head geometry must be finite")
    return values


def median_relative_geometry(rows: Iterable[Sequence[float]]):
    """Componentwise median over native detections from the train split only."""

    normalized = [[float(value) for value in row] for row in rows]
    if not normalized or any(len(row) != 4 for row in normalized):
        raise ValueError("CocoER fallback calibration requires train detections")
    values = [
        float(statistics.median(row[column] for row in normalized))
        for column in range(4)
    ]
    if (
        not all(math.isfinite(value) for value in values)
        or values[2] <= values[0]
        or values[3] <= values[1]
    ):
        raise ValueError("CocoER median relative head geometry is invalid")
    return values


def project_relative_geometry(
    image_width: int,
    image_height: int,
    body: Sequence[float],
    relative: Sequence[float],
):
    """Project the frozen train-only relative geometry onto one body box."""

    if len(relative) != 4:
        raise ValueError("CocoER fallback geometry must contain four values")
    width = float(body[2]) - float(body[0])
    height = float(body[3]) - float(body[1])
    if width <= 0 or height <= 0:
        raise ValueError("CocoER body box must have positive area")
    projected = [
        float(body[0]) + float(relative[0]) * width,
        float(body[1]) + float(relative[1]) * height,
        float(body[0]) + float(relative[2]) * width,
        float(body[1]) + float(relative[3]) * height,
    ]
    clipped = [
        max(0.0, min(float(image_width), projected[0])),
        max(0.0, min(float(image_height), projected[1])),
        max(0.0, min(float(image_width), projected[2])),
        max(0.0, min(float(image_height), projected[3])),
    ]
    result = [int(math.floor(value + 0.5)) for value in clipped]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("CocoER fallback projection produced a degenerate box")
    return result
