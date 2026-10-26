"""Recognition backend contracts and validation in original-image pixels.

No trained detector, OCR, or tip model is bundled. Backends return observations,
not model-projected substitutes; only measured or manual tips should be scored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import SECTORS, project


@dataclass(frozen=True)
class DartBox:
    """One dart's xyxy bounds, confidence in [0, 1], and backend provenance."""

    id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    provenance: str


@dataclass(frozen=True, eq=False)
class NumberReading:
    """Observed printed text and its (N, 2) polygon, never an expected label."""

    polygon: np.ndarray
    text: str
    confidence: float
    provenance: str


@dataclass(frozen=True)
class DartTip:
    """Board entry point; dart_id=None denotes an unassociated observation."""

    dart_id: int | None
    xy: tuple[float, float]
    confidence: float
    provenance: str


@dataclass(frozen=True)
class StageResult[T]:
    """An empty successful observation differs from a missing/failed backend."""

    status: Literal["ok", "not configured", "failed"]
    items: tuple[T, ...] = ()
    message: str = ""


def _inside(image: np.ndarray, points: np.ndarray) -> bool:
    height, width = image.shape[:2]
    return bool(
        np.isfinite(points).all()
        and (points >= 0).all()
        and (points[:, 0] <= width - 1).all()
        and (points[:, 1] <= height - 1).all()
    )


def _check_confidence(item: DartBox | NumberReading | DartTip) -> None:
    if not 0.0 <= item.confidence <= 1.0:
        raise ValueError(f"confidence outside [0, 1]: {item}")


def validate_boxes(image: np.ndarray, boxes: Sequence[DartBox]) -> None:
    """Reject duplicate IDs, empty/out-of-image boxes, and bad confidence."""
    ids = [box.id for box in boxes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate dart ids")
    for box in boxes:
        x0, y0, x1, y1 = box.xyxy
        if not (
            x0 < x1
            and y0 < y1
            and _inside(image, np.asarray([(x0, y0), (x1, y1)], float))
        ):
            raise ValueError(f"dart {box.id}: box is empty or outside the image")
        _check_confidence(box)


def validate_numbers(image: np.ndarray, readings: Sequence[NumberReading]) -> None:
    """Require nonempty observed text and finite in-image polygon vertices."""
    for reading in readings:
        polygon = np.asarray(reading.polygon, float)
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(polygon) < 3
            or not _inside(image, polygon)
        ):
            raise ValueError(
                f"number {reading.text!r}: polygon is not a valid outline inside the image"
            )
        if not reading.text.strip():
            raise ValueError("empty number text")
        _check_confidence(reading)


def validate_tips(
    image: np.ndarray, tips: Sequence[DartTip], boxes: Sequence[DartBox]
) -> None:
    """Check tips and associations, allowing 10% box-size padding at edges.

    This checks structural plausibility, not whether a tip is physically correct.
    Unassociated tips are allowed, including manually supplied coordinates.
    """
    boxes_by_id = {box.id: box for box in boxes}
    for tip in tips:
        point = np.asarray(tip.xy, float)
        if point.shape != (2,) or not _inside(image, point.reshape(1, 2)):
            raise ValueError(
                f"tip {tip.xy} is outside the image or not an (x, y) point"
            )
        _check_confidence(tip)
        if tip.dart_id is None:
            continue
        if tip.dart_id not in boxes_by_id:
            raise ValueError(f"tip refers to unknown dart {tip.dart_id}")
        x0, y0, x1, y1 = boxes_by_id[tip.dart_id].xyxy
        pad = 0.1 * max(x1 - x0, y1 - y0)
        if not (
            x0 - pad <= tip.xy[0] <= x1 + pad and y0 - pad <= tip.xy[1] <= y1 + pad
        ):
            raise ValueError(f"tip {tip.xy} is far from the box of dart {tip.dart_id}")


def run_stage[T](
    name: str,
    backend: Callable[..., Iterable[T]] | None,
    validate: Callable[[tuple[T, ...]], None],
    *args,
) -> StageResult[T]:
    """Materialize and validate backend output atomically; failures yield no items."""
    if backend is None:
        return StageResult("not configured", message=f"{name}: no backend configured")
    try:
        items = tuple(backend(*args))
        validate(items)
    except Exception as exc:
        return StageResult("failed", message=f"{name}: {type(exc).__name__}: {exc}")
    return StageResult("ok", items, f"{name}: {len(items)} found")


def expected_sector(H: np.ndarray, xy: tuple[float, float] | np.ndarray) -> int:
    """Model sector at a pixel bearing; a comparison helper, not OCR evidence."""
    x_mm, y_mm = project(np.linalg.inv(H), [xy])[0]
    index = int(((np.degrees(np.arctan2(x_mm, -y_mm)) + 9) % 360) // 18)
    return SECTORS[index % 20]
