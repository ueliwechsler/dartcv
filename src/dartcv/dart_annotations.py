"""Source-photo annotations and visible-region segmentation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .color_dart_pipeline import DartLikelihoodMap, NormalizedBoardView

Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = tuple[Point, ...]
Part: TypeAlias = Literal["flight", "shaft_barrel", "uncertain"]
PARTS: tuple[Part, ...] = ("flight", "shaft_barrel", "uncertain")


@dataclass(frozen=True)
class VisibleRegion:
    part: Part
    points: Polygon


@dataclass(frozen=True)
class AnnotatedDart:
    id: str
    visible: tuple[VisibleRegion, ...]
    silhouette: tuple[Polygon, ...]
    tip_status: Literal["exact", "unknown"]
    tip: Point | None
    contact: Literal["embedded", "not_embedded", "unknown"]


@dataclass(frozen=True)
class ImageAnnotation:
    image: str
    width: int
    height: int
    sha256: str
    status: Literal["draft", "completed"]
    darts: tuple[AnnotatedDart, ...]
    ignore: tuple[Polygon, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "status": self.status,
            "darts": [
                {
                    "id": dart.id,
                    "visible": [
                        {"part": region.part, "points": [list(point) for point in region.points]}
                        for region in dart.visible
                    ],
                    "silhouette": [[list(point) for point in polygon] for polygon in dart.silhouette],
                    "tip_status": dart.tip_status,
                    "tip": list(dart.tip) if dart.tip is not None else None,
                    "contact": dart.contact,
                }
                for dart in self.darts
            ],
            "ignore": [[list(point) for point in polygon] for polygon in self.ignore],
        }


def parse_annotation(data: object) -> ImageAnnotation:
    """Validate disk/API labels; never accept malformed or ambiguous completed data."""
    if not isinstance(data, dict):
        raise ValueError("Annotation must be a JSON object.")
    required = {
        "schema_version", "image", "width", "height", "sha256",
        "status", "darts", "ignore",
    }
    if set(data) != required or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Expected annotation schema_version 1 and its documented fields.")
    image = data["image"]
    if (
        not isinstance(image, str) or not image or image in (".", "..")
        or "/" in image or "\\" in image
    ):
        raise ValueError("Image must be a filename, not a path.")
    width, height = data["width"], data["height"]
    if any(type(value) is not int or value < 2 for value in (width, height)):
        raise ValueError("Image dimensions must be integers >= 2.")
    fingerprint = data["sha256"]
    if (
        not isinstance(fingerprint, str) or len(fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise ValueError("Image sha256 must be a lowercase SHA-256 fingerprint.")
    if data["status"] not in ("draft", "completed"):
        raise ValueError("Status must be draft or completed.")

    def point(value: object) -> Point:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Coordinates must be xy pairs.")
        if any(type(v) not in (float, int) or not np.isfinite(v) for v in value):
            raise ValueError("Coordinates must be finite numbers.")
        x, y = value
        if not 0 <= x <= width - 1 or not 0 <= y <= height - 1:
            raise ValueError("Coordinates lie outside the source photograph.")
        return float(x), float(y)

    def polygon(value: object) -> Polygon:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            raise ValueError("A polygon needs at least three vertices.")
        vertices = tuple(point(p) for p in value)
        contour = np.asarray(vertices, np.float32)
        if abs(cv2.contourArea(contour)) < 0.5:
            raise ValueError("A polygon must enclose nonzero area.")
        # Crossing outlines have ambiguous visible/background semantics.
        for i in range(len(vertices)):
            a, b = contour[i], contour[(i + 1) % len(vertices)]
            if np.linalg.norm(a - b) < 1e-6:
                raise ValueError("Polygon vertices must be distinct along each edge.")
            for j in range(i + 1, len(vertices)):
                if j == i + 1 or (i == 0 and j == len(vertices) - 1):
                    continue
                c, d = contour[j], contour[(j + 1) % len(vertices)]

                def cross(u, v):
                    return float(u[0] * v[1] - u[1] * v[0])

                ab_c, ab_d = cross(b - a, c - a), cross(b - a, d - a)
                cd_a, cd_b = cross(d - c, a - c), cross(d - c, b - c)
                if ab_c * ab_d <= 0 and cd_a * cd_b <= 0 and (
                    np.maximum(np.minimum(a, b), np.minimum(c, d))
                    <= np.minimum(np.maximum(a, b), np.maximum(c, d))
                ).all():
                    raise ValueError("Polygon edges must not cross or touch themselves.")
        return vertices

    def polygons(value: object) -> tuple[Polygon, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Polygon collections must be arrays.")
        return tuple(polygon(item) for item in value)

    if not isinstance(data["darts"], list):
        raise ValueError("Darts must be an array.")
    darts = []
    ids = set()
    for item in data["darts"]:
        if not isinstance(item, dict) or set(item) != {
            "id", "visible", "silhouette", "tip_status", "tip", "contact",
        }:
            raise ValueError("Each dart needs id, visible, silhouette, tip_status, tip and contact.")
        dart_id = item["id"]
        if not isinstance(dart_id, str) or not dart_id.strip() or dart_id in ids:
            raise ValueError("Dart IDs must be nonempty unique strings.")
        ids.add(dart_id)
        if not isinstance(item["visible"], list):
            raise ValueError("Visible regions must be an array.")
        visible = []
        for region in item["visible"]:
            if not isinstance(region, dict) or set(region) != {"part", "points"} or region["part"] not in PARTS:
                raise ValueError("Visible regions require a flight, shaft_barrel or uncertain part.")
            visible.append(VisibleRegion(region["part"], polygon(region["points"])))
        if data["status"] == "completed" and not visible:
            raise ValueError(f"Completed dart {dart_id} needs at least one visible region.")
        tip_status = item["tip_status"]
        if tip_status not in ("exact", "unknown"):
            raise ValueError("Tip status must be exact or unknown.")
        tip = point(item["tip"]) if tip_status == "exact" else None
        if tip_status == "unknown" and item["tip"] is not None:
            raise ValueError("An unknown tip must have null coordinates.")
        if item["contact"] not in ("embedded", "not_embedded", "unknown"):
            raise ValueError("Contact must be embedded, not_embedded or unknown.")
        darts.append(AnnotatedDart(
            dart_id, tuple(visible), polygons(item["silhouette"]),
            tip_status, tip, item["contact"],
        ))
    return ImageAnnotation(
        image, width, height, fingerprint, data["status"],
        tuple(darts), polygons(data["ignore"]),
    )


def load_annotation(path: str | Path, *, source_image: str | Path | None = None) -> ImageAnnotation:
    annotation = parse_annotation(json.loads(Path(path).read_text()))
    if source_image is not None:
        source_image = Path(source_image)
        if source_image.name != annotation.image or sha256(source_image.read_bytes()).hexdigest() != annotation.sha256:
            raise ValueError("Labels do not match the source photograph; do not evaluate a changed image.")
    return annotation


def annotation_path(directory: str | Path, image_name: str) -> Path:
    """Retain the image extension, avoiding collisions such as a.jpg and a.png."""
    if not image_name or Path(image_name).name != image_name or "\\" in image_name or image_name in (".", ".."):
        raise ValueError("Expected an image filename.")
    return Path(directory) / f"{image_name}.json"


def _fill(shape: tuple[int, int], polygons: tuple[Polygon, ...], transform=None) -> NDArray[np.uint8]:
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        vertices = np.asarray(polygon, np.float64).reshape(1, -1, 2)
        if transform is not None:
            homogeneous = np.column_stack((vertices[0], np.ones(vertices.shape[1]))) @ transform.T
            denominator = homogeneous[:, 2]
            if np.any(np.abs(denominator) < 1e-9) or np.min(denominator) * np.max(denominator) <= 0:
                raise ValueError("Annotation crosses the normalization horizon.")
            vertices = cv2.perspectiveTransform(vertices, transform)
        if not np.isfinite(vertices).all() or np.abs(vertices).max() > 1e8:
            raise ValueError("Annotation projection is outside a reliable numeric range.")
        cv2.fillPoly(mask, [np.rint(vertices[0]).astype(np.int32)], 1)
    return mask


@dataclass(frozen=True)
class VisibleMaskMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    evaluated_pixels: int
    part_recall: dict[str, float | None]
    source_visible_pixels: int
    source_evaluated_visible_pixels: int

    @property
    def recall(self) -> float | None:
        total = self.true_positive + self.false_negative
        return self.true_positive / total if total else None

    @property
    def precision(self) -> float | None:
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total else None

    @property
    def iou(self) -> float | None:
        total = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / total if total else None

    @property
    def label_coverage(self) -> float | None:
        return (
            self.source_evaluated_visible_pixels / self.source_visible_pixels
            if self.source_visible_pixels else None
        )


@dataclass(frozen=True)
class VisibleMaskEvaluation:
    metrics: VisibleMaskMetrics
    error_overlay: NDArray[np.uint8]


def evaluate_visible_masks(
    view: NormalizedBoardView,
    probabilities: NDArray[np.float32],
    annotation: ImageAnnotation,
    *,
    threshold: float = 0.5,
) -> VisibleMaskEvaluation:
    """Evaluate only completed, visible labels in observed scoring-board coverage.

    Silhouettes and tips never contribute. Inferred outlines are not observable
    foreground, and an unknown tip is not a missed detection.
    """
    annotation = parse_annotation(annotation.to_dict())
    if annotation.status != "completed":
        raise ValueError("Complete the photo before evaluating visible masks.")
    DartLikelihoodMap(probabilities)
    if probabilities.shape != view.image.shape[:2] or view.source_coverage is None:
        raise ValueError("Evaluation requires matching likelihoods and source coverage.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one.")
    shape = probabilities.shape
    visible = tuple(region.points for dart in annotation.darts for region in dart.visible)
    truth = _fill(shape, visible, view.source_to_view) > 0
    ignored = _fill(shape, annotation.ignore, view.source_to_view) > 0
    eligible = (view.board_mask > 0) & (view.source_coverage == 255) & ~ignored
    prediction = probabilities >= threshold
    tp = truth & prediction & eligible
    fp = ~truth & prediction & eligible
    fn = truth & ~prediction & eligible
    part_recall = {}
    for part in PARTS:
        polygons = tuple(
            region.points for dart in annotation.darts for region in dart.visible
            if region.part == part
        )
        selected = (_fill(shape, polygons, view.source_to_view) > 0) & eligible
        count = int(selected.sum())
        part_recall[part] = float(prediction[selected].mean()) if count else None
    source_shape = (annotation.height, annotation.width)
    source_truth = _fill(source_shape, visible) > 0
    source_ignored = _fill(source_shape, annotation.ignore) > 0
    source_truth &= ~source_ignored
    source_eligible = cv2.warpPerspective(
        eligible.astype(np.uint8), view.view_to_source,
        (annotation.width, annotation.height), flags=cv2.INTER_NEAREST,
    ) > 0
    metrics = VisibleMaskMetrics(
        int(tp.sum()), int(fp.sum()), int(fn.sum()), int(eligible.sum()), part_recall,
        int(source_truth.sum()), int((source_truth & source_eligible).sum()),
    )
    overlay = view.image.copy()
    # BGR: green retained, red missed dart, cyan false foreground, gray excluded.
    for mask, color in (
        (tp, (0, 200, 0)), (fn, (0, 0, 255)), (fp, (255, 255, 0)), (~eligible, (100, 100, 100)),
    ):
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array(color)).astype(np.uint8)
    return VisibleMaskEvaluation(metrics, overlay)


def plot_visible_mask_evaluation(evaluation: VisibleMaskEvaluation) -> Figure:
    figure = Figure(figsize=(8, 8), layout="constrained")
    axis = figure.subplots()
    axis.imshow(cv2.cvtColor(evaluation.error_overlay, cv2.COLOR_BGR2RGB))
    axis.set_axis_off()
    axis.set_title("Green: retained dart | Red: missed dart\nCyan: board/grid noise | Gray: excluded")
    return figure
