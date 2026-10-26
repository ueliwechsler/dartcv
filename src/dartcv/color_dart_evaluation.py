"""Repeatable segmentation ablations and sparse, manually labeled probe metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib.figure import Figure

from .color_dart_inspection import likelihood_overlay
from .color_dart_pipeline import (
    BoardGeometry, DartLikelihoodMap, NormalizedBoardView, OrientationEstimate, StageOutcome,
)
from .color_dart_segmentation import (
    ColorAssignment, RobustBoardColorSegmentationStage,
    keep_color_anomalies, preserve_dart_support, suppress_wire_artifacts,
)


def segmentation_variants() -> dict[str, RobustBoardColorSegmentationStage]:
    """Cross color assignment with no, hard, and support-preserving grid cleanup."""
    assignments: tuple[ColorAssignment, ...] = ("field", "board_union", "global")
    cleanups = {
        "none": keep_color_anomalies,
        "hard": suppress_wire_artifacts,
        "support": preserve_dart_support,
    }
    return {
        f"{assignment}/{name}": RobustBoardColorSegmentationStage(
            color_assignment=assignment, cleanup=cleanup
        )
        for assignment in assignments
        for name, cleanup in cleanups.items()
    }


def compare_segmentations(
    view: NormalizedBoardView, board: BoardGeometry, orientation: OrientationEstimate,
) -> dict[str, StageOutcome[DartLikelihoodMap]]:
    """Use identical rectification/orientation for every variant; setup is not rerun."""
    return {
        name: stage(view, board, orientation, diagnostics=False)
        for name, stage in segmentation_variants().items()
    }


def plot_segmentation_comparison(
    view: NormalizedBoardView,
    outcomes: Mapping[str, StageOutcome[DartLikelihoodMap]],
) -> Figure:
    """Return labeled final-output overlays; failures remain visible."""
    if not outcomes:
        raise ValueError("Provide at least one segmentation comparison.")
    columns = min(3, len(outcomes))
    rows = (len(outcomes) + columns - 1) // columns
    figure = Figure(figsize=(5 * columns, 5 * rows), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, (name, outcome) in zip(axes.flat, outcomes.items()):
        if outcome.value is not None:
            image = likelihood_overlay(view.image, outcome.value.probabilities)
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axis.text(.5, .5, outcome.failure, ha="center", va="center", color="red", wrap=True)
        milliseconds = sum(outcome.substep_timings_ms.values())
        axis.set_title(f"{name} ({milliseconds:.0f} ms)", fontsize=10)
    figure.suptitle(
        "Same image, color distribution / cleanup ablation; red = anomaly, NOT confirmed dart"
    )
    return figure


@dataclass(frozen=True, slots=True)
class SparseProbeMetrics:
    """Positive rate over eligible patches; not whole-image precision/recall or IoU."""

    eligible: int
    positive: int
    excluded: int
    scores: tuple[float, ...]

    @property
    def positive_rate(self) -> float | None:
        return self.positive / self.eligible if self.eligible else None


def evaluate_sparse_probes(
    view: NormalizedBoardView,
    board: BoardGeometry,
    probabilities: np.ndarray,
    reference_model_to_source: np.ndarray,
    probes: Mapping[str, Sequence[Sequence[float]]],
    *,
    reference_size_px: int = 768,
    threshold: float = 0.5,
) -> dict[str, SparseProbeMetrics]:
    """Project independently marked reference patches into the current view.

    Ignore patches outside observed scoring geometry or across missing coverage.
    Positive rates mean sensitivity on flight/shaft probes and false-positive
    rates on board/wire probes. These sparse development labels cannot establish
    full-mask accuracy or generalization to different boards/darts.
    """
    if probabilities.shape != view.image.shape[:2] or view.source_coverage is None:
        raise ValueError("Probe evaluation requires matching probabilities and source coverage.")
    DartLikelihoodMap(probabilities)
    if reference_size_px < 2 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid reference size or probe threshold.")
    reference_board = BoardGeometry(reference_model_to_source, 0)
    scale = (reference_size_px - 1) / 340.0
    center = (reference_size_px - 1) / 2
    reference_model_to_view = np.array([[scale, 0, center], [0, scale, center], [0, 0, 1.]])
    transform = (
        view.model_to_view @ np.linalg.inv(board.model_to_source)
        @ reference_board.model_to_source @ np.linalg.inv(reference_model_to_view)
    )
    height, width = probabilities.shape
    observed = (view.board_mask > 0) & (view.source_coverage == 255)
    results = {}
    for category, coordinates in probes.items():
        if category not in ("shaft", "flight", "board", "wire"):
            raise ValueError(f"Unknown probe category: {category}")
        scores = []
        excluded = 0
        radius = 1 if category == "shaft" else 2
        for coordinate in coordinates:
            point = np.asarray(coordinate, dtype=np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("Probe coordinates must be finite xy pairs.")
            neighbors = np.array(
                [[point, point + [radius, 0], point + [0, radius]]], np.float64
            )
            projected = cv2.perspectiveTransform(neighbors, transform)[0]
            if not np.isfinite(projected).all():
                raise ValueError("Probe projection must remain finite.")
            x, y = np.rint(projected[0]).astype(int)
            patch_radius = max(1, round(max(
                np.linalg.norm(projected[1] - projected[0]),
                np.linalg.norm(projected[2] - projected[0]),
            )))
            left, right = x - patch_radius, x + patch_radius + 1
            top, bottom = y - patch_radius, y + patch_radius + 1
            if (
                left < 0 or top < 0 or right > width or bottom > height
                or not observed[top:bottom, left:right].all()
            ):
                excluded += 1
                continue
            scores.append(float(probabilities[top:bottom, left:right].mean()))
        results[category] = SparseProbeMetrics(
            len(scores), sum(score >= threshold for score in scores), excluded, tuple(scores)
        )
    return results
