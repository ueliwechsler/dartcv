"""Execute the implemented color-pipeline prefix without fake scoring stages."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter

from .color_dart_pipeline import (
    BoardLocalizationStage, BoardNormalizationStage, BoardPreparationResult,
    ColorPipelineStep, DartLikelihoodMap, Image, OrientationEstimate,
    OrientationStage, SegmentationStage, StageOutcome, StageStatus,
    _snapshot_artifacts, _stage_record, detect_board_stage, normalize_board_view, prepare_board,
)


@dataclass(frozen=True, slots=True)
class ColorBoardAnalysis:
    """Partial prefix result; there are deliberately no dart counts or scores."""

    preparation: BoardPreparationResult
    orientation: StageOutcome[OrientationEstimate] | None
    segmentation: StageOutcome[DartLikelihoodMap] | None
    steps: tuple[ColorPipelineStep, ...]


def analyze_board_colors(
    image: Image,
    *,
    orientation: OrientationStage,
    segmentation: SegmentationStage,
    normalized_size_px: int = 1024,
    diagnostics: bool = False,
    board_localization: BoardLocalizationStage = detect_board_stage,
    board_normalization: BoardNormalizationStage = partial(
        normalize_board_view, number_ring_radius_mm=240.0
    ),
) -> ColorBoardAnalysis:
    """Shared preparation/orientation/segmentation path for overview and batches."""
    preparation = prepare_board(
        image, normalized_size_px=normalized_size_px, diagnostics=diagnostics,
        board_localization=board_localization, board_normalization=board_normalization,
    )
    steps = list(preparation.steps)
    oriented = None
    segmented = None
    if preparation.board_view is not None and preparation.board is not None:
        view = preparation.board_view
        inputs = {"normalized_board": view.image}
        if view.number_ring is not None:
            inputs["number_ring"] = view.number_ring.image
        snapshots = _snapshot_artifacts(inputs, diagnostics)
        start = perf_counter()
        oriented = orientation(view, diagnostics=diagnostics)
        steps.append(_stage_record(
            "orientation", start, oriented, diagnostics=diagnostics, inputs=snapshots
        ))
        if oriented.value is not None:
            snapshots = _snapshot_artifacts({"normalized_board": view.image}, diagnostics)
            start = perf_counter()
            segmented = segmentation(
                view, preparation.board, oriented.value, diagnostics=diagnostics
            )
            steps.append(_stage_record(
                "dart_segmentation", start, segmented, diagnostics=diagnostics, inputs=snapshots
            ))
    for name in ("orientation", "dart_segmentation"):
        if not any(step.name == name for step in steps):
            steps.append(ColorPipelineStep(name, StageStatus.SKIPPED, 0.0))
    return ColorBoardAnalysis(preparation, oriented, segmented, tuple(steps))
