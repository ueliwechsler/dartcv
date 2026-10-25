"""Modular, color-based recognition of darts against a standard dartboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from time import perf_counter
from typing import Generic, Mapping, Protocol, TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray

from .detect import DetectionError, detect_board

Image = NDArray[np.uint8]
Matrix3 = NDArray[np.float64]
T = TypeVar("T")

BOARD_RADII_MM: Mapping[str, float] = {
    "inner_bull": 6.35,
    "outer_bull": 15.9,
    "triple_inner": 99.0,
    "triple_outer": 107.0,
    "double_inner": 162.0,
    "double_outer": 170.0,
}
BOARD_DIAMETER_MM = 2.0 * BOARD_RADII_MM["double_outer"]
SECTOR_ORDER = (
    20,
    1,
    18,
    4,
    13,
    6,
    10,
    15,
    2,
    17,
    3,
    19,
    7,
    16,
    8,
    11,
    14,
    9,
    12,
    5,
)


class StageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ColorPipelineStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"


class EntryProvenance(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNRESOLVED = "unresolved"


class UncertaintyReason(StrEnum):
    ENTRY_INFERRED = "entry_inferred"
    ENTRY_UNLOCALIZED = "entry_unlocalized"
    ORIENTATION_UNCERTAIN = "orientation_uncertain"
    SCORE_BOUNDARY = "score_boundary"
    SCORE_UNAVAILABLE = "score_unavailable"


class ScoringRing(StrEnum):
    MISS = "miss"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    OUTER_BULL = "outer_bull"
    INNER_BULL = "inner_bull"


@dataclass(frozen=True, slots=True)
class StageOutcome(Generic[T]):
    """A stage's expected success/failure, plus optional diagnostics."""

    value: T | None
    failure: str | None = None
    artifacts: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)
    substep_timings_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.value is None) == (self.failure is None):
            raise ValueError("A stage outcome must contain either a value or a failure.")
        for name, duration in self.substep_timings_ms.items():
            if not name or not isfinite(duration) or duration < 0:
                raise ValueError("Substep timings need names and finite non-negative durations.")

    @classmethod
    def succeeded(
        cls,
        value: T,
        *,
        artifacts: Mapping[str, NDArray[np.generic]] | None = None,
        substep_timings_ms: Mapping[str, float] | None = None,
    ) -> StageOutcome[T]:
        return cls(
            value=value,
            artifacts=artifacts or {},
            substep_timings_ms=substep_timings_ms or {},
        )

    @classmethod
    def failed(
        cls,
        message: str,
        *,
        artifacts: Mapping[str, NDArray[np.generic]] | None = None,
        substep_timings_ms: Mapping[str, float] | None = None,
    ) -> StageOutcome[T]:
        if not message:
            raise ValueError("A failed stage needs a non-empty message.")
        return cls(
            value=None,
            failure=message,
            artifacts=artifacts or {},
            substep_timings_ms=substep_timings_ms or {},
        )


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    """Board-model-to-source transform and color-derived, ambiguous rotation."""

    model_to_source: Matrix3
    detector_rotation_deg: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.model_to_source, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Board homography must be a finite 3x3 matrix.")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("Board homography must be invertible.")
        if not isfinite(self.detector_rotation_deg):
            raise ValueError("Board rotation must be finite.")
        object.__setattr__(self, "model_to_source", matrix.copy())


@dataclass(frozen=True, slots=True)
class NumberRingView:
    """Optional wider rectification; coverage distinguishes missing source pixels."""

    image: Image
    source_coverage: Image
    model_to_view: Matrix3


@dataclass(frozen=True, slots=True)
class NormalizedBoardView:
    """Rectified view with source/view transforms; board diameter fills the canvas."""

    image: Image
    board_mask: Image
    model_to_view: Matrix3
    source_to_view: Matrix3
    view_to_source: Matrix3
    number_ring: NumberRingView | None = None


@dataclass(frozen=True, slots=True)
class OrientationEstimate:
    """Clockwise correction that places the printed 20 at the top of the view."""

    correction_deg: float
    confidence: float
    confident: bool

    def __post_init__(self) -> None:
        if not isfinite(self.correction_deg):
            raise ValueError("Orientation correction must be finite.")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Orientation confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class DartLikelihoodMap:
    """Float probability map indicating pixels likely to belong to darts."""

    probabilities: NDArray[np.float32]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.probabilities, np.ndarray)
            or self.probabilities.dtype != np.float32
        ):
            raise ValueError("Dart likelihood must be a float32 NumPy array.")
        if self.probabilities.ndim != 2:
            raise ValueError("Dart likelihood must be a single-channel map.")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("Dart likelihood values must be finite.")
        if np.any(self.probabilities < 0.0) or np.any(self.probabilities > 1.0):
            raise ValueError("Dart likelihood values must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class BoardEntryEstimate:
    """Estimated point where a dart enters the board, in source-image pixels."""

    xy: tuple[float, float] | None
    confidence: float
    provenance: EntryProvenance
    uncertainty_reasons: tuple[UncertaintyReason, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Entry confidence must be between 0 and 1.")
        if self.xy is None:
            if self.provenance is not EntryProvenance.UNRESOLVED:
                raise ValueError("An entry without coordinates must be unresolved.")
        else:
            if len(self.xy) != 2:
                raise ValueError("Board-entry coordinates must contain exactly two values.")
            if not all(isfinite(coordinate) for coordinate in self.xy):
                raise ValueError("Board-entry coordinates must be finite.")


@dataclass(frozen=True, slots=True)
class ScoreEstimate:
    """Best-guess scoring region and points for one board-entry estimate."""

    sector: int | None
    ring: ScoringRing
    points: int
    uncertainty_reasons: tuple[UncertaintyReason, ...] = ()

    def __post_init__(self) -> None:
        if self.sector is not None and not 1 <= self.sector <= 20:
            raise ValueError("A scoring sector must be between 1 and 20.")
        if self.points < 0:
            raise ValueError("A score cannot be negative.")
        if self.ring is ScoringRing.MISS and (self.sector is not None or self.points != 0):
            raise ValueError("A miss must have no sector and be worth zero points.")


@dataclass(frozen=True, slots=True)
class DartResult:
    """One expected dart's entry estimate and optional best-guess score."""

    entry: BoardEntryEstimate
    score: ScoreEstimate | None
    uncertainty_reasons: tuple[UncertaintyReason, ...]


@dataclass(frozen=True, slots=True)
class ColorPipelineStep:
    """Timing, outcome, and optional image snapshots for one conceptual stage."""

    name: str
    status: StageStatus
    elapsed_ms: float
    failure: str | None = None
    substep_timings_ms: Mapping[str, float] = field(default_factory=dict)
    input_artifacts: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)
    output_artifacts: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ColorPipelineFailure:
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class ColorPipelineResult:
    """Per-run output, with explicit total completeness and score uncertainty."""

    status: ColorPipelineStatus
    expected_dart_count: int
    board: BoardGeometry | None
    orientation: OrientationEstimate | None
    darts: tuple[DartResult, ...]
    total_points: int | None
    total_complete: bool
    total_uncertain: bool
    failure: ColorPipelineFailure | None
    steps: tuple[ColorPipelineStep, ...]
    diagnostics_enabled: bool


@dataclass(frozen=True, slots=True)
class BoardPreparationResult:
    """Board-only run, usable before recognition stages are implemented."""

    board: BoardGeometry | None
    board_view: NormalizedBoardView | None
    failure: ColorPipelineFailure | None
    steps: tuple[ColorPipelineStep, ...]
    diagnostics_enabled: bool


class BoardLocalizationStage(Protocol):
    """Localize a board without mutating its input image."""

    def __call__(
        self, image: Image, *, diagnostics: bool
    ) -> StageOutcome[BoardGeometry]: ...


class BoardNormalizationStage(Protocol):
    """Rectify a board without mutating its input image or geometry."""

    def __call__(
        self,
        image: Image,
        board: BoardGeometry,
        output_size_px: int,
        *,
        diagnostics: bool,
    ) -> StageOutcome[NormalizedBoardView]: ...


class OrientationStage(Protocol):
    """Estimate the numbered-20 alignment without mutating the board view."""

    def __call__(
        self, board_view: NormalizedBoardView, *, diagnostics: bool
    ) -> StageOutcome[OrientationEstimate]: ...


class SegmentationStage(Protocol):
    """Build a dart-likelihood map without mutating prior stage outputs."""

    def __call__(
        self,
        board_view: NormalizedBoardView,
        board: BoardGeometry,
        orientation: OrientationEstimate,
        *,
        diagnostics: bool,
    ) -> StageOutcome[DartLikelihoodMap]: ...


class TipLocalizationStage(Protocol):
    """Estimate source-image entry points without mutating prior stage outputs."""

    def __call__(
        self,
        image: Image,
        board_view: NormalizedBoardView,
        board: BoardGeometry,
        orientation: OrientationEstimate,
        likelihood: DartLikelihoodMap,
        expected_dart_count: int,
        *,
        diagnostics: bool,
    ) -> StageOutcome[tuple[BoardEntryEstimate, ...]]: ...


class ScoringStage(Protocol):
    """Score entry estimates without mutating them."""

    def __call__(
        self,
        board: BoardGeometry,
        orientation: OrientationEstimate,
        entries: tuple[BoardEntryEstimate, ...],
        boundary_margin_mm: float,
        *,
        diagnostics: bool,
    ) -> StageOutcome[tuple[ScoreEstimate | None, ...]]: ...


def _validate_image(image: Image) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("Expected a non-empty BGR uint8 image with shape (height, width, 3).")


def _board_outline(image: Image, model_to_source: Matrix3) -> Image:
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    model_points = np.column_stack(
        (
            BOARD_RADII_MM["double_outer"] * np.cos(angles),
            BOARD_RADII_MM["double_outer"] * np.sin(angles),
            np.ones_like(angles),
        )
    )
    projected = model_points @ model_to_source.T
    pixels = projected[:, :2] / projected[:, 2:3]
    points = np.rint(pixels).astype(np.int32).reshape((-1, 1, 2))
    output = image.copy()
    cv2.polylines(
        output,
        [points],
        isClosed=True,
        color=(255, 0, 255),
        thickness=max(1, round(min(image.shape[:2]) / 1000)),
        lineType=cv2.LINE_AA,
    )
    return output


def detect_board_stage(
    image: Image, *, diagnostics: bool
) -> StageOutcome[BoardGeometry]:
    """Adapt the existing board detector to this pipeline's stage contract."""
    _validate_image(image)
    try:
        detection = detect_board(image, upright_deg=0.0, diagnostics=diagnostics)
    except DetectionError as error:
        return StageOutcome.failed(str(error))

    board = BoardGeometry(
        model_to_source=detection.H,
        detector_rotation_deg=detection.rotation_deg,
    )
    artifacts: dict[str, NDArray[np.generic]] = {}
    if diagnostics:
        artifacts = {"board_outline": _board_outline(image, board.model_to_source)}
        if detection.diagnostics is not None:
            for name in (
                "colour_mask",
                "boundary_points",
                "ellipse_inliers",
                "ecc_edges",
                "ecc_template",
                "ecc_mask",
                "H_before_bull",
                "H_before_rings",
            ):
                artifact: NDArray[np.generic] | None = getattr(
                    detection.diagnostics, name
                )
                if artifact is not None:
                    artifacts[name] = artifact
    return StageOutcome.succeeded(board, artifacts=artifacts)


def normalize_board_view(
    image: Image,
    board: BoardGeometry,
    output_size_px: int,
    *,
    diagnostics: bool,
    number_ring_radius_mm: float | None = None,
) -> StageOutcome[NormalizedBoardView]:
    """Rectify scoring geometry and, optionally, a wider printed-number view.

    Use functools.partial to request a number ring while retaining the normalizer
    stage signature. The scoring image and its transforms are unchanged.
    """
    del diagnostics
    _validate_image(image)
    if output_size_px < 2:
        raise ValueError("Normalized board size must be at least 2 pixels.")
    if number_ring_radius_mm is not None and (
        not isfinite(number_ring_radius_mm)
        or number_ring_radius_mm <= BOARD_RADII_MM["double_outer"]
    ):
        raise ValueError("Number-ring radius must be finite and exceed the scoring radius.")

    scale = (output_size_px - 1) / BOARD_DIAMETER_MM
    center = (output_size_px - 1) / 2.0
    model_to_view = np.array(
        [[scale, 0.0, center], [0.0, scale, center], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    source_to_view = model_to_view @ np.linalg.inv(board.model_to_source)
    view_to_source = np.linalg.inv(source_to_view)
    normalized = cv2.warpPerspective(
        image,
        source_to_view,
        (output_size_px, output_size_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask = np.zeros((output_size_px, output_size_px), dtype=np.uint8)
    cv2.circle(
        mask,
        (round(center), round(center)),
        round(scale * BOARD_RADII_MM["double_outer"]),
        255,
        thickness=cv2.FILLED,
    )
    artifacts = {"normalized_board": normalized, "board_mask": mask}
    number_ring = None
    if number_ring_radius_mm is not None:
        ring_scale = (output_size_px - 1) / (2.0 * number_ring_radius_mm)
        ring_transform = np.array(
            [[ring_scale, 0.0, center], [0.0, ring_scale, center], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        ring_to_source = board.model_to_source @ np.linalg.inv(ring_transform)
        yy, xx = np.indices((output_size_px, output_size_px), dtype=np.float32)
        points = np.stack((xx, yy), axis=-1)
        source_points = cv2.perspectiveTransform(points, ring_to_source)
        map_x, map_y = source_points[..., 0], source_points[..., 1]
        ring_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        coverage = (
            np.isfinite(source_points).all(axis=-1)
            & (map_x >= 0)
            & (map_x <= image.shape[1] - 1)
            & (map_y >= 0)
            & (map_y <= image.shape[0] - 1)
        ).astype(np.uint8) * 255
        number_ring = NumberRingView(ring_image, coverage, ring_transform)
        artifacts.update(number_ring=ring_image, number_ring_coverage=coverage)
    view = NormalizedBoardView(
        image=normalized,
        board_mask=mask,
        model_to_view=model_to_view,
        source_to_view=source_to_view,
        view_to_source=view_to_source,
        number_ring=number_ring,
    )
    return StageOutcome.succeeded(
        view,
        artifacts=artifacts,
    )


def _snapshot_artifacts(
    artifacts: Mapping[str, NDArray[np.generic]], enabled: bool
) -> Mapping[str, NDArray[np.generic]]:
    if not enabled:
        return {}
    return {name: image.copy() for name, image in artifacts.items()}


def _stage_record(
    name: str,
    started: float,
    outcome: StageOutcome[T],
    *,
    diagnostics: bool,
    inputs: Mapping[str, NDArray[np.generic]],
) -> ColorPipelineStep:
    return ColorPipelineStep(
        name=name,
        status=StageStatus.FAILED if outcome.failure is not None else StageStatus.SUCCEEDED,
        elapsed_ms=(perf_counter() - started) * 1000.0,
        failure=outcome.failure,
        substep_timings_ms=dict(outcome.substep_timings_ms),
        input_artifacts=dict(inputs),
        output_artifacts=_snapshot_artifacts(outcome.artifacts, diagnostics),
    )


_STAGE_NAMES = (
    "board_localization",
    "board_normalization",
    "orientation",
    "dart_segmentation",
    "tip_localization",
    "scoring",
)


def prepare_board(
    image: Image,
    *,
    normalized_size_px: int = 1024,
    diagnostics: bool = False,
    board_localization: BoardLocalizationStage = detect_board_stage,
    board_normalization: BoardNormalizationStage = normalize_board_view,
) -> BoardPreparationResult:
    """Run only localization and normalization, with the full pipeline's contracts."""
    _validate_image(image)
    if (
        isinstance(normalized_size_px, bool)
        or not isinstance(normalized_size_px, int)
        or normalized_size_px < 2
    ):
        raise ValueError("Normalized board size must be an integer of at least 2 pixels.")

    inputs = _snapshot_artifacts({"image": image}, diagnostics)
    started = perf_counter()
    localization = board_localization(image, diagnostics=diagnostics)
    localization_step = _stage_record(
        _STAGE_NAMES[0], started, localization, diagnostics=diagnostics, inputs=inputs
    )
    if localization.failure is not None:
        skipped = ColorPipelineStep(
            name=_STAGE_NAMES[1],
            status=StageStatus.SKIPPED,
            elapsed_ms=0.0,
            failure="Skipped after board_localization failed.",
        )
        return BoardPreparationResult(
            board=None,
            board_view=None,
            failure=ColorPipelineFailure(_STAGE_NAMES[0], localization.failure),
            steps=(localization_step, skipped),
            diagnostics_enabled=diagnostics,
        )
    board = localization.value
    if not isinstance(board, BoardGeometry):
        raise TypeError("Board localization returned an invalid result.")

    inputs = _snapshot_artifacts({"image": image}, diagnostics)
    started = perf_counter()
    normalization = board_normalization(
        image, board, normalized_size_px, diagnostics=diagnostics
    )
    normalization_step = _stage_record(
        _STAGE_NAMES[1], started, normalization, diagnostics=diagnostics, inputs=inputs
    )
    if normalization.failure is not None:
        return BoardPreparationResult(
            board=board,
            board_view=None,
            failure=ColorPipelineFailure(_STAGE_NAMES[1], normalization.failure),
            steps=(localization_step, normalization_step),
            diagnostics_enabled=diagnostics,
        )
    board_view = normalization.value
    if not isinstance(board_view, NormalizedBoardView):
        raise TypeError("Board normalization returned an invalid result.")
    return BoardPreparationResult(
        board=board,
        board_view=board_view,
        failure=None,
        steps=(localization_step, normalization_step),
        diagnostics_enabled=diagnostics,
    )


class ColorBasedDartRecognitionPipeline:
    """Orchestrate replaceable board-aware stages for multi-dart scoring."""

    def __init__(
        self,
        *,
        orientation: OrientationStage,
        segmentation: SegmentationStage,
        tip_localization: TipLocalizationStage,
        scoring: ScoringStage,
        board_localization: BoardLocalizationStage = detect_board_stage,
        board_normalization: BoardNormalizationStage = normalize_board_view,
        normalized_size_px: int = 1024,
        boundary_margin_mm: float = 2.5,
    ) -> None:
        if normalized_size_px < 2:
            raise ValueError("Normalized board size must be at least 2 pixels.")
        if not isfinite(boundary_margin_mm) or boundary_margin_mm < 0:
            raise ValueError("Boundary margin must be finite and non-negative.")
        self._orientation = orientation
        self._segmentation = segmentation
        self._tip_localization = tip_localization
        self._scoring = scoring
        self._board_localization = board_localization
        self._board_normalization = board_normalization
        self._normalized_size_px = normalized_size_px
        self._boundary_margin_mm = boundary_margin_mm

    def run(
        self,
        image: Image,
        *,
        expected_dart_count: int = 3,
        diagnostics: bool = False,
    ) -> ColorPipelineResult:
        """Run the pipeline once; expected stage failures yield a partial result."""
        _validate_image(image)
        if (
            isinstance(expected_dart_count, bool)
            or not isinstance(expected_dart_count, int)
            or expected_dart_count < 1
        ):
            raise ValueError("Expected dart count must be a positive integer.")

        steps: list[ColorPipelineStep] = []
        board: BoardGeometry | None = None
        orientation: OrientationEstimate | None = None
        entries = _unknown_entries(expected_dart_count)

        def record_failure(
            stage_index: int, step: ColorPipelineStep
        ) -> ColorPipelineResult:
            steps.append(step)
            for skipped_name in _STAGE_NAMES[stage_index + 1 :]:
                steps.append(
                    ColorPipelineStep(
                        name=skipped_name,
                        status=StageStatus.SKIPPED,
                        elapsed_ms=0.0,
                        failure=f"Skipped after {step.name} failed.",
                    )
                )
            darts = tuple(
                DartResult(
                    entry=entry,
                    score=None,
                    uncertainty_reasons=_entry_reasons(entry, orientation),
                )
                for entry in entries
            )
            return ColorPipelineResult(
                status=ColorPipelineStatus.PARTIAL,
                expected_dart_count=expected_dart_count,
                board=board,
                orientation=orientation,
                darts=darts,
                total_points=None,
                total_complete=False,
                total_uncertain=any(dart.uncertainty_reasons for dart in darts),
                failure=ColorPipelineFailure(
                    stage=step.name, message=step.failure or "Stage failed."
                ),
                steps=tuple(steps),
                diagnostics_enabled=diagnostics,
            )

        preparation = prepare_board(
            image,
            normalized_size_px=self._normalized_size_px,
            diagnostics=diagnostics,
            board_localization=self._board_localization,
            board_normalization=self._board_normalization,
        )
        board = preparation.board
        if preparation.failure is not None:
            for index, step in enumerate(preparation.steps):
                if step.status is StageStatus.FAILED:
                    return record_failure(index, step)
                steps.append(step)
            raise ValueError("Board preparation failure has no failed stage.")
        board_view = preparation.board_view
        if board is None or board_view is None:
            raise ValueError("Successful board preparation must include geometry and view.")
        steps.extend(preparation.steps)

        stage_inputs = _snapshot_artifacts(
            {
                "normalized_board": board_view.image,
                "board_mask": board_view.board_mask,
            },
            diagnostics,
        )
        started = perf_counter()
        orientation_outcome = self._orientation(board_view, diagnostics=diagnostics)
        step = _stage_record(
            _STAGE_NAMES[2],
            started,
            orientation_outcome,
            diagnostics=diagnostics,
            inputs=stage_inputs,
        )
        if orientation_outcome.failure is not None:
            return record_failure(2, step)
        orientation = orientation_outcome.value
        if not isinstance(orientation, OrientationEstimate):
            raise TypeError("Orientation stage returned an invalid result.")
        steps.append(step)

        stage_inputs = _snapshot_artifacts(
            {
                "normalized_board": board_view.image,
                "board_mask": board_view.board_mask,
            },
            diagnostics,
        )
        started = perf_counter()
        segmentation = self._segmentation(
            board_view,
            board,
            orientation,
            diagnostics=diagnostics,
        )
        step = _stage_record(
            _STAGE_NAMES[3],
            started,
            segmentation,
            diagnostics=diagnostics,
            inputs=stage_inputs,
        )
        if segmentation.failure is not None:
            return record_failure(3, step)
        likelihood = segmentation.value
        if not isinstance(likelihood, DartLikelihoodMap):
            raise TypeError("Segmentation returned an invalid result.")
        if likelihood.probabilities.shape != board_view.image.shape[:2]:
            raise ValueError("Dart likelihood must match the normalized board dimensions.")
        steps.append(step)

        stage_inputs = _snapshot_artifacts(
            {
                "image": image,
                "normalized_board": board_view.image,
                "board_mask": board_view.board_mask,
                "dart_likelihood": likelihood.probabilities,
            },
            diagnostics,
        )
        started = perf_counter()
        localization = self._tip_localization(
            image,
            board_view,
            board,
            orientation,
            likelihood,
            expected_dart_count,
            diagnostics=diagnostics,
        )
        step = _stage_record(
            _STAGE_NAMES[4],
            started,
            localization,
            diagnostics=diagnostics,
            inputs=stage_inputs,
        )
        if localization.failure is not None:
            return record_failure(4, step)
        localized = localization.value
        if not isinstance(localized, tuple) or not all(
            isinstance(entry, BoardEntryEstimate) for entry in localized
        ):
            raise TypeError("Tip localization must return a tuple of board-entry estimates.")
        if len(localized) > expected_dart_count:
            raise ValueError("Tip localization returned more entries than expected.")
        entries = _ordered_entries(localized, expected_dart_count)
        steps.append(step)

        stage_inputs: Mapping[str, NDArray[np.generic]] = {}
        started = perf_counter()
        scoring = self._scoring(
            board,
            orientation,
            entries,
            self._boundary_margin_mm,
            diagnostics=diagnostics,
        )
        step = _stage_record(
            _STAGE_NAMES[5],
            started,
            scoring,
            diagnostics=diagnostics,
            inputs=stage_inputs,
        )
        if scoring.failure is not None:
            return record_failure(5, step)
        scores = scoring.value
        if not isinstance(scores, tuple) or not all(
            score is None or isinstance(score, ScoreEstimate) for score in scores
        ):
            raise TypeError("Scoring must return a tuple of score estimates or unknowns.")
        if len(scores) != expected_dart_count:
            raise ValueError("Scoring must return one score slot per expected dart.")
        if any(
            entry.xy is None and score is not None
            for entry, score in zip(entries, scores)
        ):
            raise ValueError("An unresolved board entry cannot have a score.")
        steps.append(step)

        darts = tuple(
            _make_dart_result(entry, score, orientation)
            for entry, score in zip(entries, scores, strict=True)
        )
        total_points = (
            sum(dart.score.points for dart in darts if dart.score is not None)
            if all(dart.score is not None for dart in darts)
            else None
        )
        total_complete = total_points is not None
        return ColorPipelineResult(
            status=ColorPipelineStatus.COMPLETED,
            expected_dart_count=expected_dart_count,
            board=board,
            orientation=orientation,
            darts=darts,
            total_points=total_points,
            total_complete=total_complete,
            total_uncertain=any(dart.uncertainty_reasons for dart in darts),
            failure=None,
            steps=tuple(steps),
            diagnostics_enabled=diagnostics,
        )


def _unknown_entries(count: int) -> tuple[BoardEntryEstimate, ...]:
    return tuple(
        BoardEntryEstimate(
            xy=None,
            confidence=0.0,
            provenance=EntryProvenance.UNRESOLVED,
            uncertainty_reasons=(UncertaintyReason.ENTRY_UNLOCALIZED,),
        )
        for _ in range(count)
    )


def _ordered_entries(
    localized: tuple[BoardEntryEstimate, ...], expected_count: int
) -> tuple[BoardEntryEstimate, ...]:
    def source_xy(entry: BoardEntryEstimate) -> tuple[float, float]:
        if entry.xy is None:
            raise ValueError("An unresolved entry has no source-image ordering.")
        return entry.xy

    found = sorted(
        (entry for entry in localized if entry.xy is not None),
        key=source_xy,
    )
    unresolved = [entry for entry in localized if entry.xy is None]
    unresolved.extend(_unknown_entries(expected_count - len(localized)))
    return tuple(found + unresolved)


def _make_dart_result(
    entry: BoardEntryEstimate,
    score: ScoreEstimate | None,
    orientation: OrientationEstimate,
) -> DartResult:
    reasons = list(_entry_reasons(entry, orientation))
    if score is not None:
        reasons.extend(score.uncertainty_reasons)
    elif entry.xy is not None:
        reasons.append(UncertaintyReason.SCORE_UNAVAILABLE)
    return DartResult(
        entry=entry,
        score=score,
        uncertainty_reasons=tuple(dict.fromkeys(reasons)),
    )


def _entry_reasons(
    entry: BoardEntryEstimate,
    orientation: OrientationEstimate | None,
) -> tuple[UncertaintyReason, ...]:
    reasons = list(entry.uncertainty_reasons)
    if (
        entry.provenance is EntryProvenance.INFERRED
        and UncertaintyReason.ENTRY_INFERRED not in reasons
    ):
        reasons.append(UncertaintyReason.ENTRY_INFERRED)
    if entry.xy is None and UncertaintyReason.ENTRY_UNLOCALIZED not in reasons:
        reasons.append(UncertaintyReason.ENTRY_UNLOCALIZED)
    if orientation is not None and not orientation.confident:
        reasons.append(UncertaintyReason.ORIENTATION_UNCERTAIN)
    return tuple(dict.fromkeys(reasons))


def format_timing_summary(result: ColorPipelineResult | BoardPreparationResult) -> str:
    """Format per-stage and substep durations without printing."""
    total_ms = sum(step.elapsed_ms for step in result.steps)
    lines = [f"Total: {total_ms:.2f} ms"]
    for step in result.steps:
        lines.append(f"{step.name}: {step.elapsed_ms:.2f} ms ({step.status.value})")
        lines.extend(
            f"  {name}: {duration:.2f} ms"
            for name, duration in step.substep_timings_ms.items()
        )
    return "\n".join(lines)
