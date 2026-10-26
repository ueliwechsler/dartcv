"""Geometry-constrained, per-image board colors and initial dart anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .color_dart_pipeline import (
    BOARD_RADII_MM, BoardGeometry, DartLikelihoodMap, NormalizedBoardView,
    OrientationEstimate, StageOutcome,
)

COLOR_NAMES = ("red", "green", "beige", "black")
OUTSIDE_FIELD = 255


class InsufficientColorSamples(ValueError):
    """Expected failure when observed interiors cannot support all four models."""


@dataclass(frozen=True, slots=True)
class FieldMasks:
    """Expected color IDs, trustworthy sampling interiors, and wire exclusions."""

    expected_color: NDArray[np.uint8]
    sample_interior: NDArray[np.bool_]
    wire_region: NDArray[np.bool_]
    observed_board: NDArray[np.bool_]
    pixels_per_mm: float = 1.0


@dataclass(frozen=True, slots=True)
class BoardColorModel:
    """Robust diagonal distribution in value-weighted HSV cone coordinates."""

    name: str
    center: NDArray[np.float32]
    scale: NDArray[np.float32]
    sample_count: int
    retained_count: int


def board_color_features(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Continuous hue wrapping; dark/achromatic pixels have little hue weight."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[..., 0] * (2 * np.pi / 180)
    value = hsv[..., 2] / 255
    chroma = (hsv[..., 1] / 255) * value
    return np.stack((chroma * np.cos(hue), chroma * np.sin(hue), value), axis=-1)


def build_field_masks(
    view: NormalizedBoardView,
    orientation: OrientationEstimate,
    *,
    sample_margin_mm: float = 2.5,
    wire_margin_mm: float = 1.0,
) -> FieldMasks:
    """Assign standard fields using the explicit clockwise orientation correction."""
    if (
        not np.isfinite([sample_margin_mm, wire_margin_mm]).all()
        or not 0 <= wire_margin_mm <= sample_margin_mm
    ):
        raise ValueError("Margins must satisfy 0 <= wire <= sampling margin, in mm.")
    height, width = view.image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    points = cv2.perspectiveTransform(
        np.stack((xx, yy), axis=-1), np.linalg.inv(view.model_to_view)
    )
    x, y = points[..., 0], points[..., 1]
    radius = np.hypot(x, y)
    angle = np.rad2deg(np.arctan2(x, -y)) + orientation.correction_deg
    sector = np.floor((angle + 9) / 18).astype(np.int32) % 20
    colored_ring = (
        ((radius >= BOARD_RADII_MM["triple_inner"])
         & (radius <= BOARD_RADII_MM["triple_outer"]))
        | ((radius >= BOARD_RADII_MM["double_inner"])
           & (radius <= BOARD_RADII_MM["double_outer"]))
    )
    expected = np.where(colored_ring, sector % 2, np.where(sector % 2 == 0, 3, 2))
    expected[radius <= BOARD_RADII_MM["outer_bull"]] = 1
    expected[radius <= BOARD_RADII_MM["inner_bull"]] = 0
    if view.source_coverage is None:
        raise ValueError("Color modeling requires observed source coverage from normalization.")
    observed = (
        (radius <= BOARD_RADII_MM["double_outer"])
        & (view.board_mask > 0) & (view.source_coverage == 255)
    )
    expected = np.where(observed, expected, OUTSIDE_FIELD).astype(np.uint8)
    ring_distance = np.full(radius.shape, np.inf, np.float32)
    for ring_radius in BOARD_RADII_MM.values():
        np.minimum(ring_distance, np.abs(radius - ring_radius), out=ring_distance)
    # Sector boundaries are at +/-9 degrees; bull interiors have no radial wires.
    offset = np.abs((angle + 9) % 18 - 9)
    spoke_distance = radius * np.sin(np.deg2rad(9 - offset))
    spoke_distance[radius <= BOARD_RADII_MM["outer_bull"]] = np.inf
    wire_distance = np.minimum(ring_distance, spoke_distance)
    origin = np.array([[[0., 0.], [1., 0.], [0., 1.]]], np.float32)
    pixels = cv2.perspectiveTransform(origin, view.model_to_view)[0]
    pixels_per_mm = float(np.sqrt(abs(np.linalg.det(
        np.stack((pixels[1] - pixels[0], pixels[2] - pixels[0]))
    ))))
    return FieldMasks(
        expected, observed & (wire_distance > sample_margin_mm),
        observed & (wire_distance <= wire_margin_mm), observed, pixels_per_mm,
    )


def fit_board_color_models(
    features: NDArray[np.float32],
    masks: FieldMasks,
    *,
    minimum_samples: int = 64,
    maximum_samples: int = 12000,
) -> tuple[BoardColorModel, ...]:
    """Pool each color's field interiors; median/MAD trim sparse dart pollution."""
    if minimum_samples < 4 or maximum_samples < minimum_samples:
        raise ValueError("Sample limits must satisfy 4 <= minimum <= maximum.")
    models = []
    floors = np.array([0.025, 0.025, 0.05], np.float32)
    for color_id, name in enumerate(COLOR_NAMES):
        samples = features[masks.sample_interior & (masks.expected_color == color_id)]
        if len(samples) < minimum_samples:
            raise InsufficientColorSamples(f"Insufficient {name} field-interior samples: {len(samples)}.")
        if len(samples) > maximum_samples:
            samples = samples[np.linspace(0, len(samples) - 1, maximum_samples, dtype=int)]
        retained = samples
        for _ in range(3):
            center = np.median(retained, axis=0)
            scale = np.maximum(1.4826 * np.median(np.abs(retained - center), axis=0), floors)
            distance = np.linalg.norm((samples - center) / scale, axis=1)
            retained = samples[distance <= 4.5]
            if len(retained) < minimum_samples:
                raise InsufficientColorSamples(f"Insufficient uncontaminated {name} samples after trimming.")
        center = np.median(retained, axis=0).astype(np.float32)
        scale = np.maximum(
            1.4826 * np.median(np.abs(retained - center), axis=0), floors
        ).astype(np.float32)
        models.append(BoardColorModel(name, center, scale, len(samples), len(retained)))
    return tuple(models)


def fit_global_color_models(
    features: NDArray[np.float32],
    observed_board: NDArray[np.bool_],
    *,
    minimum_samples: int = 64,
    maximum_samples: int = 16000,
) -> tuple[BoardColorModel, ...]:
    """Experimental four-mode distribution of all observed board pixels.

    No numbered-field assignment or wire exclusion is used for fitting. Clusters
    are unlabeled modes, not guaranteed to correspond to the four paint colors.
    """
    from sklearn.cluster import KMeans

    if minimum_samples < 4 or maximum_samples < 4 * minimum_samples:
        raise ValueError("Global sample limits must support four color modes.")
    samples = features[observed_board]
    if len(samples) < 4 * minimum_samples:
        raise InsufficientColorSamples("Insufficient observed board samples for four global modes.")
    if len(samples) > maximum_samples:
        samples = samples[np.linspace(0, len(samples) - 1, maximum_samples, dtype=int)]
    if len(np.unique(samples, axis=0)) < 4:
        raise InsufficientColorSamples("The observed board has fewer than four distinct colors.")
    clustering = KMeans(n_clusters=4, n_init=3, random_state=0).fit(samples)
    models = []
    floors = np.array([0.025, 0.025, 0.05], np.float32)
    for index in range(4):
        cluster = samples[clustering.labels_ == index]
        retained = cluster
        for _ in range(3):
            if len(retained) < minimum_samples:
                raise InsufficientColorSamples(f"Insufficient support for global color mode {index}.")
            center = np.median(retained, axis=0)
            scale = np.maximum(1.4826 * np.median(np.abs(retained - center), axis=0), floors)
            distance = np.linalg.norm((retained - center) / scale, axis=1)
            retained = retained[distance <= 4.5]
        if len(retained) < minimum_samples:
            raise InsufficientColorSamples(f"Insufficient support for global color mode {index}.")
        center = np.median(retained, axis=0).astype(np.float32)
        scale = np.maximum(
            1.4826 * np.median(np.abs(retained - center), axis=0), floors
        ).astype(np.float32)
        models.append(BoardColorModel(
            f"mode_{index}", center, scale, len(cluster), len(retained)
        ))
    return tuple(models)


def keep_color_anomalies(
    probabilities: NDArray[np.float32], masks: FieldMasks, minimum_area_px: int
) -> NDArray[np.float32]:
    """Comparison baseline: no grid or component removal, but respect coverage."""
    del minimum_area_px
    return np.where(masks.observed_board, probabilities, 0).astype(np.float32)


def suppress_wire_artifacts(
    probabilities: NDArray[np.float32], masks: FieldMasks, minimum_area_px: int
) -> NDArray[np.float32]:
    """Replaceable conservative cleanup; raw wire-crossing evidence stays diagnostic."""
    cleaned = probabilities.copy()
    cleaned[masks.wire_region | ~masks.observed_board] = 0
    support = (cleaned >= 0.5).astype(np.uint8)
    support = cv2.morphologyEx(
        support, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area_px
    cleaned[~keep[labels]] = 0
    return cleaned


def preserve_dart_support(
    probabilities: NDArray[np.float32], masks: FieldMasks, minimum_area_px: int
) -> NDArray[np.float32]:
    """Keep thin interior components and restore only nearby wire-crossing evidence.

    Wire pixels never seed components. A single bounded dilation can restore
    raw evidence within 3 mm of retained interiors, without flooding the grid.
    No opening is applied: it would erase narrow shafts.
    """
    support = (
        (probabilities >= 0.2) & masks.sample_interior & masks.observed_board
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area_px
    core = keep[labels]
    radius = max(1, round(3.0 * masks.pixels_per_mm))
    near_core = cv2.dilate(
        core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    ) > 0
    retain = core | (~masks.sample_interior & near_core & masks.observed_board)
    return np.where(retain, probabilities, 0).astype(np.float32)


CleanupStage = Callable[
    [NDArray[np.float32], FieldMasks, int], NDArray[np.float32]
]
ColorAssignment = Literal["field", "board_union", "global"]


class RobustBoardColorSegmentationStage:
    """Initial anomaly scores, not a trained/calibrated dart classifier or tip finder."""

    def __init__(
        self,
        *,
        sample_margin_mm: float = 2.5,
        wire_margin_mm: float = 1.0,
        minimum_samples: int = 64,
        minimum_component_area_mm2: float = 3.0,
        cleanup: CleanupStage = preserve_dart_support,
        color_assignment: ColorAssignment = "field",
    ) -> None:
        if (
            not np.isfinite([sample_margin_mm, wire_margin_mm, minimum_component_area_mm2]).all()
            or not 0 <= wire_margin_mm <= sample_margin_mm
            or minimum_component_area_mm2 < 0
            or isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int)
            or not 4 <= minimum_samples <= 12000
        ):
            raise ValueError("Invalid sampling, wire, component-area or sample-count settings.")
        if color_assignment not in ("field", "board_union", "global"):
            raise ValueError("Color assignment must be field, board_union or global.")
        self.sample_margin_mm = sample_margin_mm
        self.wire_margin_mm = wire_margin_mm
        self.minimum_samples = minimum_samples
        self.minimum_component_area_mm2 = minimum_component_area_mm2
        self.cleanup = cleanup
        self.color_assignment = color_assignment

    def __call__(
        self, view: NormalizedBoardView, board: BoardGeometry,
        orientation: OrientationEstimate, *, diagnostics: bool,
    ) -> StageOutcome[DartLikelihoodMap]:
        del board
        timings: dict[str, float] = {}
        artifacts: dict[str, np.ndarray] = {}
        start = perf_counter()
        if view.source_coverage is None:
            return StageOutcome.failed("Color modeling requires observed source coverage.")
        # The detector already aligns alternating paint modulo 36 degrees.
        # An untrusted numeral must not redefine that paint alignment.
        color_orientation = orientation if orientation.confident else OrientationEstimate(0, 0, False)
        masks = build_field_masks(
            view, color_orientation, sample_margin_mm=self.sample_margin_mm,
            wire_margin_mm=self.wire_margin_mm,
        )
        features = board_color_features(view.image)
        timings["field_masks_and_features"] = (perf_counter() - start) * 1000
        if diagnostics:
            artifacts.update(
                expected_field_color=masks.expected_color,
                field_sample_interiors=masks.sample_interior.astype(np.uint8) * 255,
                expected_wires=masks.wire_region.astype(np.uint8) * 255,
                source_coverage=view.source_coverage.copy(),
                color_alignment_correction_deg=np.array([color_orientation.correction_deg]),
                color_alignment_uses_detector=np.array([not orientation.confident], dtype=np.uint8),
                color_assignment=np.array([self.color_assignment]),
            )
        start = perf_counter()
        try:
            if self.color_assignment == "global":
                models = fit_global_color_models(
                    features, masks.observed_board, minimum_samples=self.minimum_samples,
                    maximum_samples=max(16000, 4 * self.minimum_samples),
                )
            else:
                models = fit_board_color_models(features, masks, minimum_samples=self.minimum_samples)
        except InsufficientColorSamples as exc:
            timings["fit_color_models"] = (perf_counter() - start) * 1000
            return StageOutcome.failed(str(exc), artifacts=artifacts, substep_timings_ms=timings)
        timings["fit_color_models"] = (perf_counter() - start) * 1000
        start = perf_counter()
        raw = np.zeros(view.image.shape[:2], np.float32)
        if self.color_assignment == "field":
            for color_id, model in enumerate(models):
                field = masks.expected_color == color_id
                distance = np.linalg.norm((features[field] - model.center) / model.scale, axis=1)
                raw[field] = np.clip((distance - 3.0) / 5.0, 0, 1)
        else:
            observed_features = features[masks.observed_board]
            distance = np.full(len(observed_features), np.inf, np.float32)
            for model in models:
                np.minimum(
                    distance, np.linalg.norm((observed_features - model.center) / model.scale, axis=1),
                    out=distance,
                )
            raw[masks.observed_board] = np.clip((distance - 3.0) / 5.0, 0, 1)
        timings["color_anomalies"] = (perf_counter() - start) * 1000
        start = perf_counter()
        minimum_area = max(1, round(self.minimum_component_area_mm2 * masks.pixels_per_mm**2))
        cleaned = self.cleanup(raw.copy(), masks, minimum_area)
        if cleaned.shape != raw.shape:
            raise ValueError("Segmentation cleanup changed the likelihood-map shape.")
        likelihood = DartLikelihoodMap(cleaned)
        timings["wire_component_cleanup"] = (perf_counter() - start) * 1000
        if diagnostics:
            artifacts.update(
                normalized_board=view.image.copy(),
                color_model_centers=np.stack([model.center for model in models]),
                color_model_scales=np.stack([model.scale for model in models]),
                color_model_support=np.array(
                    [[model.sample_count, model.retained_count] for model in models], dtype=np.int32
                ),
                color_model_names=np.array([model.name for model in models]),
                raw_color_anomalies=raw,
                dart_likelihood=cleaned.copy(),
                orientation_confident=np.array([orientation.confident], dtype=np.uint8),
            )
        return StageOutcome.succeeded(likelihood, artifacts=artifacts, substep_timings_ms=timings)
