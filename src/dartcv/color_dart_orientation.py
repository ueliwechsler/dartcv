"""Local, template-based printed-20 orientation without OCR or image I/O."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .color_dart_pipeline import (
    Image,
    NormalizedBoardView,
    OrientationEstimate,
    StageOutcome,
    _validate_image,
)

SAMPLES_PER_DEGREE = 4
INNER_RADIUS_MM = 194.0
OUTER_RADIUS_MM = 236.0
RADIAL_SAMPLES = 85
ANGULAR_SAMPLES = 360 * SAMPLES_PER_DEGREE


@dataclass(frozen=True, slots=True)
class NumberRingStrip:
    """Annulus unwrapped clockwise from -180 degrees; outer edge is row zero."""

    image: Image
    source_coverage: Image


def unwrap_number_ring(view: NormalizedBoardView) -> NumberRingStrip:
    """Sample the optional number view, never the cropped scoring canvas."""
    ring = view.number_ring
    if ring is None:
        raise ValueError("Normalization must include a wider number-ring view.")
    angles = np.deg2rad(np.arange(ANGULAR_SAMPLES) / SAMPLES_PER_DEGREE - 180.0)
    radii = np.linspace(OUTER_RADIUS_MM, INNER_RADIUS_MM, RADIAL_SAMPLES)
    x = radii[:, None] * np.sin(angles)
    y = -radii[:, None] * np.cos(angles)
    model = np.stack((x, y), axis=-1).astype(np.float32)
    pixels = cv2.perspectiveTransform(model, ring.model_to_view)
    map_x, map_y = pixels[..., 0], pixels[..., 1]
    image = cv2.remap(ring.image, map_x, map_y, cv2.INTER_LINEAR)
    # Linear sampling plus a strict threshold excludes partially missing borders.
    coverage = cv2.remap(ring.source_coverage, map_x, map_y, cv2.INTER_LINEAR)
    coverage = np.where(coverage == 255, 255, 0).astype(np.uint8)
    return NumberRingStrip(image, coverage)


def extract_printed20_template(
    view: NormalizedBoardView,
    *,
    label_angle_deg: float,
    width_deg: float = 14.0,
    inner_radius_mm: float = 202.0,
    outer_radius_mm: float = 228.0,
) -> Image:
    """Extract a manually identified 20, not an automatically guessed label.

    Angle is clockwise from model-space top. The returned BGR patch is an
    explicit reusable template; callers own template selection and persistence.
    """
    if not np.isfinite([label_angle_deg, width_deg, inner_radius_mm, outer_radius_mm]).all():
        raise ValueError("Template coordinates must be finite.")
    if not 2.0 <= width_deg < 18.0:
        raise ValueError("Template width must be between 2 and 18 degrees.")
    if not INNER_RADIUS_MM <= inner_radius_mm < outer_radius_mm <= OUTER_RADIUS_MM:
        raise ValueError("Template radii must lie within the sampled number annulus.")
    strip = unwrap_number_ring(view)
    width = round(width_deg * SAMPLES_PER_DEGREE)
    center = (label_angle_deg + 180.0) * SAMPLES_PER_DEGREE
    columns = np.rint(center + np.arange(width) - (width - 1) / 2).astype(int)
    columns %= ANGULAR_SAMPLES
    radial_scale = (RADIAL_SAMPLES - 1) / (OUTER_RADIUS_MM - INNER_RADIUS_MM)
    top = round((OUTER_RADIUS_MM - outer_radius_mm) * radial_scale)
    bottom = round((OUTER_RADIUS_MM - inner_radius_mm) * radial_scale) + 1
    coverage = strip.source_coverage[top:bottom, columns]
    if not np.all(coverage == 255):
        raise ValueError("The selected template contains missing source pixels.")
    patch = strip.image[top:bottom, columns].copy()
    if min(patch.shape[:2]) < 4:
        raise ValueError("The selected template is too small.")
    return patch


def _features(image: Image) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), 2.0)


def _wrap_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class Printed20OrientationStage:
    """Match an explicit board-style template around the complete number ring.

    Confidence is a heuristic correlation strength, NOT a calibrated probability.
    Weak matches retain an uncertain best guess; insufficient evidence fails.
    Instances cache template features only, never the most recent run.
    """

    def __init__(
        self,
        template: Image,
        *,
        scales: tuple[float, ...] = (0.9, 1.0, 1.1),
        minimum_evidence: float = 0.25,
        confident_match: float = 0.65,
        ambiguity_margin: float = 0.10,
    ) -> None:
        _validate_image(template)
        if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError("Template scales must be finite and positive.")
        if not 0 <= minimum_evidence <= confident_match <= 1:
            raise ValueError("Evidence thresholds must satisfy 0 <= minimum <= confident <= 1.")
        if not np.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1:
            raise ValueError("Ambiguity margin must be between zero and one.")
        self.minimum_evidence = minimum_evidence
        self.confident_match = confident_match
        self.ambiguity_margin = ambiguity_margin
        self._templates: list[np.ndarray] = []
        for scale in scales:
            width, height = (round(template.shape[1] * scale), round(template.shape[0] * scale))
            if not 4 <= height <= RADIAL_SAMPLES or not 4 <= width < 72:
                raise ValueError("Scaled templates must fit the annulus and be narrower than 18 degrees.")
            features = _features(cv2.resize(template, (width, height)))
            if float(np.std(features)) < 1.0:
                raise ValueError("A printed-20 template must contain visible contrast.")
            self._templates.append(features)
        self.template = template.copy()

    def __call__(
        self, view: NormalizedBoardView, *, diagnostics: bool
    ) -> StageOutcome[OrientationEstimate]:
        if view.number_ring is None:
            return StageOutcome.failed("Printed-20 orientation requires a wider number-ring view.")
        start = perf_counter()
        strip = unwrap_number_ring(view)
        unwrap_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        features = _features(strip.image)
        angular_response = np.full(ANGULAR_SAMPLES, -1.0, dtype=np.float32)
        valid = (strip.source_coverage == 255).astype(np.float32)
        for template in self._templates:
            height, width = template.shape
            left = (width - 1) // 2
            right = width - 1 - left
            padded = np.pad(features, ((0, 0), (left, right)), mode="wrap")
            padded_coverage = np.pad(valid, ((0, 0), (left, right)), mode="wrap")
            response = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
            integral = cv2.integral(padded_coverage, sdepth=cv2.CV_64F)
            coverage = (
                integral[height:, width:] - integral[:-height, width:]
                - integral[height:, :-width] + integral[:-height, :-width]
            )
            response[coverage < height * width] = -1.0
            angular_response = np.maximum(angular_response, response.max(axis=0))

        # Distinct hypotheses, not neighboring pixels of the same match.
        remaining = angular_response.copy()
        candidates: list[tuple[float, float]] = []
        columns = np.arange(ANGULAR_SAMPLES)
        for _ in range(5):
            index = int(np.argmax(remaining))
            strength = float(remaining[index])
            if strength < 0:
                break
            angle = _wrap_angle(index / SAMPLES_PER_DEGREE - 180.0)
            candidates.append((angle, strength))
            distance = np.abs(columns - index)
            distance = np.minimum(distance, ANGULAR_SAMPLES - distance)
            remaining[distance < 18 * SAMPLES_PER_DEGREE] = -1.0
        match_ms = (perf_counter() - start) * 1000
        artifacts = {}
        if diagnostics:
            artifacts = {
                "number_ring": view.number_ring.image,
                "number_ring_model_to_view": view.number_ring.model_to_view,
                "number_strip": strip.image,
                "number_strip_coverage": strip.source_coverage,
                "printed20_template": self.template.copy(),
                "angular_response": angular_response,
                "orientation_candidates": np.array(candidates, dtype=np.float64).reshape(-1, 2),
            }
        timings = {"unwrap_number_ring": unwrap_ms, "match_printed20": match_ms}
        if not candidates or candidates[0][1] < self.minimum_evidence:
            return StageOutcome.failed(
                "No supported printed-20 match in the observed number ring.",
                artifacts=artifacts,
                substep_timings_ms=timings,
            )
        angle, strength = candidates[0]
        runner_up = candidates[1][1] if len(candidates) > 1 else -1.0
        estimate = OrientationEstimate(
            correction_deg=_wrap_angle(-angle),
            confidence=float(np.clip(strength, 0, 1)),
            confident=strength >= self.confident_match
            and strength - runner_up >= self.ambiguity_margin,
        )
        return StageOutcome.succeeded(
            estimate, artifacts=artifacts, substep_timings_ms=timings
        )
