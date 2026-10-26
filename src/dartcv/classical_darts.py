"""Experimental single-dart detection from flight colour and shaft geometry.

Works on the original perspective within an expanded calibrated-board region.
This is a visible-flight prototype, not a general dart recognizer or
tip locator. Heuristic quality scores are not probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import RADII, polar, project
from .recognition import DartBox, validate_boxes


@dataclass
class DartHypothesis:
    """One flight/paired-shaft explanation, all coordinates at working scale."""

    flight: np.ndarray
    shaft_edges: np.ndarray
    bounds: np.ndarray
    score: float
    overlap: float
    contrast: float


@dataclass
class ClassicalDartDiagnostics:
    """Latest attempt; working pixels map to source pixels by division by scale_xy."""

    image: np.ndarray
    scale_xy: np.ndarray
    search_mask: np.ndarray
    flight_mask: np.ndarray
    flights: list[np.ndarray]
    segments: np.ndarray
    hypotheses: list[DartHypothesis] = field(default_factory=list)
    message: str = ""
    accepted: bool = False


class ClassicalDartDetector:
    """Callable returning zero or one whole-dart box in original-image pixels.

    H maps board millimetres to source pixels. The search extends to 1.4 times
    the outer-double radius, without rectifying protruding darts. Automatic
    flight proposals use local chromatic contrast and bright neutral contrast;
    dark flights remain unsupported. flight_hsv supplies inclusive OpenCV HSV
    bounds, with hue wrapping when the lower hue exceeds the upper hue.
    """

    def __init__(
        self,
        H: np.ndarray,
        *,
        flight_hsv: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
        work_size: int = 1200,
        min_score: float = 0.65,
        ambiguity_margin: float = 0.05,
    ) -> None:
        matrix = np.asarray(H, dtype=float)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) < 3
        ):
            raise ValueError("H must be a finite, invertible 3 x 3 homography")
        if not isinstance(work_size, int) or work_size < 100:
            raise ValueError("work_size must be an integer >= 100")
        if not 0 <= min_score <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("quality thresholds must be in [0, 1]")
        if flight_hsv is not None:
            bounds = np.asarray(flight_hsv, dtype=float)
            if (
                bounds.shape != (2, 3)
                or not np.isfinite(bounds).all()
                or not (bounds == np.floor(bounds)).all()
                or (bounds < 0).any()
                or (bounds > [179, 255, 255]).any()
                or (bounds[0, 1:] > bounds[1, 1:]).any()
            ):
                raise ValueError(
                    "flight_hsv must contain valid lower/upper OpenCV HSV bounds"
                )
        self.H = matrix.copy()
        self.flight_hsv = flight_hsv
        self.work_size = work_size
        self.min_score = min_score
        self.ambiguity_margin = ambiguity_margin
        self.diagnostics: ClassicalDartDiagnostics | None = None

    def _flight_mask(self, image: np.ndarray, search: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.flight_hsv is not None:
            lower, upper = (np.asarray(bound, np.uint8) for bound in self.flight_hsv)
            if lower[0] <= upper[0]:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = cv2.inRange(
                    hsv, lower, np.array([179, upper[1], upper[2]], np.uint8)
                )
                mask |= cv2.inRange(
                    hsv, np.array([0, lower[1], lower[2]], np.uint8), upper
                )
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            background = cv2.GaussianBlur(lab[..., 1:], (0, 0), 9)
            contrast = np.linalg.norm(lab[..., 1:] - background, axis=2)
            light_background = cv2.GaussianBlur(lab[..., 0], (0, 0), 9)
            coloured = (contrast > 12) & (hsv[..., 1] > 60) & (hsv[..., 2] > 45)
            neutral = (
                (lab[..., 0] - light_background > 15)
                & (hsv[..., 1] < 65)
                & (hsv[..., 2] > 180)
            )
            neutral_mask = cv2.morphologyEx(
                neutral.astype(np.uint8) * 255,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            mask = coloured.astype(np.uint8) * 255 | neutral_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.bitwise_and(mask, search)

    def _shaft_hypotheses(
        self, gray: np.ndarray, flight: np.ndarray, segments: np.ndarray
    ) -> list[DartHypothesis]:
        if not len(segments):
            return []
        moments = cv2.moments(flight)
        centre = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
        _, _, width, height = cv2.boundingRect(flight)
        flight_size = float(max(width, height))
        endpoints = segments.reshape(-1, 2, 2).copy()
        distances = np.linalg.norm(endpoints - centre, axis=2)
        swap = distances[:, 0] > distances[:, 1]
        endpoints[swap] = endpoints[swap, ::-1]
        lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
        nearby = (
            (distances.min(axis=1) < 0.9 * flight_size)
            & (lengths > 0.7 * flight_size)
            & (lengths < 6 * flight_size)
        )
        attachment = np.array(
            [
                cv2.pointPolygonTest(
                    flight, tuple(float(value) for value in edge[0]), True
                )
                for edge in endpoints
            ]
        )
        nearby &= attachment >= -0.12 * flight_size
        endpoints, lengths = endpoints[nearby], lengths[nearby]
        if not len(endpoints):
            return []
        directions = (endpoints[:, 1] - endpoints[:, 0]) / lengths[:, None]
        hypotheses = []
        solidity = cv2.contourArea(flight) / cv2.contourArea(cv2.convexHull(flight))
        for index, (edge, length, direction) in enumerate(
            zip(endpoints, lengths, directions)
        ):
            normal = np.array([-direction[1], direction[0]])
            if (edge[1] - centre) @ direction < flight_size:
                continue
            along = (endpoints - edge[0]) @ direction
            start = np.maximum(0, along.min(axis=1))
            stop = np.minimum(length, along.max(axis=1))
            overlap = np.maximum(0, stop - start) / length
            separation = (endpoints.mean(axis=1) - edge.mean(axis=0)) @ normal
            parallel = np.abs(directions @ direction) > np.cos(np.deg2rad(8))
            partners = np.flatnonzero(
                parallel
                & (overlap > 0.6)
                & (np.abs(separation) >= max(1.5, 0.08 * flight_size))
                & (np.abs(separation) < 0.25 * flight_size)
            )
            for partner in partners:
                if partner == index:
                    continue
                offset = separation[partner]
                samples = (
                    edge[0]
                    + np.linspace(start[partner], stop[partner], 24)[:, None]
                    * direction
                )
                middle = samples + normal * offset / 2
                outside = abs(offset) / 2 + 2
                strips = np.stack(
                    [middle, middle + normal * outside, middle - normal * outside]
                )
                values = cv2.remap(
                    gray,
                    strips[..., 0].astype(np.float32),
                    strips[..., 1].astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                ).astype(float)
                contrast = float(
                    np.median(np.abs(values[0] - (values[1] + values[2]) / 2))
                )
                if contrast < 10:
                    continue
                shaft_edges = np.stack([edge, endpoints[partner]])
                points = np.concatenate(
                    [flight.reshape(-1, 2), shaft_edges.reshape(-1, 2)]
                )
                padding = max(2, flight_size * 0.04)
                bounds = np.array(
                    [points.min(axis=0) - padding, points.max(axis=0) + padding]
                )
                score = float(
                    0.4 * overlap[partner]
                    + 0.3 * min(contrast / 40, 1)
                    + 0.15 * solidity
                    + 0.15 * min(length / (2 * flight_size), 1)
                )
                hypotheses.append(
                    DartHypothesis(
                        flight,
                        shaft_edges,
                        bounds,
                        score,
                        float(overlap[partner]),
                        contrast,
                    )
                )
        return hypotheses

    def __call__(self, image: np.ndarray) -> tuple[DartBox, ...]:
        """Return no result for weak/ambiguous evidence; retain latest diagnostics."""
        self.diagnostics = None
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or min(image.shape[:2]) < 2
        ):
            raise ValueError("expected a nonempty H x W x 3 BGR uint8 image")
        height, width = image.shape[:2]
        factor = min(1.0, self.work_size / max(height, width))
        work = cv2.resize(
            image, (max(2, round(width * factor)), max(2, round(height * factor)))
        )
        scale_xy = np.array([work.shape[1] / width, work.shape[0] / height])
        transform = np.diag([*scale_xy, 1.0]) @ self.H
        boundary = project(
            transform,
            [polar(RADII["double_outer"] * 1.4, angle) for angle in range(360)],
        )
        nominal = project(
            transform,
            [polar(RADII["double_outer"], angle) for angle in range(0, 360, 10)],
        )
        centre = project(transform, [[0, 0]])[0]
        radius = float(np.median(np.linalg.norm(nominal - centre, axis=1)))
        if not np.isfinite(boundary).all() or not np.isfinite(radius) or radius < 10:
            raise ValueError(
                "calibrated board is too small or not finite at working scale"
            )
        search = np.zeros(work.shape[:2], np.uint8)
        cv2.fillPoly(
            search, [np.round(np.clip(boundary, -1e6, 1e6)).astype(np.int32)], 255
        )
        mask = self._flight_mask(work, search)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flights = []
        image_to_model = np.linalg.inv(transform)
        for contour in contours:
            area = cv2.contourArea(contour)
            _, _, flight_width, flight_height = cv2.boundingRect(contour)
            if (
                (0.025 * radius) ** 2 < area < (0.4 * radius) ** 2
                and min(flight_width, flight_height) >= 4
                and max(flight_width, flight_height) / min(flight_width, flight_height)
                < 3
                and area / cv2.contourArea(cv2.convexHull(contour)) > 0.45
            ):
                radii = np.linalg.norm(project(image_to_model, contour), axis=1)
                paint = radii < RADII["outer_bull"] + 3
                for band in ("treble", "double"):
                    paint |= (radii > RADII[f"{band}_inner"] - 3) & (
                        radii < RADII[f"{band}_outer"] + 3
                    )
                if paint.mean() < 0.8:
                    flights.append(contour)
        flights.sort(key=cv2.contourArea, reverse=True)
        flights = flights[:80]
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 120)
        detected = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 360,
            threshold=20,
            minLineLength=max(12, radius * 0.08),
            maxLineGap=max(3, radius * 0.08),
        )
        segments = (
            np.empty((0, 2, 2), float)
            if detected is None
            else detected.reshape(-1, 2, 2)
        )
        midpoints = np.round(segments.mean(axis=1)).astype(int)
        if len(segments):
            inside = (
                search[
                    np.clip(midpoints[:, 1], 0, work.shape[0] - 1),
                    np.clip(midpoints[:, 0], 0, work.shape[1] - 1),
                ]
                > 0
            )
            segments = segments[inside]
        diagnostics = ClassicalDartDiagnostics(
            work, scale_xy, search, mask, flights, segments
        )
        self.diagnostics = diagnostics
        for flight in flights:
            explanations = self._shaft_hypotheses(gray, flight, segments)
            if explanations:
                diagnostics.hypotheses.append(
                    max(explanations, key=lambda item: item.score)
                )
        diagnostics.hypotheses.sort(key=lambda item: item.score, reverse=True)
        if (
            not diagnostics.hypotheses
            or diagnostics.hypotheses[0].score < self.min_score
        ):
            diagnostics.message = "no flight with sufficient paired-shaft evidence"
            return ()
        best = diagnostics.hypotheses[0]
        if (
            len(diagnostics.hypotheses) > 1
            and best.score - diagnostics.hypotheses[1].score < self.ambiguity_margin
        ):
            diagnostics.message = "ambiguous competing flight/shaft hypotheses"
            return ()
        bounds = np.clip(
            best.bounds / scale_xy, [0, 0], [width - 1, height - 1]
        ).reshape(-1)
        box = DartBox(
            1,
            tuple(float(value) for value in bounds),
            best.score,
            "classical: colour flight + paired shaft edges",
        )
        validate_boxes(image, [box])
        diagnostics.accepted = True
        diagnostics.message = (
            "accepted strongest flight/shaft hypothesis; score is heuristic"
        )
        return (box,)

    def diagnostic_views(self) -> dict[str, np.ndarray]:
        """Working-resolution BGR views; green accepted, orange rejected candidates."""
        diagnostics = self.diagnostics
        if diagnostics is None:
            return {}
        segmentation = diagnostics.image.copy()
        segmentation[diagnostics.flight_mask > 0] = (255, 80, 200)
        search = diagnostics.image.copy()
        search[diagnostics.search_mask == 0] //= 4
        lines = diagnostics.image.copy()
        for segment in diagnostics.segments:
            start, end = np.round(segment).astype(int)
            cv2.line(lines, tuple(start), tuple(end), (255, 200, 0), 1, cv2.LINE_AA)
        candidates = diagnostics.image.copy()
        cv2.drawContours(candidates, diagnostics.flights, -1, (255, 200, 0), 1)
        for index, hypothesis in enumerate(diagnostics.hypotheses):
            colour = (
                (0, 220, 0) if index == 0 and diagnostics.accepted else (0, 140, 255)
            )
            start, end = np.round(hypothesis.bounds).astype(int)
            cv2.rectangle(candidates, tuple(start), tuple(end), colour, 2)
            for edge in hypothesis.shaft_edges:
                first, last = np.round(edge).astype(int)
                cv2.line(candidates, tuple(first), tuple(last), colour, 2)
            cv2.putText(
                candidates,
                f"{index + 1}: {hypothesis.score:.2f}",
                tuple(start),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )
        return {
            "search": search,
            "flight_mask": segmentation,
            "shaft_segments": lines,
            "hypotheses": candidates,
        }
