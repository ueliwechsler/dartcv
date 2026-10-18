"""Visual explanations of the measured and projected calibration geometry."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import Detection
from .geometry import CALIB_ANGLES, CALIB_NAMES, RADII, contour_radius, polar, project
from .overlay import draw_overlay, topdown_size_px

# BGR: one consistent colour per calibration boundary in every stage.
CALIB_COLOURS = ((0, 220, 255), (255, 200, 0), (255, 100, 255), (80, 255, 80))


def _draw_scale(img):
    return max(1, round(max(img.shape[:2]) / 1600))


def _text(img, text, origin, colour=(255, 255, 255)):
    scale = _draw_scale(img)
    cv2.putText(
        img,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale,
        (0, 0, 0),
        4 * scale,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale,
        colour,
        scale,
        cv2.LINE_AA,
    )


def _caption(img, *lines):
    scale = _draw_scale(img)
    for i, line in enumerate(lines):
        _text(img, line, (12 * scale, (26 + 26 * i) * scale))
    return img


def calibration_overlay(img, H, *, initial_H=None, supplied=False, ring_contours=None):
    """Highlight the four wires and outer-double intersections, not measured edges.

    In manual mode those intersections coincide with the supplied image points.
    In automatic mode they are predictions of the estimated homography.
    """
    out = img.copy()
    height, width = out.shape[:2]
    scale = _draw_scale(out)
    for angle, name, colour in zip(CALIB_ANGLES, CALIB_NAMES, CALIB_COLOURS):
        radius = RADII["double_outer"]
        if ring_contours and "double_outer" in ring_contours:
            radius = contour_radius(H, ring_contours["double_outer"], angle)
        wire = [polar(RADII["outer_bull"], angle), polar(radius, angle)]
        start, end = np.round(project(H, wire)).astype(int)
        if initial_H is not None:
            previous = tuple(
                np.round(
                    project(initial_H, [polar(RADII["double_outer"], angle)])[0]
                ).astype(int)
            )
            cv2.circle(
                out, previous, 9 * scale, (160, 160, 160), 2 * scale, cv2.LINE_AA
            )
            cv2.line(out, previous, tuple(end), (160, 160, 160), scale, cv2.LINE_AA)
        cv2.line(out, tuple(start), tuple(end), colour, 3 * scale, cv2.LINE_AA)
        cv2.drawMarker(
            out,
            tuple(end),
            colour,
            cv2.MARKER_CROSS,
            22 * scale,
            3 * scale,
            cv2.LINE_AA,
        )
        label = name.replace("|", "/")
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, scale
        )
        origin = (
            max(2, min(int(end[0]) + 12 * scale, width - tw - 4)),
            max(th + 4, min(int(end[1]) - 12 * scale, height - 4)),
        )
        _text(out, label, origin, colour)
    kind = (
        "Supplied calibration points"
        if supplied
        else "Projected calibration boundaries (not independently detected)"
    )
    lines = [kind]
    if initial_H is not None:
        lines.append(
            "Grey circles: initial estimate; crosses: final coloured outer edge"
        )
    return _caption(out, *lines)


def bull_alignment_overlay(img: np.ndarray, detection: Detection) -> np.ndarray:
    """Side-by-side, source-resolution bull crops with independently measured truth.

    The left panel is explicitly the ellipse-only seed, not the previous
    version's ECC result. Residuals measure agreement with colour detection,
    not independently annotated ground truth.
    """
    bull, debug = detection.bull, detection.diagnostics
    if bull is None or debug is None or debug.H_before_bull is None:
        return _caption(
            img.copy(),
            "6. No reliable red/green bull found; no bull constraint applied",
        )
    centre = bull.centre
    outer = project(
        detection.H, [polar(RADII["outer_bull"], a) for a in range(0, 360, 30)]
    )
    reach = max(30.0, float(np.linalg.norm(outer - centre, axis=1).max()) * 2.8)
    # Include the old predicted centre even for strongly tilted boards.
    old_centre = project(debug.H_before_bull, [[0, 0]])[0]
    reach = max(reach, float(np.linalg.norm(old_centre - centre)) * 1.3)
    size = 720
    scale = size / (2 * reach)
    crop = np.array(
        [
            [scale, 0, size / 2 - scale * centre[0]],
            [0, scale, size / 2 - scale * centre[1]],
            [0, 0, 1],
        ]
    )
    source = cv2.warpPerspective(img, crop, (size, size))
    contour = np.round(project(crop, bull.contour)).astype(np.int32)
    panels = []
    for label, H, colour in (
        (
            "Ellipse-only seed (before bull correction)",
            debug.H_before_bull,
            (0, 180, 255),
        ),
        ("Final bull-constrained homography", detection.H, (0, 255, 0)),
    ):
        panel = source.copy()
        for name in ("bull", "outer_bull"):
            points = project(crop @ H, [polar(RADII[name], a) for a in range(361)])
            cv2.polylines(
                panel, [np.round(points).astype(np.int32)], True, colour, 2, cv2.LINE_AA
            )
        cv2.polylines(panel, [contour], True, (255, 255, 0), 1, cv2.LINE_AA)
        predicted = project(H, [[0, 0]])[0]
        point = tuple(np.round(project(crop, [predicted])[0]).astype(int))
        cv2.drawMarker(panel, point, colour, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        residual = np.linalg.norm(predicted - centre)
        panel = cv2.copyMakeBorder(panel, 85, 0, 0, 0, cv2.BORDER_CONSTANT)
        _caption(
            panel,
            label,
            f"Centre residual to measured bull: {residual:.2f} source px",
            "Cyan: measured red boundary; coloured rings: model",
        )
        panels.append(panel)
    return np.hstack(panels)


def ring_alignment_overlay(img: np.ndarray, detection: Detection) -> np.ndarray:
    """Show before/after colour-boundary close-ups without using grey wire edges."""
    debug = detection.diagnostics
    if not detection.rings or debug is None or debug.H_before_rings is None:
        return _caption(
            img.copy(), "7. Insufficient colour-band coverage; nominal rings retained"
        )
    rows = []
    for band in ("double", "treble"):
        names = (f"{band}_inner", f"{band}_outer")
        if not all(name in detection.rings for name in names):
            continue
        angle = 18.0  # middle of a coloured sector, away from radial metal wires
        radii = [
            contour_radius(detection.H, detection.rings[name].contour, angle)
            for name in names
        ]
        edges = project(detection.H, [polar(r, angle) for r in radii])
        centre = edges.mean(axis=0)
        reach = max(25.0, float(np.linalg.norm(edges[1] - edges[0])) * 2.5)
        size = 720
        scale = size / (2 * reach)
        crop = np.array(
            [
                [scale, 0, size / 2 - scale * centre[0]],
                [0, scale, size / 2 - scale * centre[1]],
                [0, 0, 1],
            ]
        )
        source = cv2.warpPerspective(img, crop, (size, size))
        panels = []
        for measured in (False, True):
            panel = source.copy()
            for name in names:
                ring = detection.rings[name]
                if measured:
                    contour = ring.contour
                else:
                    contour = project(
                        debug.H_before_rings,
                        [polar(RADII[name], a) for a in range(361)],
                    )
                points = np.round(project(crop, contour)).astype(np.int32)
                colour = (0, 255, 0) if measured else (0, 180, 255)
                cv2.polylines(panel, [points], True, colour, 2, cv2.LINE_AA)
            panel = cv2.copyMakeBorder(panel, 85, 0, 0, 0, cv2.BORDER_CONSTANT)
            if measured:
                errors = [detection.rings[name].median_error_px for name in names]
                lines = (
                    f"{band.title()}: fitted coloured edges",
                    "Grey metal excluded; bull remains anchored",
                    f"Median fit residuals: {errors[0]:.2f} / {errors[1]:.2f} source px",
                )
            else:
                lines = (
                    f"{band.title()}: previous nominal rings",
                    "Before colour-band refinement",
                    "Same source crop in both panels",
                )
            _caption(panel, *lines)
            panels.append(panel)
        rows.append(np.hstack(panels))
    return np.vstack(rows)


def detection_overlays(img: np.ndarray, detection: Detection) -> dict[str, np.ndarray]:
    """Render captured pipeline stages without altering the source image."""
    debug = detection.diagnostics
    if debug is None:
        raise ValueError("detect_board(..., diagnostics=True) is required")

    mask = (
        cv2.resize(
            debug.colour_mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    segmented = (img.astype(np.float32) * 0.3).astype(np.uint8)
    segmented[mask] = cv2.addWeighted(
        img, 0.5, np.full_like(img, (0, 255, 255)), 0.5, 0
    )[mask]
    _caption(segmented, "1. Red/green segmentation (after opening + dilation)")

    ellipse = img.copy()
    scale = _draw_scale(img)
    for point, keep in zip(debug.boundary_points, debug.ellipse_inliers):
        cv2.circle(
            ellipse,
            tuple(np.round(point).astype(int)),
            3 * scale,
            (0, 255, 0) if keep else (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
    cv2.ellipse(ellipse, detection.ellipse, (0, 255, 255), 2 * scale, cv2.LINE_AA)
    _caption(
        ellipse,
        "2. Outer double-ring ellipse",
        "Green: retained boundary samples; red: rejected; yellow: fit",
    )

    initial = draw_overlay(img, detection.H_initial, color=(0, 220, 255))
    _caption(
        initial,
        "3. Ellipse + bull constraint + rotation estimate (before ECC)",
        f"Rotation: {detection.rotation_deg:.1f} deg; colour pattern repeats every 36 deg",
    )

    ring_contours = {name: ring.contour for name, ring in detection.rings.items()}
    calibration = calibration_overlay(
        img, detection.H, initial_H=detection.H_initial, ring_contours=ring_contours
    )
    result = {
        "01_colour_mask.png": segmented,
        "02_outer_ellipse.png": ellipse,
        "03_initial_alignment.png": initial,
    }
    if debug.ecc_edges is not None and debug.ecc_template is not None:
        # The actual gradient maps passed to the last attempted ECC iteration.
        edges = np.round(np.clip(debug.ecc_edges, 0, 1) * 255).astype(np.uint8)
        edge_view = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        template = np.clip(debug.ecc_template, 0, 1)[..., None]
        edge_view = np.round(
            edge_view * (1 - template) + np.array([255, 255, 0]) * template
        ).astype(np.uint8)
        if debug.ecc_mask is not None:
            edge_view[debug.ecc_mask == 0] = 0
        size = topdown_size_px()
        edge_view = cv2.resize(edge_view, (size, size), interpolation=cv2.INTER_NEAREST)
        score = detection.ecc_score
        outcome = (
            f"Accepted ECC score: {score:.3f} (before final bull constraint)"
            if score is not None
            else "ECC failed/rejected: initial homography retained"
        )
        _caption(
            edge_view,
            "4. Board-masked ECC input: gradients + cyan wire template",
            outcome,
            "Initial rectification; last attempted scale, before ECC alignment",
        )
        result["04_ecc_edges.png"] = edge_view
    result["05_calibration.png"] = calibration
    result["06_bull_alignment.png"] = bull_alignment_overlay(img, detection)
    result["07_ring_alignment.png"] = ring_alignment_overlay(img, detection)
    return result
