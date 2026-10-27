"""Measure the coloured scoring bands, not the surrounding metal wires.

The standard radii constrain the common projective geometry using band centres.
Each coloured inner/outer edge is then fitted independently: actual paint widths
need not coincide exactly with the nominal wire-based model dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import RADII, contour_radius, project

BANDS = (("treble_inner", "treble_outer"), ("double_inner", "double_outer"))


@dataclass
class RingBoundary:
    samples: np.ndarray  # measured colour transitions in original image pixels
    contour: np.ndarray  # robustly fitted coloured edge in original pixels
    inliers: np.ndarray
    median_error_px: float  # radial fit residual, not annotated accuracy


def _sample_bands(img: np.ndarray, H: np.ndarray) -> dict[str, np.ndarray]:
    """Sample full-resolution BGR strips without dilation or grey-edge gradients.

    Skip radial sector wires and require a complete, plausible-width red/green
    run bracketed by non-colour on both sides. Missing/occluded spokes are omitted.
    """
    angles = np.arange(0.0, 360.0)
    angles = angles[np.abs((angles + 9) % 18 - 9) <= 6]
    theta = np.deg2rad(angles)
    direction = np.column_stack([np.sin(theta), -np.cos(theta)])
    result = {}
    for inner, outer in BANDS:
        middle = (RADII[inner] + RADII[outer]) / 2
        radii = np.arange(middle - 20, middle + 20.01, 0.2)
        model = direction[:, None, :] * radii[None, :, None]
        pixels = project(H, model.reshape(-1, 2)).reshape(model.shape)
        strips = cv2.remap(
            img,
            pixels[..., 0],
            pixels[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        hsv = cv2.cvtColor(strips, cv2.COLOR_BGR2HSV)
        hue, sat, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        colour = ((hue < 12) | (hue > 168) | ((hue > 35) & (hue < 95))) & (value > 45)
        # Soft saturation crossings locate the coloured material without growing
        # into grey metal. Bilinear sampling provides subpixel edge positions.
        strength = np.clip((sat.astype(float) - 65) / 35, 0, 1) * colour
        edges = {inner: [], outer: []}
        for i, signal in enumerate(strength):
            active = signal >= 0.5
            changes = np.diff(np.r_[False, active, False].astype(int))
            starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            candidates = []
            for first, stop in zip(starts, stops):
                if first == 0 or stop == len(radii):
                    continue
                lo = np.interp(
                    0.5, signal[first - 1 : first + 1], radii[first - 1 : first + 1]
                )
                hi = np.interp(
                    0.5,
                    signal[stop - 1 : stop + 1][::-1],
                    radii[stop - 1 : stop + 1][::-1],
                )
                width = hi - lo
                if 2.0 < width < 12.0:
                    quality = abs((lo + hi) / 2 - middle) + 0.15 * abs(width - 8)
                    candidates.append((quality, lo, hi))
            if not candidates:
                continue
            _, lo, hi = min(candidates)
            for name, radius in ((inner, lo), (outer, hi)):
                edges[name].append(direction[i] * radius)
        # Distributed support prevents a small visible arc from defining a ring.
        if len(edges[inner]) < 60:
            continue
        points = np.asarray(edges[inner])
        quadrants = (
            np.degrees(np.arctan2(points[:, 0], -points[:, 1])) % 360 // 90
        ).astype(int)
        if len(np.unique(quadrants)) < 4:
            continue
        for name, points in edges.items():
            result[name] = project(H, points)
    return result


def _fit_boundary(
    points: np.ndarray, H: np.ndarray, nominal: float
) -> RingBoundary | None:
    """Robust ellipse in rectified coordinates, with rejection of colour clutter."""
    model = project(np.linalg.inv(H), points)
    keep = np.ones(len(model), bool)
    for _ in range(6):
        ell = cv2.fitEllipse(model[keep].astype(np.float32))
        (cx, cy), (w, h), angle = ell
        if min(w, h) <= 0:
            return None
        a = np.deg2rad(angle)
        dx, dy = model[:, 0] - cx, model[:, 1] - cy
        u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
        residual = np.abs(np.hypot(u / (w / 2), v / (h / 2)) - 1) * (w + h) / 4
        new = residual < max(0.35, 3 * np.median(residual[keep]))
        if new.sum() < 50:
            return None
        if np.array_equal(new, keep):
            break
        keep = new
    ell = cv2.fitEllipse(model[keep].astype(np.float32))
    (cx, cy), (w, h), angle = ell
    if np.hypot(cx, cy) > 6 or not (
        0.85 * nominal < min(w, h) / 2 <= max(w, h) / 2 < 1.15 * nominal
    ):
        return None
    # Also check angular coverage AFTER rejecting outliers.
    bearings = np.arctan2(model[keep, 0], -model[keep, 1])
    if len(np.unique(((bearings + np.pi) / (np.pi / 2)).astype(int) % 4)) < 4:
        return None
    t = np.linspace(0, 2 * np.pi, 721)
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    contour_model = np.column_stack(
        [w / 2 * np.cos(t), h / 2 * np.sin(t)]
    ) @ rotation.T + [cx, cy]
    contour = project(H, contour_model)
    # Evaluate the continuous ellipse, not distance to discretized vertices:
    # vertex spacing alone could otherwise report several pixels of false error.
    centred = model - [cx, cy]
    uv = centred @ rotation
    rho = np.hypot(uv[:, 0] / (w / 2), uv[:, 1] / (h / 2))
    fitted = project(H, centred / rho[:, None] + [cx, cy])
    distances = np.linalg.norm(points - fitted, axis=1)
    return RingBoundary(points, contour, keep, float(np.median(distances[keep])))


def _solve_ring_perspective_irls(
    original_model: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Solve 5-parameter ring perspective correction via pure-NumPy Gauss-Newton / IRLS."""
    p = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    lower = np.array([0.85, -0.1, 0.85, -0.1, -0.1], dtype=np.float64)
    upper = np.array([1.15, 0.1, 1.15, 0.1, 0.1], dtype=np.float64)
    X = original_model[:, 0]
    Y = original_model[:, 1]
    n_pts = len(targets)

    scale = 0.5
    scale_sq = scale * scale
    inv_scale = 2.0

    def _eval(p_cur):
        det = p_cur[0] * p_cur[2] - p_cur[1] * p_cur[1]
        inv_det = 1.0 / det
        s00 = p_cur[2] * inv_det
        s01 = -p_cur[1] * inv_det
        s11 = p_cur[0] * inv_det
        xt = X * s00 + Y * s01
        yt = X * s01 + Y * s11
        w = 1.0 - (xt * (p_cur[3] * (1.0 / 170.0)) + yt * (p_cur[4] * (1.0 / 170.0)))
        inv_w = 1.0 / w
        u = xt * inv_w
        v = yt * inv_w
        r = np.hypot(u, v)
        return r - targets, u, v, r, s00, s01, s11, inv_w

    res, u, v, r, s00, s01, s11, inv_w = _eval(p)
    loss = np.sum(2.0 * scale_sq * (np.sqrt(1.0 + (res * inv_scale) ** 2) - 1.0))
    J = np.empty((n_pts, 5), dtype=np.float64)

    for _ in range(12):
        ur = u / r
        vr = v / r
        c3 = p[3] * (1.0 / 170.0)
        c4 = p[4] * (1.0 / 170.0)

        # Analytical Jacobian
        dxt0 = -X * (s00 * s00) - Y * (s00 * s01)
        dyt0 = -X * (s00 * s01) - Y * (s01 * s01)
        dw0 = -(c3 * dxt0 + c4 * dyt0)
        J[:, 0] = ur * (dxt0 - u * dw0) * inv_w + vr * (dyt0 - v * dw0) * inv_w

        dxt1 = -X * (2.0 * s00 * s01) - Y * (s00 * s11 + s01 * s01)
        dyt1 = -X * (s00 * s11 + s01 * s01) - Y * (2.0 * s01 * s11)
        dw1 = -(c3 * dxt1 + c4 * dyt1)
        J[:, 1] = ur * (dxt1 - u * dw1) * inv_w + vr * (dyt1 - v * dw1) * inv_w

        dxt2 = -X * (s01 * s01) - Y * (s01 * s11)
        dyt2 = -X * (s01 * s11) - Y * (s11 * s11)
        dw2 = -(c3 * dxt2 + c4 * dyt2)
        J[:, 2] = ur * (dxt2 - u * dw2) * inv_w + vr * (dyt2 - v * dw2) * inv_w

        J[:, 3] = (u * r) * (1.0 / 170.0)
        J[:, 4] = (v * r) * (1.0 / 170.0)

        w_huber = 1.0 / np.sqrt(1.0 + (res * inv_scale) ** 2)
        Jw = J.T * w_huber
        H_mat = Jw @ J + 1e-4 * np.eye(5, dtype=np.float64)
        g = Jw @ res

        try:
            dp = -np.linalg.solve(H_mat, g)
        except np.linalg.LinAlgError:
            break

        step_accepted = False
        step = 1.0
        for _ in range(4):
            p_cand = np.clip(p + step * dp, lower, upper)
            res_cand, u_cand, v_cand, r_cand, s00_cand, s01_cand, s11_cand, inv_w_cand = _eval(p_cand)
            loss_cand = np.sum(2.0 * scale_sq * (np.sqrt(1.0 + (res_cand * inv_scale) ** 2) - 1.0))
            if loss_cand < loss:
                p = p_cand
                res, u, v, r, s00, s01, s11, inv_w = res_cand, u_cand, v_cand, r_cand, s00_cand, s01_cand, s11_cand, inv_w_cand
                loss = loss_cand
                step_accepted = True
                break
            step *= 0.5

        if not step_accepted or np.max(np.abs(step * dp)) < 1e-5:
            break

    return p


def refine_coloured_rings(
    img: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, dict[str, RingBoundary]]:
    """Fit common geometry to colour-band centres, then measure all four edges.

    The five-parameter correction preserves H's model origin EXACTLY (the bull)
    and has no free rotation. Robust radial residuals reject glare/darts. A band
    without reliable coverage remains nominal rather than fabricating an edge.
    """
    measured = _sample_bands(img, H)
    refined = H.copy()
    if len(measured) == 4:
        in_dbl = project(np.linalg.inv(H), measured["double_inner"])
        out_dbl = project(np.linalg.inv(H), measured["double_outer"])
        w_dbl = float(
            np.median(
                np.hypot(out_dbl[:, 0], out_dbl[:, 1])
                - np.hypot(in_dbl[:, 0], in_dbl[:, 1])
            )
        )
        tgt_dbl_mid = RADII["double_outer"] - w_dbl / 2.0

        in_trb = project(np.linalg.inv(H), measured["treble_inner"])
        out_trb = project(np.linalg.inv(H), measured["treble_outer"])
        w_trb = float(
            np.median(
                np.hypot(out_trb[:, 0], out_trb[:, 1])
                - np.hypot(in_trb[:, 0], in_trb[:, 1])
            )
        )
        tgt_trb_mid = RADII["treble_outer"] - w_trb / 2.0

        midpoints, targets = [], []
        m_dbl = project(H, (in_dbl + out_dbl) / 2.0)
        midpoints.extend(m_dbl)
        targets.extend([tgt_dbl_mid] * len(m_dbl))

        m_trb = project(H, (in_trb + out_trb) / 2.0)
        midpoints.extend(m_trb)
        targets.extend([tgt_trb_mid] * len(m_trb))

        original_model = project(np.linalg.inv(H), midpoints).astype(np.float64)
        targets = np.asarray(targets)

        def correction(p):
            return np.array(
                [[p[0], p[1], 0], [p[1], p[2], 0], [p[3] / 170, p[4] / 170, 1]]
            )

        def residual(p):
            homogeneous = np.column_stack([original_model, np.ones(len(targets))])
            transformed = homogeneous @ np.linalg.inv(correction(p)).T
            return (
                np.linalg.norm(transformed[:, :2] / transformed[:, 2:], axis=1)
                - targets
            )

        initial = np.array([1.0, 0, 1.0, 0, 0])
        p_opt = _solve_ring_perspective_irls(original_model, targets)
        if np.median(np.abs(residual(p_opt))) <= np.median(
            np.abs(residual(initial))
        ):
            refined = H @ correction(p_opt)
            refined /= refined[2, 2]
            measured = _sample_bands(img, refined)

    boundaries = {}
    for inner, outer in BANDS:
        if inner not in measured or outer not in measured:
            continue
        fitted_inner = _fit_boundary(measured[inner], refined, RADII[inner])
        fitted_outer = _fit_boundary(measured[outer], refined, RADII[outer])
        if fitted_inner is None or fitted_outer is None:
            continue
        # Independent coloured edges must still form a non-crossing narrow band.
        ri = contour_radius(refined, fitted_inner.contour, np.arange(360))
        ro = contour_radius(refined, fitted_outer.contour, np.arange(360))
        if not np.all((ro - ri > 2) & (ro - ri < 12)):
            continue
        boundaries[inner], boundaries[outer] = fitted_inner, fitted_outer
    return refined, boundaries
