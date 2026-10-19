"""Automatic dartboard detection: image -> homography (model mm -> image px).

Pipeline
1. Segment saturated red/green pixels (double/treble rings, bull).
2. Find the outer edge of the double ring along rays from the blob centre and
   fit an ellipse (iteratively rejecting outliers).
3. Independently locate the red bull inside its green surround and correct
    interior perspective while preserving the outer double-ring conic.
4. Resolve the board rotation from the alternating red/green double-ring
   pattern. That pattern repeats every 36 deg, so the candidate with the 20
   closest to the image "up" direction is chosen (override with `upright_deg`).
5. Refine against board-only image gradients with ECC, then reapply the bull
    constraint so global correlation cannot displace the measured centre.
6. Measure undilated red/green double and treble bands at source resolution,
   refine their shared geometry without moving the bull, and fit paint edges
   separately from the nominal wire dimensions.

See docs/cv-pipeline.md for contracts, acceptance rules, and fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import RADII, polar, project
from .overlay import model_to_topdown, topdown_size_px, warp_topdown
from .rings import RingBoundary, refine_coloured_rings

WORK_SIZE = 1000  # longest image side used for detection


class DetectionError(RuntimeError):
    pass


@dataclass
class BullDetection:
    centre: np.ndarray  # independently measured, original image pixels
    contour: np.ndarray  # undilated inner-bull boundary, original pixels


@dataclass
class DetectionDiagnostics:
    colour_mask: np.ndarray  # detection-resolution segmentation
    boundary_points: np.ndarray  # original image pixels
    ellipse_inliers: np.ndarray  # boolean mask over boundary_points
    ecc_edges: np.ndarray | None = None  # last attempted ECC scale
    ecc_template: np.ndarray | None = None
    ecc_mask: np.ndarray | None = None
    H_before_bull: np.ndarray | None = None  # ellipse-only diagnostic baseline
    H_before_rings: np.ndarray | None = None


@dataclass
class Detection:
    H: np.ndarray  # model mm -> original image px
    H_initial: np.ndarray  # before ECC refinement
    ellipse: tuple  # ((cx, cy), (w, h), angle) in original px
    rotation_deg: float  # clockwise angle of the 20 from image up (unwarped)
    ecc_score: float | None  # ECC correlation (None if refinement failed)
    diagnostics: DetectionDiagnostics | None = None
    bull: BullDetection | None = None
    rings: dict[str, RingBoundary] = field(default_factory=dict)


def _detect_bull(img: np.ndarray, H: np.ndarray) -> BullDetection | None:
    """Find a compact red inner bull with green support in a full-resolution ROI.

    H only defines the search region and approximate scale, not the measurement.
    No candidate means no constraint; never invent a centre from the prediction.
    """
    centre = project(H, [[0, 0]])[0]
    axes = project(H, [[RADII["bull"], 0], [0, RADII["bull"]]]) - centre
    radius = float(np.sqrt(abs(np.linalg.det(axes))))
    if radius < 2:
        return None
    # Perspective can move the projected board centre substantially away from
    # the outer ellipse centre; include its green surround, not just the red disk.
    reach = int(np.ceil(radius * 15))
    x0, y0 = np.maximum(np.floor(centre - reach).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(centre + reach).astype(int), img.shape[1::-1])
    roi = img[y0:y1, x0:x1]
    if not roi.size:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 12) | (h > 168)) & (s > 90) & (v > 45)).astype(np.uint8) * 255
    green = (h > 35) & (h < 95) & (s > 65) & (v > 40)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    best_score = -np.inf
    yy, xx = np.mgrid[: roi.shape[0], : roi.shape[1]]
    for contour in contours:
        area = cv2.contourArea(contour)
        if len(contour) < 5 or not 0.25 < area / (np.pi * radius**2) < 2.5:
            continue
        (cx, cy), (w, height), angle = cv2.fitEllipse(contour)
        if min(w, height) < radius * 0.5 or max(w, height) / min(w, height) > 3:
            continue
        if area / (np.pi * w * height / 4) < 0.65:
            continue
        t = np.deg2rad(angle)
        dx, dy = xx - cx, yy - cy
        u = (dx * np.cos(t) + dy * np.sin(t)) / (w / 2)
        z = (-dx * np.sin(t) + dy * np.cos(t)) / (height / 2)
        rr = np.hypot(u, z)
        annulus = (rr > 1.35) & (rr < 2.2)
        support = float(green[annulus].mean()) if annulus.any() else 0.0
        if support < 0.35:
            continue
        point = np.array([cx + x0, cy + y0])
        distance = np.linalg.norm(point - centre) / radius
        model_point = project(np.linalg.inv(H), [point])[0]
        if np.linalg.norm(model_point) > 0.4 * RADII["double_outer"]:
            continue
        quality = (
            support - 0.04 * distance - 0.1 * abs(np.log(area / (np.pi * radius**2)))
        )
        if quality > best_score:
            best_score = quality
            best = BullDetection(point, contour.reshape(-1, 2) + [x0, y0])
    return best


def _anchor_bull(H: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Move the model origin to a measured bull without translating the board.

    A projective automorphism of the unit disk preserves its boundary circle.
    Conjugating it by H therefore preserves the fitted outer-double conic while
    correcting the interior perspective. This is not an overlay-only offset.
    """
    radius = RADII["double_outer"]
    v = project(np.linalg.inv(H), [centre])[0].astype(np.float64) / radius
    vv = float(v @ v)
    if vv >= 0.45**2:
        raise ValueError("bull too far from the predicted board centre")
    if vv < 1e-16:
        return H.copy()
    gamma = 1 / np.sqrt(1 - vv)
    boost = np.eye(3)
    boost[:2, :2] += (gamma - 1) * np.outer(v, v) / vv
    boost[:2, 2] = gamma * v
    boost[2, :2] = gamma * v
    boost[2, 2] = gamma
    scale = np.diag([radius, radius, 1.0])
    anchored = H @ scale @ boost @ np.linalg.inv(scale)
    return anchored / anchored[2, 2]


def _colour_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 10) | (h > 170)) & (s > 110) & (v > 60)
    green = (h > 40) & (h < 90) & (s > 90) & (v > 50)
    mask = (red | green).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))


def _fit_outer_ellipse(mask: np.ndarray):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise DetectionError("no red/green regions found")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = labels == k
    centre = cents[k]
    ys, xs = np.nonzero(comp)
    d = np.hypot(xs - centre[0], ys - centre[1])
    ang = np.arctan2(ys - centre[1], xs - centre[0])
    bins = ((ang + np.pi) / (2 * np.pi) * 180).astype(int) % 180
    pts = []
    for b in range(180):  # farthest pixel per 2-degree wedge
        sel = np.nonzero(bins == b)[0]
        if sel.size:
            i = sel[np.argmax(d[sel])]
            pts.append((xs[i], ys[i]))
    pts = np.float32(pts)
    if len(pts) < 40:
        raise DetectionError("too few boundary points for an ellipse")
    keep = np.ones(len(pts), bool)
    for _ in range(6):
        ell = cv2.fitEllipse(pts[keep])
        (cx, cy), (w, h), a = ell
        t = np.deg2rad(a)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        res = np.abs(np.sqrt((u / (w / 2)) ** 2 + (v / (h / 2)) ** 2) - 1) * (w + h) / 4
        thr = max(2.0, 2.5 * np.median(res[keep]) + 1)
        new = res < thr
        if new.sum() < 30 or (new == keep).all():
            break
        keep = new
    return cv2.fitEllipse(pts[keep]), pts, keep


def _ellipse_to_H(ell) -> np.ndarray:
    """Model(mm, 20 up) -> image, circle mapped to the ellipse, no rotation."""
    (cx, cy), (w, h), a = ell
    t = np.deg2rad(a)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A = (
        R
        @ np.diag([w / 2 / RADII["double_outer"], h / 2 / RADII["double_outer"]])
        @ R.T
    )
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = (cx, cy)
    return H


def _rot(deg: float) -> np.ndarray:
    """Clockwise rotation (image coords, y down) as a 3x3 matrix."""
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _estimate_rotation(img: np.ndarray, H_base: np.ndarray, upright_deg: float):
    """Angle delta (deg, cw) such that the 20 sits at delta in H_base's frame."""
    n = 720
    thetas = np.arange(n) * 360.0 / n
    signal = np.zeros(n)
    f = img.astype(np.float32)
    for r in (164.0, 166.0, 168.0):
        p = project(H_base, [polar(r, t) for t in thetas])
        sx = cv2.remap(
            f,
            p[:, 0].reshape(1, -1).astype(np.float32),
            p[:, 1].reshape(1, -1).astype(np.float32),
            cv2.INTER_LINEAR,
        )[0]
        signal += sx[:, 2] - sx[:, 1]  # R - G: red positive, green negative
    signal -= signal.mean()

    def model(delta):
        ang = (thetas - delta + 9.0) % 360.0
        return np.where((ang // 18).astype(int) % 2 == 0, 1.0, -1.0)

    deltas = np.arange(0, 36, 0.5)
    scores = [float(signal @ model(d)) for d in deltas]
    best = float(deltas[int(np.argmax(scores))])
    # pick the 36-deg-periodic candidate closest to the requested "up"
    cands = best + 36.0 * np.arange(-5, 5)
    wrapped = (cands - upright_deg + 180) % 360 - 180
    return float(cands[int(np.argmin(np.abs(wrapped)))] % 360)


def _template() -> np.ndarray:
    size = topdown_size_px()
    T = model_to_topdown()
    img = np.zeros((size, size), np.uint8)
    angles = np.linspace(0, 360, 721)
    for r in (
        "bull",
        "outer_bull",
        "treble_inner",
        "treble_outer",
        "double_inner",
        "double_outer",
    ):
        poly = project(T, [polar(RADII[r], a) for a in angles])
        cv2.polylines(img, [np.round(poly).astype(np.int32)], True, 255, 2, cv2.LINE_AA)
    for k in range(20):
        a = 9 + 18 * k
        p = project(T, [polar(RADII["outer_bull"], a), polar(RADII["double_outer"], a)])
        cv2.line(
            img,
            tuple(int(round(v)) for v in p[0]),
            tuple(int(round(v)) for v in p[1]),
            255,
            2,
            cv2.LINE_AA,
        )
    return img.astype(np.float32) / 255.0


def _edge_map(gray: np.ndarray, k: float) -> np.ndarray:
    f = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(f, (0, 0), 1.5)
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return e / (e.max() + 1e-6)


def _refine_ecc(
    img: np.ndarray, H0: np.ndarray, diagnostics: DetectionDiagnostics | None = None
):
    """Coarse-to-fine ECC of a rendered wire template to the rectified image.

    Returns (H, score); (H0, None) if ECC fails.
    """
    T = model_to_topdown()
    top = warp_topdown(img, H0)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tmpl_full = _template() * 255
    # Keep lettering, branding and background gradients out of ECC's objective.
    size = top.shape[0]
    yy, xx = np.mgrid[:size, :size]
    board_mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (
        RADII["double_outer"] * model_to_topdown()[0, 0]
    ) ** 2
    W = np.eye(3)
    score = None
    for k in (0.125, 0.25, 0.5):
        S = np.diag([k, k, 1.0])
        template_edges = _edge_map(tmpl_full, k)
        image_edges = _edge_map(gray, k)
        input_mask = cv2.resize(
            board_mask.astype(np.uint8) * 255,
            image_edges.shape[1::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        if diagnostics is not None:
            diagnostics.ecc_edges = image_edges
            diagnostics.ecc_template = template_edges
            diagnostics.ecc_mask = input_mask
        try:
            score, Wk = cv2.findTransformECC(
                template_edges,
                image_edges,
                (S @ W @ np.linalg.inv(S)).astype(np.float32),
                cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                input_mask,
                5,
            )
        except cv2.error:
            return H0, None
        W = np.linalg.inv(S) @ Wk.astype(np.float64) @ S
    H = H0 @ np.linalg.inv(T) @ W @ T
    return H / H[2, 2], float(score)


def detect_board(
    img: np.ndarray,
    upright_deg: float = 0.0,
    refine: bool = True,
    diagnostics: bool = False,
) -> Detection:
    """Detect the board, optionally retaining intermediate diagnostic data."""
    s = WORK_SIZE / max(img.shape[:2])
    small = (
        cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if s < 1
        else img
    )
    s = small.shape[1] / img.shape[1]

    mask = _colour_mask(small)
    ell, boundary, inliers = _fit_outer_ellipse(mask)
    debug = DetectionDiagnostics(mask, boundary / s, inliers) if diagnostics else None
    (_, (w, h), _) = ell
    if min(w, h) < 0.15 * max(small.shape[:2]) or max(w, h) / min(w, h) > 3:
        raise DetectionError(f"implausible ellipse {ell}")

    H_base = _ellipse_to_H(ell)
    S = np.diag([1 / s, 1 / s, 1.0])
    bull = _detect_bull(img, S @ H_base)
    if bull is not None:
        H_base = _anchor_bull(H_base, bull.centre * s)
    delta = _estimate_rotation(small, H_base, upright_deg)
    H0 = H_base @ _rot(delta)
    H0 /= H0[2, 2]
    if debug is not None:
        debug.H_before_bull = S @ _ellipse_to_H(ell) @ _rot(delta)

    H, score = (H0, None)
    if refine:
        H, score = _refine_ecc(small, H0, debug)
        if (
            score is not None and score < 0.2
        ):  # poor alignment: keep the ellipse estimate
            H, score = H0, None
        if bull is not None:
            try:
                H = _anchor_bull(H, bull.centre * s)
            except ValueError:  # ECC wandered away from the measured board geometry
                H, score = H0, None

    H_original = S @ H
    if debug is not None:
        debug.H_before_rings = H_original.copy()
    rings = {}
    if refine:
        H_original, rings = refine_coloured_rings(img, H_original)

    (cx, cy), (ew, eh), ea = ell
    return Detection(
        H=H_original,
        H_initial=S @ H0,
        ellipse=((cx / s, cy / s), (ew / s, eh / s), ea),
        rotation_deg=delta,
        ecc_score=score,
        diagnostics=debug,
        bull=bull,
        rings=rings,
    )
