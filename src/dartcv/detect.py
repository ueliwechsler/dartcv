"""Automatic dartboard detection: image -> homography (model mm -> image px).

Pipeline
1. Segment saturated red/green pixels (double/treble rings, bull).
2. Find the outer edge of the double ring along rays from the blob centre and
   fit an ellipse (iteratively rejecting outliers).
3. Resolve the board rotation from the alternating red/green double-ring
   pattern. That pattern repeats every 36 deg, so the candidate with the 20
   closest to the image "up" direction is chosen (override with `upright_deg`).
4. Refine the homography with ECC alignment of a rendered wire template to the
   rectified image gradients, which also corrects perspective.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import RADII, polar, project
from .overlay import TOPDOWN_SCALE, model_to_topdown, topdown_size_px, warp_topdown

WORK_SIZE = 1000  # longest image side used for detection


class DetectionError(RuntimeError):
    pass


@dataclass
class Detection:
    H: np.ndarray  # model mm -> original image px
    H_initial: np.ndarray  # before ECC refinement
    ellipse: tuple  # ((cx, cy), (w, h), angle) in original px
    rotation_deg: float  # clockwise angle of the 20 from image up (unwarped)
    ecc_score: float | None  # ECC correlation (None if refinement failed)


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
    return cv2.fitEllipse(pts[keep])


def _ellipse_to_H(ell) -> np.ndarray:
    """Model(mm, 20 up) -> image, circle mapped to the ellipse, no rotation."""
    (cx, cy), (w, h), a = ell
    t = np.deg2rad(a)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A = R @ np.diag([w / 2 / RADII["double_outer"], h / 2 / RADII["double_outer"]]) @ R.T
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
        sx = cv2.remap(f, p[:, 0].reshape(1, -1).astype(np.float32),
                       p[:, 1].reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR)[0]
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
    for r in ("bull", "outer_bull", "treble_inner", "treble_outer", "double_inner", "double_outer"):
        poly = project(T, [polar(RADII[r], a) for a in angles])
        cv2.polylines(img, [np.round(poly).astype(np.int32)], True, 255, 2, cv2.LINE_AA)
    for k in range(20):
        a = 9 + 18 * k
        p = project(T, [polar(RADII["outer_bull"], a), polar(RADII["double_outer"], a)])
        cv2.line(img, tuple(int(round(v)) for v in p[0]), tuple(int(round(v)) for v in p[1]),
                 255, 2, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def _edge_map(gray: np.ndarray, k: float) -> np.ndarray:
    f = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(f, (0, 0), 1.5)
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return e / (e.max() + 1e-6)


def _refine_ecc(img: np.ndarray, H0: np.ndarray):
    """Coarse-to-fine ECC of a rendered wire template to the rectified image.

    Returns (H, score); (H0, None) if ECC fails.
    """
    T = model_to_topdown()
    top = warp_topdown(img, H0)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tmpl_full = _template() * 255
    W = np.eye(3)
    score = None
    for k in (0.125, 0.25, 0.5):
        S = np.diag([k, k, 1.0])
        try:
            score, Wk = cv2.findTransformECC(
                _edge_map(tmpl_full, k), _edge_map(gray, k), (S @ W @ np.linalg.inv(S)).astype(np.float32),
                cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                None, 5)
        except cv2.error:
            return H0, None
        W = np.linalg.inv(S) @ Wk.astype(np.float64) @ S
    H = H0 @ np.linalg.inv(T) @ W @ T
    return H / H[2, 2], float(score)


def detect_board(img: np.ndarray, upright_deg: float = 0.0, refine: bool = True) -> Detection:
    """Detect the dartboard in a BGR image."""
    s = WORK_SIZE / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    s = small.shape[1] / img.shape[1]

    ell = _fit_outer_ellipse(_colour_mask(small))
    (_, (w, h), _) = ell
    if min(w, h) < 0.15 * max(small.shape[:2]) or max(w, h) / min(w, h) > 3:
        raise DetectionError(f"implausible ellipse {ell}")

    H_base = _ellipse_to_H(ell)
    delta = _estimate_rotation(small, H_base, upright_deg)
    H0 = H_base @ _rot(delta)
    H0 /= H0[2, 2]

    H, score = (H0, None)
    if refine:
        H, score = _refine_ecc(small, H0)
        if score is not None and score < 0.2:  # poor alignment: keep the ellipse estimate
            H, score = H0, None

    S = np.diag([1 / s, 1 / s, 1.0])
    (cx, cy), (ew, eh), ea = ell
    return Detection(H=S @ H, H_initial=S @ H0, ellipse=((cx / s, cy / s), (ew / s, eh / s), ea),
                     rotation_deg=delta, ecc_score=score)
