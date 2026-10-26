"""Single-photo dart scorer implemented as a project module.

This is the classical CV algorithm from ``examples/dart_scorer.py`` packaged for
reuse in the project. It performs board calibration, radial mapping, barrel/flight
segmentation and dart scoring from one image.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from .geometry import score_from_image_point, score_from_model_point

__all__ = [
    "BLACK",
    "BEIGE",
    "GREEN",
    "RED",
    "OUTLIER",
    "R_IB",
    "R_OB",
    "R_TI",
    "R_TO",
    "R_DI",
    "R_DO",
    "SECTORS",
    "DartScorer",
    "annotate",
    "barrels_and_flights",
    "board_class",
    "calibrate",
    "classify",
    "cone",
    "ellipse_dist",
    "ellipse_ransac",
    "extend_tip",
    "find_bull",
    "find_darts",
    "fit_models",
    "normalise_scale",
    "polar",
    "ring_points",
    "rot",
    "score_at",
    "score_image",
    "soft",
    "template",
    "wire_margin",
    "score_image_point",
    "score_model_point",
]

# ------------------------------------------------------------------ board geometry (mm)
R_IB, R_OB = 6.35, 15.9
R_TI, R_TO = 99.0, 107.0
R_DI, R_DO = 162.0, 170.0
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
BLACK, BEIGE, GREEN, RED, OUTLIER = 0, 1, 2, 3, 4
PROTOS = [
    (18, 0.34, 0.09),
    (33, 0.37, 0.94),
    (142, 0.96, 0.53),
    (1, 0.83, 0.91),
]  # H deg, S, V
ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


class DartScorer:
    """Convenience wrapper that exposes the classical score pipeline to callers."""

    @staticmethod
    def score_model_point(x_mm: float, y_mm: float) -> int:
        return score_from_model_point(float(x_mm), float(y_mm))

    @staticmethod
    def score_image_point(H: np.ndarray, xy: tuple[float, float] | np.ndarray) -> int:
        return score_from_image_point(H, xy)

    @staticmethod
    def score_at(x: float, y: float) -> tuple[int, str]:
        return score_at(x, y)

    @staticmethod
    def score_image(
        path: str,
        out_path: str | None = None,
        *,
        image_path: str | None = None,
        debug_dir: str | None = None,
        verbose: bool = True,
    ):
        return score_image(path, out_path, image_path=image_path, debug_dir=debug_dir, verbose=verbose)


def polar(x, y):
    return np.hypot(x, y), np.degrees(np.arctan2(x, y)) % 360


def board_class(r, th):
    idx = np.floor(((th + 9) % 360) / 18).astype(int)
    even = idx % 2 == 0
    c = np.where(even, BLACK, BEIGE)
    ring = np.where(even, RED, GREEN)
    c = np.where((r >= R_TI) & (r < R_TO), ring, c)
    c = np.where((r >= R_DI) & (r < R_DO), ring, c)
    c = np.where(r < R_OB, GREEN, c)
    c = np.where(r < R_IB, RED, c)
    return np.where(r >= R_DO, BLACK, c)


def score_at(x, y):
    r, th = polar(x, y)
    s = SECTORS[int(((th + 9) % 360) // 18)]
    if r < R_IB:
        return 50, "Bull (50)"
    if r < R_OB:
        return 25, "Outer bull (25)"
    if r >= R_DO:
        return 0, "Miss"
    if r >= R_DI:
        return 2 * s, f"D{s}"
    if R_TI <= r < R_TO:
        return 3 * s, f"T{s}"
    return s, f"S{s}"


def wire_margin(x, y):
    r, th = polar(x, y)
    m = np.min(
        np.abs(np.asarray(r)[..., None] - np.array([R_IB, R_OB, R_TI, R_TO, R_DI, R_DO])),
        -1,
    )
    d = (th + 9) % 18
    d = np.minimum(d, 18 - d)
    return np.where(r > R_OB, np.minimum(m, r * np.sin(np.radians(d))), m)


def cone(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    H = hsv[..., 0] / 255 * 2 * np.pi
    S = hsv[..., 1] / 255
    V = hsv[..., 2] / 255
    return np.stack([S * V * np.cos(H), S * V * np.sin(H), V], -1)


def fit_models(F, labels_per_class, rng=np.random.default_rng(0)):
    ms = []
    for sel in labels_per_class:
        X = F[sel]
        X = X[rng.choice(len(X), min(20000, len(X)), replace=False)]
        ms.append(MinCovDet(support_fraction=0.75, random_state=0).fit(X))
    return ms


def classify(F, models, p=0.999):
    D = np.stack([m.mahalanobis(F.reshape(-1, 3)) for m in models], 1).reshape(F.shape[:2] + (4,))
    cls = D.argmin(-1)
    return cls, np.where(D.min(-1) > chi2.ppf(p, 3), OUTLIER, cls)


def find_bull(cls):
    red = (cls == RED).astype(np.uint8)
    grn = cls == GREEN
    n, _, st, cen = cv2.connectedComponentsWithStats(red)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (8 < w < 80 and 8 < h < 80 and 0.6 < w / h < 1.6 and a > 0.5 * w * h):
            continue
        cx, cy = cen[i]
        R = int(max(w, h) * 1.6)
        f = grn[
            max(0, int(cy) - R) : int(cy) + R, max(0, int(cx) - R) : int(cx) + R
        ].mean()
        if best is None or f > best[0]:
            best = (f, np.array([cx, cy]))
    return best[1]


def ring_points(cls, c, rmin, rmax, n=720):
    rg = (cls == RED) | (cls == GREEN)
    pts = []
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        rs = np.arange(rmin, rmax)
        x = (c[0] + rs * np.cos(t)).astype(int)
        y = (c[1] + rs * np.sin(t)).astype(int)
        if x.min() < 0 or y.min() < 0 or x.max() >= cls.shape[1] or y.max() >= cls.shape[0]:
            continue
        h = np.where(rg[y, x])[0]
        if len(h):
            pts.append((x[h[-1]], y[h[-1]]))
    return np.array(pts, np.float64)


def ellipse_dist(p, e):
    (cx, cy), (A, B), ang = e
    a, b = A / 2, B / 2
    t = np.radians(ang)
    x = (p[:, 0] - cx) * np.cos(t) + (p[:, 1] - cy) * np.sin(t)
    y = -(p[:, 0] - cx) * np.sin(t) + (p[:, 1] - cy) * np.cos(t)
    k = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    return np.abs(k - 1) * np.hypot(x, y) / np.maximum(k, 1e-9)


def ellipse_ransac(p, rng=np.random.default_rng(1)):
    best = None
    for _ in range(400):
        try:
            e = cv2.fitEllipse(p[rng.choice(len(p), 6, replace=False)].astype(np.float32))
        except cv2.error:
            continue
        if min(e[1]) < 2:
            continue
        inl = ellipse_dist(p, e) < 4
        if best is None or inl.sum() > best.sum():
            best = inl
    return cv2.fitEllipse(p[best].astype(np.float32))


def conic(e):
    (cx, cy), (A, B), ang = e
    t = np.radians(ang)
    c, s = np.cos(t), np.sin(t)
    M = np.array([[c, s, -c * cx - s * cy], [-s, c, s * cx - c * cy], [0, 0, 1]])
    return M.T @ np.diag([4 / A**2, 4 / B**2, -1]) @ M


def rectify(e, bull):
    bull = np.asarray(bull, dtype=np.float64)
    l = conic(e) @ np.array([*bull, 1.0], dtype=np.float64)
    l /= l[2]
    Hp = np.array([[1, 0, 0], [0, 1, 0], [l[0], l[1], 1.0]], dtype=np.float64)
    (cx, cy), (A, B), ang = e
    P = cv2.ellipse2Poly((int(cx), int(cy)), (int(A / 2), int(B / 2)), int(ang), 0, 360, 1).astype(np.float64)
    q = cv2.perspectiveTransform(P[None], Hp)[0]
    b2 = cv2.perspectiveTransform(np.array([[bull]], dtype=np.float64), Hp)[0, 0]
    sc = 1000 / np.ptp(q[:, 0])
    (ex, ey), (A2, B2), a2 = cv2.fitEllipse(((q - b2) * sc).astype(np.float32))
    t = np.radians(a2)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    Aff = np.eye(3)
    Aff[:2, :2] = R.T @ np.diag([R_DO / (A2 / 2), R_DO / (B2 / 2)]) @ R
    T = np.array([[sc, 0, -b2[0] * sc], [0, sc, -b2[1] * sc], [0, 0, 1]])
    return np.diag([1, -1, 1]) @ Aff @ T @ Hp


def rot(a):
    t = np.radians(a)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def soft(cls, sigma, excl):
    P = np.stack([(cls == c).astype(np.float32) for c in range(4)], -1)
    P[excl] = 0.25
    return (
        np.stack([cv2.GaussianBlur(P[..., c], (0, 0), sigma) for c in range(4)], -1)
        if sigma
        else P
    )


def template(rmax=168):
    r, th = np.meshgrid(np.arange(2, rmax, 1.0), np.arange(0, 360, 0.5))
    r, th = r.ravel(), th.ravel()
    return np.stack([r * np.sin(np.radians(th)), r * np.cos(np.radians(th))], 1), board_class(r, th)


def agreement(G, P, tp, tc):
    q = cv2.perspectiveTransform(tp[None], G)[0]
    ok = (
        (q[:, 0] >= 0)
        & (q[:, 0] < P.shape[1])
        & (q[:, 1] >= 0)
        & (q[:, 1] < P.shape[0])
    )
    n = len(q)
    m = -(-n // 1024) * 1024
    xp = np.pad(q[:, 0], (0, m - n)).astype(np.float32).reshape(-1, 1024)
    yp = np.pad(q[:, 1], (0, m - n)).astype(np.float32).reshape(-1, 1024)
    V = cv2.remap(P, xp, yp, cv2.INTER_LINEAR).reshape(-1, 4)[:n]
    return (V[np.arange(n), tc] * ok).sum() / n


def calibrate(cls, cls_o):
    bull = find_bull(cls)
    e = ellipse_ransac(ring_points(cls_o, bull, 250, 560))
    H0 = rectify(e, bull)
    G0 = np.linalg.inv(H0)
    tp, tc = template()
    P = soft(cls, 6, cls_o == OUTLIER)
    angs = np.arange(0, 360, 0.5)
    sc = np.array([agreement(G0 @ rot(a), P, tp, tc) for a in angs])

    def up_dev(a):
        q = cv2.perspectiveTransform(np.array([[[0, 0], [0, 100.0]]]), G0 @ rot(a))[0]
        d = q[1] - q[0]
        return abs((np.degrees(np.arctan2(d[0], -d[1])) + 180) % 360 - 180)

    cand = [a for a, s in zip(angs, sc) if s > sc.max() - 0.02]
    G = G0 @ rot(min(cand, key=up_dev))
    for s in (6, 3, 1.5):
        P = soft(cls, s, cls_o == OUTLIER)

        def f(p):
            return -agreement(
                G @ (np.eye(3) + np.r_[p[:6], p[6:] * 1e-3, 0].reshape(3, 3)),
                P,
                tp,
                tc,
            )

        p = minimize(f, np.zeros(8), method="Powell", options={"xtol": 1e-4, "ftol": 1e-6, "maxfev": 6000}).x
        G = G @ (np.eye(3) + np.r_[p[:6], p[6:] * 1e-3, 0].reshape(3, 3))
    return G, agreement(G, soft(cls, 0, cls_o == OUTLIER), tp, tc)


def barrels_and_flights(im, cls_o, r):
    hsv = cv2.cvtColor(cv2.medianBlur(im, 3), cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    S, V = hsv[..., 1] / 255, hsv[..., 2] / 255
    out = cls_o == OUTLIER
    thin = ((V > 0.2) & (V < 0.72) & (S < 0.55) & out & (r < R_DO + 70)).astype(np.uint8)
    thick = cv2.morphologyEx(thin, cv2.MORPH_OPEN, ell(7))
    n, cc = cv2.connectedComponents(thick)
    inside = np.unique(cc[(thick > 0) & (r < R_DO + 10)])
    inside = inside[inside > 0]
    thick = np.isin(cc, inside).astype(np.uint8)
    white = ((V > 0.72) & (S < 0.30) & out).astype(np.uint8)
    flight = cv2.morphologyEx(white, cv2.MORPH_OPEN, ell(9))
    return thin, thick, flight, white


def runs(t, gap=12):
    o = np.argsort(t)
    ts = t[o]
    segs = np.split(np.arange(len(ts)), np.where(np.diff(ts) > gap)[0] + 1)
    s = max(segs, key=len)
    return o[s], ts[s[0]], ts[s[-1]]


def find_darts(thick, flight, white, rng=np.random.default_rng(0)):
    ys, xs = np.nonzero(thick)
    P = np.stack([xs, ys], 1).astype(float)
    fy, fx = np.nonzero(flight)
    FP = np.stack([fx, fy], 1).astype(float)
    rem = np.ones(len(P), bool)
    out = []
    for _ in range(8):
        idx = np.where(rem)[0]
        if len(idx) < 80:
            break
        best = None
        for _ in range(1500):
            a, b = P[rng.choice(idx, 2, replace=False)]
            d = b - a
            L = np.hypot(*d)
            if L < 20:
                continue
            n = np.array([-d[1], d[0]]) / L
            inl = idx[np.abs((P[idx] - a) @ n) < 7]
            if len(inl) < 25:
                continue
            k = len(runs((P[inl] - a) @ (d / L))[0])
            if best is None or k > best[0]:
                best = (k, a, d / L)
        if best is None:
            break
        _, a, u = best
        n = np.array([-u[1], u[0]])
        inl = idx[np.abs((P[idx] - a) @ n) < 7]
        m = P[inl].mean(0)
        u = np.linalg.svd(P[inl] - m)[2][0]
        n = np.array([-u[1], u[0]])
        inl = idx[np.abs((P[idx] - m) @ n) < 7]
        sel, tl, th = runs((P[inl] - m) @ u)
        width = 2 * np.percentile(np.abs((P[inl[sel]] - m) @ n), 95)
        tt = (P - m) @ u
        rem &= ~((np.abs((P - m) @ n) < 10) & (tt > tl - 5) & (tt < th + 5))
        if th - tl < 22 or width > 30:
            continue
        e1, e2 = m + tl * u, m + th * u
        near = FP[np.abs((FP - m) @ n) < 60]
        if not len(near):
            continue

        def white_run(e, dirv):
            nv = np.array([-dirv[1], dirv[0]])
            run = gap = 0
            for k in range(1, 46):
                hit = False
                for o in (-4, -2, 0, 2, 4):
                    q = e + k * dirv + o * nv
                    x, y = int(round(q[0])), int(round(q[1]))
                    if 0 <= y < white.shape[0] and 0 <= x < white.shape[1] and white[y, x]:
                        hit = True
                        break
                if hit:
                    run, gap = k, 0
                else:
                    gap += 1
                    if gap > (6 if run == 0 else 4):
                        break
            return run

        w1, w2 = white_run(e1, -u), white_run(e2, u)
        if max(w1, w2) >= 10 and min(w1, w2) < 0.4 * max(w1, w2):
            tip, tail = (e2, e1) if w1 > w2 else (e1, e2)
        else:
            tf = np.median((near - m) @ u)
            tip, tail = (e2, e1) if abs(th - tf) > abs(tl - tf) else (e1, e2)
        bp = P[inl[sel]]
        if th - tl < 50 and len(FP):
            fl = FP[np.hypot(*(FP - bp.mean(0)).T) < 90]
            if len(fl) > 100:
                v = bp.mean(0) - fl.mean(0)
                v /= np.linalg.norm(v)
                tp = bp @ v
                tip = bp[tp > np.percentile(tp, 95)].mean(0)
                tail = bp[tp < np.percentile(tp, 5)].mean(0)
        ax = (tip - tail) / np.linalg.norm(tip - tail)
        nn = np.array([-ax[1], ax[0]])
        F = FP - tail
        behind = ((F @ -ax > -10) & (F @ -ax < 80) & (np.abs(F @ nn) < 45)).sum()
        if behind < 300:
            continue
        out.append(dict(tip_barrel=tip, tail=tail, axis=ax, length=th - tl, width=width))
    return out


def extend_tip(d, thin, max_ext=25):
    """Extend only while the shaft remains consistently thick along the axis.

    The original logic used ``any(...)`` on a five-pixel cross-section, which let a
    single stray thin pixel keep the tip moving well past the actual dart point.
    That over-extension is exactly how a sector-4 dart can drift into sector 13.
    Requiring a majority of the cross-section to remain in the thin barrel keeps the
    endpoint anchored to the real shaft tip instead of a noisy continuation.
    """
    u = d["axis"]
    n = np.array([-u[1], u[0]])
    p0 = d["tip_barrel"]
    last = miss = 0
    for s in range(1, max_ext + 1):
        hits = 0
        for o in (-2, -1, 0, 1, 2):
            q = p0 + s * u + o * n
            x, y = int(round(q[0])), int(round(q[1]))
            if (
                0 <= y < thin.shape[0]
                and 0 <= x < thin.shape[1]
                and thin[y, x]
            ):
                hits += 1
        hit = hits >= 4
        if hit:
            last, miss = s, 0
        else:
            miss += 1
            if miss > 3:
                break
    return p0 + last * u


TARGET_R = 420.0


def normalise_scale(im):
    """Rescale so the double ring has ~TARGET_R px radius."""
    s = 1600 / max(im.shape[:2])
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    F = cone(im)
    seeds = np.array(
        [[q * v * np.cos(np.radians(h)), q * v * np.sin(np.radians(h)), v] for h, q, v in PROTOS]
    )
    cls = np.argmin(((F[..., None, :] - seeds) ** 2).sum(-1), -1)
    bull = find_bull(cls)
    e = ellipse_ransac(ring_points(cls, bull, 120, int(0.48 * max(im.shape[:2]))))
    R = np.mean(e[1]) / 2
    k = TARGET_R / R
    return cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def score_image(path, out_path=None, image_path: str | None = None, debug_dir: str | None = None, verbose: bool = True):
    """Score a single image and optionally dump the debug trail for a failing run.

    Parameters
    ----------
    path:
        Input image path used for the scorer. This is kept for backward
        compatibility with the CLI entry point.
    out_path:
        Optional annotated output path.
    image_path:
        Explicit image path override for notebook/debug workflows. If supplied,
        it takes precedence over ``path``.
    debug_dir:
        Optional directory where intermediate masks, images, and JSON metrics are
        saved to make debugging easier.
    verbose:
        If true, print a concise pipeline summary including agreement, detected
        darts, total score, and the output debug folder when active.
    """
    source = image_path or path
    source_path = Path(source)
    debug_root = Path(debug_dir) if debug_dir is not None else None
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)

    im = normalise_scale(cv2.imread(source))
    if debug_root is not None:
        cv2.imwrite(str(debug_root / "01_normalised.png"), im)
    imf = cv2.bilateralFilter(cv2.medianBlur(im, 5), 9, 40, 7)
    F, Ff = cone(im), cone(imf)
    seeds = np.array(
        [[s * v * np.cos(np.radians(h)), s * v * np.sin(np.radians(h)), v] for h, s, v in PROTOS]
    )
    nearest = np.argmin(((F[..., None, :] - seeds) ** 2).sum(-1), -1)
    hh, ww = im.shape[:2]
    reg = np.zeros((hh, ww), bool)
    reg[hh // 6 : -hh // 12, ww // 8 : -ww // 24] = True
    cls, cls_o = classify(Ff, fit_models(F, [reg & (nearest == c) for c in range(4)]))

    if debug_root is not None:
        cv2.imwrite(str(debug_root / "02_outlier_mask.png"), (cls_o == OUTLIER).astype(np.uint8) * 255)
        cv2.imwrite(str(debug_root / "03_red_mask.png"), (cls == RED).astype(np.uint8) * 255)
        cv2.imwrite(str(debug_root / "04_green_mask.png"), (cls == GREEN).astype(np.uint8) * 255)

    bull = find_bull(cls)
    if np.isnan(bull).any() or np.allclose(bull, 0):
        bull = np.array([ww // 2, hh // 2], dtype=float)
    if debug_root is not None:
        bull_img = im.copy()
        cv2.circle(bull_img, tuple(np.round(bull).astype(int)), 12, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(debug_root / "05_bull_mark.png"), bull_img)

    e = ellipse_ransac(ring_points(cls_o, bull, 250, 560))
    if debug_root is not None and e is not None:
        ellipse_img = im.copy()
        (cx, cy), (A, B), ang = e
        cv2.ellipse(ellipse_img, (int(round(cx)), int(round(cy))), (int(round(A / 2)), int(round(B / 2))), int(round(ang)), 0, 360, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(debug_root / "05b_ellipse.png"), ellipse_img)

    G, agree = calibrate(cls, cls_o)
    H = np.linalg.inv(G)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    b = cv2.perspectiveTransform(np.stack([xx.ravel(), yy.ravel()], 1)[None], H)[0]
    r, th = polar(b[:, 0], b[:, 1])
    r = r.reshape(hh, ww)
    th = th.reshape(hh, ww)
    exp = board_class(r, th)
    wm = wire_margin(b[:, 0], b[:, 1]).reshape(hh, ww)
    cls, cls_o = classify(Ff, fit_models(F, [(r < R_DO - 1) & (exp == c) & (wm > 2.5) for c in range(4)]))
    thin, thick, flight, white = barrels_and_flights(im, cls_o, r)
    darts = find_darts(thick, flight, white)
    if debug_root is not None:
        cv2.imwrite(str(debug_root / "06_stage2_class.png"), (cls * 65).clip(0, 255).astype(np.uint8))
        cv2.imwrite(str(debug_root / "07_thin.png"), (thin * 255).astype(np.uint8))
        cv2.imwrite(str(debug_root / "08_thick.png"), (thick * 255).astype(np.uint8))
        cv2.imwrite(str(debug_root / "09_flight.png"), (flight * 255).astype(np.uint8))
        cv2.imwrite(str(debug_root / "10_white.png"), (white * 255).astype(np.uint8))

    total = 0
    res = []
    for d in darts:
        tip = extend_tip(d, thin)
        u = d["axis"]
        n = np.array([-u[1], u[0]])
        bt = cv2.perspectiveTransform(tip.reshape(1, 1, 2), H)[0, 0]
        sc, lab = score_at(*bt)
        votes = Counter(
            score_at(*cv2.perspectiveTransform((tip + a * u + c * n).reshape(1, 1, 2), H)[0, 0])
            for a in range(-4, 5, 2)
            for c in (-2, 0, 2)
        )
        conf = votes[(sc, lab)] / sum(votes.values())
        alts = [l for (s2, l), _ in votes.most_common() if l != lab]
        res.append(
            dict(
                tip=tip,
                tail=d["tail"],
                board_mm=bt,
                score=sc,
                label=lab,
                conf=conf,
                alts=alts,
                margin=float(wire_margin(*bt)),
            )
        )
        total += sc
    res.sort(key=lambda d: d["tip"][0])

    if debug_root is not None:
        debug_metrics = {
            "source": str(source_path),
            "image_shape": list(im.shape),
            "agreement": float(agree),
            "bull_px": np.round(bull).astype(float).tolist(),
            "ellipse": {
                "center": np.round(e[0]).astype(float).tolist() if e is not None else None,
                "axes": np.round(e[1]).astype(float).tolist() if e is not None else None,
                "angle": float(e[2]) if e is not None else None,
            },
            "n_detected_darts": len(darts),
            "n_scored_darts": len(res),
            "total": int(total),
            "darts": [
                {
                    "index": i + 1,
                    "tip_px": np.round(item["tip"]).astype(float).tolist(),
                    "tail_px": np.round(item["tail"]).astype(float).tolist(),
                    "board_mm": np.round(item["board_mm"]).astype(float).tolist(),
                    "score": int(item["score"]),
                    "label": item["label"],
                    "margin_mm": float(item["margin"]),
                    "confidence": float(item["conf"]),
                    "alts": item["alts"],
                }
                for i, item in enumerate(res)
            ],
        }
        (debug_root / "debug_metrics.json").write_text(json.dumps(debug_metrics, indent=2))

    if out_path:
        annotate(im, G, res, total, agree, out_path)
    if debug_root is not None:
        annotate(im, G, res, total, agree, str(debug_root / "11_annotated.png"))

    if verbose:
        print(f"source={source_path}")
        print(f"agreement={agree:.1%}  darts={len(res)}  total={total}")
        if debug_root is not None:
            print(f"debug_dir={debug_root}")
            print(f"debug_metrics={debug_root / 'debug_metrics.json'}")
    return total, res, agree


def annotate(im, G, res, total, agree, path):
    d = im.copy()
    t = np.linspace(0, 2 * np.pi, 721)
    col = (255, 230, 0)
    for rr in (R_IB, R_OB, R_TI, R_TO, R_DI, R_DO):
        q = cv2.perspectiveTransform(np.stack([rr * np.sin(t), rr * np.cos(t)], 1)[None], G)[0]
        cv2.polylines(d, [q.astype(np.int32)], True, col, 1, cv2.LINE_AA)
    for k in range(20):
        a = np.radians(9 + 18 * k)
        q = cv2.perspectiveTransform(
            np.array([[[R_OB * np.sin(a), R_OB * np.cos(a)], [R_DO * np.sin(a), R_DO * np.cos(a)]]]),
            G,
        )[0]
        cv2.line(d, tuple(q[0].astype(int)), tuple(q[1].astype(int)), col, 1, cv2.LINE_AA)
    for i, x in enumerate(res):
        cv2.line(d, tuple(x["tail"].astype(int)), tuple(x["tip"].astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(d, tuple(x["tip"].astype(int)), 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(d, tuple(x["tip"].astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        txt = f"{i + 1}: {x['label']} = {x['score']}"
        lx = int(max(r["tip"][0] for r in res)) + 90
        rank = sorted(r["tip"][1] for r in res).index(x["tip"][1])
        ly = int(min(r["tip"][1] for r in res)) + 60 * rank + 40
        cv2.line(d, tuple(x["tip"].astype(int)), (lx - 8, ly - 10), (255, 255, 255), 2, cv2.LINE_AA)
        p = (lx, ly)
        cv2.putText(d, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(d, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    banner = f"TOTAL: {total}"
    cv2.rectangle(d, (0, 0), (d.shape[1], 70), (30, 30, 30), -1)
    cv2.putText(d, banner, (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(
        d,
        f"calibration agreement {agree * 100:.0f}%",
        (d.shape[1] - 430, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(path, d)


if __name__ == "__main__":
    total, res, agree = score_image(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "scored.jpg")
    print(f"calibration agreement: {agree:.1%}")
    for i, x in enumerate(res, 1):
        alt = f"  (alt: {', '.join(x['alts'])})" if x['alts'] else ""
        print(
            f"dart {i}: {x['label']:>4} = {x['score']:>2}   tip {x['board_mm'].round(1)} mm   "
            f"wire margin {x['margin']:.1f} mm   confidence {x['conf']:.0%}{alt}"
        )
    print("TOTAL", total)
