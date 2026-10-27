"""Self-referenced single-image dart and tip detector.

Uses dense top-down anomaly reconstruction, distance transform candidate peaks,
vectorized inward ray searches, center-minus-flank barrel spine contrast filtering,
lateral ridge snapping, and morphological needle tracing to localize dart tips to
sub-millimeter precision.
"""

from __future__ import annotations

import math
import cv2
import numpy as np

from .detect import Detection
from .geometry import RADII, contour_radius, project, score_point_detailed
from .types import DartDetection, DartDiagnostics

TOPDOWN_SIZE = 720
MAX_REFINE_DEG = 25.0
NEEDLE_MAX_MM = 14.0
NEEDLE_GAP_MM = 2.0
NEEDLE_START_MM = 3.0
NEEDLE_MIN_EVIDENCE = 12.0
TOPDOWN_EXTENT_MM = 360.0


def pt_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point p to line segment ab in mm."""
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-6:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _refine_axis(
    thin9_axial: np.ndarray,
    c: np.ndarray,
    ang_deg: float,
    t_lo: float,
    t_hi: float,
    s: float,
) -> tuple[np.ndarray, float] | None:
    """Re-aim a dart axis through the lateral ridge centres of its barrel segment.

    Samples the thin-structure map across the axis for t in [t_lo, t_hi], takes the strongest lateral offset
    within +-6 mm per station and fits offset(t) = a + b t by iteratively trimmed least squares.
    Returns the shifted origin and corrected angle, or None when the fit is unsupported.
    """
    if t_hi - t_lo < 15.0:
        return None
    rad = math.radians(ang_deg)
    d = np.array([math.cos(rad), math.sin(rad)])
    n = np.array([-d[1], d[0]])
    ts = np.arange(t_lo, t_hi, 1.5)
    ws = np.arange(-6.0, 6.01, 0.5)
    P = c[None, None, :] + ts[:, None, None] * d + ws[None, :, None] * n
    U = (P[..., 0] * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
    V = (P[..., 1] * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
    vals = cv2.remap(thin9_axial, U, V, cv2.INTER_LINEAR)
    vals = cv2.blur(vals, (3, 1))
    idx = np.argmax(vals, axis=1)
    strength = vals[np.arange(len(ts)), idx] - np.median(vals, axis=1)
    off = ws[idx]
    keep = strength >= 0.5 * float(strength.max())
    if keep.sum() < 8:
        return None
    for _ in range(3):
        A = np.stack([np.ones(int(keep.sum())), ts[keep]], axis=1)
        coef, *_ = np.linalg.lstsq(A, off[keep], rcond=None)
        resid = np.abs(off - (coef[0] + coef[1] * ts))
        new_keep = keep & (resid <= 2.0)
        if new_keep.sum() < 8:
            return None
        keep = new_keep
    a0, b0 = float(coef[0]), float(coef[1])
    d_ang = math.degrees(math.atan(b0))
    if abs(a0) < 0.3 and abs(d_ang) < 0.5:
        return None
    d_ang = float(np.clip(d_ang, -MAX_REFINE_DEG, MAX_REFINE_DEG))
    return c + a0 * n, ang_deg + d_ang


def _bridge_gap(
    S_nw: np.ndarray,
    fine_ts: np.ndarray,
    start_idx: int,
    drop_idx: int,
    max_gap_mm: float = 25.0,
    min_seg_mm: float = 10.0,
) -> int | None:
    """Find barrel resurgence beyond ``drop_idx``; returns the new drop index, -1 for no end, or None."""
    step = float(fine_ts[1] - fine_ts[0])
    seg = int(round(min_seg_mm / step))
    peak_nw = float(np.max(S_nw[start_idx : drop_idx + 1]))
    thr = max(15.0, 0.40 * peak_nw)
    thr_drop = max(10.0, 0.28 * peak_nw)
    last = min(len(fine_ts) - seg - 8, drop_idx + int(round(max_gap_mm / step)))
    for k in range(drop_idx + 1, last):
        if S_nw[k] >= thr and np.mean(S_nw[k : k + seg] >= thr) >= 0.8:
            for j in range(k + seg, len(fine_ts) - 8):
                if np.all(S_nw[j : j + 8] < thr_drop):
                    return j
            return -1
    return None


def _same_axis(a: DartDetection, b: DartDetection) -> bool:
    """True when two candidates lie on one dart axis and overlap along it."""
    ang = abs((a.ray_angle - b.ray_angle + 180.0) % 360.0 - 180.0)
    if ang > 10.0:
        return False
    rad = math.radians(b.ray_angle)
    d = np.array([math.cos(rad), math.sin(rad)])
    n = np.array([-d[1], d[0]])
    rel_c = a.c_mm - b.c_mm
    rel_t = a.tip_mm - b.c_mm
    if abs(float(rel_c @ n)) > 7.0 or abs(float(rel_t @ n)) > 5.0:
        return False
    a_lo, a_hi = sorted((float(rel_c @ d), float(rel_t @ d)))
    b_lo, b_hi = sorted((0.0, float((b.tip_mm - b.c_mm) @ d)))
    return min(a_hi, b_hi) - max(a_lo, b_lo) >= 10.0


def _build_board_class_map(board: Detection, r_mm: np.ndarray, ang_deg: np.ndarray) -> np.ndarray:
    """Build continuous geometric board class map adjusted to measured ring boundaries.

    Classes:
        0: inner bull (red)
        1: outer bull (green)
        2: black single beds (even sectors)
        3: cream single beds (odd sectors)
        4: red rings (treble/double on even sectors)
        5: green rings (treble/double on odd sectors)
        6: black catch ring
       -1: outer surround / wall
    """
    sec_idx = (ang_deg // 18.0).astype(int) % 20
    even_sec = (sec_idx % 2 == 0)

    radii_lut = {}
    angles_360 = np.arange(360, dtype=np.float64)
    for name in ("treble_inner", "treble_outer", "double_inner", "double_outer"):
        if getattr(board, "rings", None) and name in board.rings:
            val = board.rings[name]
            contour = val.contour if hasattr(val, "contour") else val
            radii_lut[name] = contour_radius(board.H, contour, angles_360)
        else:
            radii_lut[name] = np.full(360, RADII[name], dtype=np.float64)

    ang_int = (ang_deg.astype(int)) % 360
    ti = radii_lut["treble_inner"][ang_int]
    to = radii_lut["treble_outer"][ang_int]
    di = radii_lut["double_inner"][ang_int]
    do = radii_lut["double_outer"][ang_int]

    c_map = np.full(r_mm.shape, -1, dtype=int)
    c_map[r_mm < RADII["bull"]] = 0
    c_map[(r_mm >= RADII["bull"]) & (r_mm < RADII["outer_bull"])] = 1
    c_map[(r_mm >= RADII["outer_bull"]) & (r_mm < ti) & even_sec] = 2
    c_map[(r_mm >= RADII["outer_bull"]) & (r_mm < ti) & (~even_sec)] = 3
    c_map[(r_mm >= ti) & (r_mm <= to) & even_sec] = 4
    c_map[(r_mm >= ti) & (r_mm <= to) & (~even_sec)] = 5
    c_map[(r_mm > to) & (r_mm < di) & even_sec] = 2
    c_map[(r_mm > to) & (r_mm < di) & (~even_sec)] = 3
    c_map[(r_mm >= di) & (r_mm <= do) & even_sec] = 4
    c_map[(r_mm >= di) & (r_mm <= do) & (~even_sec)] = 5
    c_map[(r_mm > do) & (r_mm <= 220.0)] = 6
    return c_map


def _build_wire_mask(board: Detection, r_mm: np.ndarray, ang_deg: np.ndarray) -> np.ndarray:
    """Mask covering nominal/measured circular rings and 20 radial wires within +-2.2 mm."""
    radii_lut = {}
    angles_360 = np.arange(360, dtype=np.float64)
    for name in ("treble_inner", "treble_outer", "double_inner", "double_outer"):
        if getattr(board, "rings", None) and name in board.rings:
            val = board.rings[name]
            contour = val.contour if hasattr(val, "contour") else val
            radii_lut[name] = contour_radius(board.H, contour, angles_360)
        else:
            radii_lut[name] = np.full(360, RADII[name], dtype=np.float64)

    ang_int = (ang_deg.astype(int)) % 360
    ti = radii_lut["treble_inner"][ang_int]
    to = radii_lut["treble_outer"][ang_int]
    di = radii_lut["double_inner"][ang_int]
    do = radii_lut["double_outer"][ang_int]

    ring_mask = (
        (np.abs(r_mm - RADII["bull"]) <= 2.2) |
        (np.abs(r_mm - RADII["outer_bull"]) <= 2.2) |
        (np.abs(r_mm - ti) <= 2.8) |
        (np.abs(r_mm - to) <= 2.8) |
        (np.abs(r_mm - di) <= 2.8) |
        (np.abs(r_mm - do) <= 2.8) |
        (np.abs(r_mm - 195.0) <= 3.2)
    )
    d_ang = np.abs((ang_deg % 18.0) - 0.0)
    d_ang = np.minimum(d_ang, 18.0 - d_ang)
    wire_dist = d_ang * (np.pi / 180.0) * r_mm
    radial_mask = (wire_dist <= 2.5) & (r_mm >= 15.0) & (r_mm <= 172.0)
    return ring_mask | radial_mask


def _reconstruct_expected_board(
    topdown: np.ndarray,
    c_map: np.ndarray,
    sec_idx: np.ndarray,
    on_wire: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct expected LAB and BGR boards using 3-sector angular medians."""
    lab = cv2.cvtColor(topdown, cv2.COLOR_BGR2LAB).astype(np.float32)
    expected_lab = np.zeros_like(lab)

    for c in range(7):
        mask = (c_map == c) & (~on_wire) & valid_mask
        if not mask.any():
            mask = (c_map == c) & valid_mask
            if not mask.any():
                continue

        if c in (2, 3, 4, 5):
            sec_meds = {}
            for k in range(20):
                m_k = mask & (sec_idx == k)
                if m_k.sum() > 20:
                    sec_meds[k] = np.median(lab[m_k], axis=0)

            global_med = np.median(list(sec_meds.values()), axis=0) if sec_meds else np.median(lab[mask], axis=0)

            for k in range(20):
                m_k = (c_map == c) & (sec_idx == k)
                if not m_k.any():
                    continue

                nbrs = [sec_meds[kn] for kn in ((k - 2) % 20, k, (k + 2) % 20) if kn in sec_meds]
                if not nbrs:
                    expected_lab[m_k] = global_med
                else:
                    expected_lab[m_k] = np.median(nbrs, axis=0)
        else:
            expected_lab[c_map == c] = np.median(lab[mask], axis=0)

    outer_mask = (c_map < 0) & valid_mask
    if outer_mask.any():
        c6_mask = (c_map == 6) & valid_mask
        if c6_mask.any():
            expected_lab[outer_mask] = np.median(lab[c6_mask], axis=0)
        else:
            expected_lab[outer_mask] = np.array([30.0, 128.0, 128.0], dtype=np.float32)

    expected_bgr = cv2.cvtColor(
        np.clip(expected_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )
    return expected_lab, expected_bgr


def detect_darts(
    img: np.ndarray,
    board: Detection,
    *,
    max_darts: int = 3,
    diagnostics: bool = False,
) -> tuple[list[DartDetection], DartDiagnostics | None]:
    """Detect dart tips and estimate scores from a single board image.

    Args:
        img: Original BGR photograph.
        board: Board detection containing the model mm -> image px homography.
        max_darts: Maximum number of darts to extract (1 to 3).
        diagnostics: Whether to capture top-down anomaly, ridge, and strip diagnostic data.

    Returns:
        (darts, diagnostics): Detected darts and optional DartDiagnostics.
    """
    s = TOPDOWN_SIZE / (2.0 * TOPDOWN_EXTENT_MM)  # 1.0 px/mm
    T_mat = np.array(
        [[s, 0.0, TOPDOWN_SIZE / 2.0], [0.0, s, TOPDOWN_SIZE / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    M = T_mat @ np.linalg.inv(board.H)
    topdown = cv2.warpPerspective(img, M, (TOPDOWN_SIZE, TOPDOWN_SIZE), flags=cv2.INTER_LINEAR)
    img_mask = np.ones(img.shape[:2], dtype=np.uint8)
    valid_mask = cv2.warpPerspective(img_mask, M, (TOPDOWN_SIZE, TOPDOWN_SIZE), flags=cv2.INTER_NEAREST) > 0

    yy, xx = np.mgrid[:TOPDOWN_SIZE, :TOPDOWN_SIZE]
    x_mm = (xx - TOPDOWN_SIZE / 2.0) / s
    y_mm = (yy - TOPDOWN_SIZE / 2.0) / s
    r_mm = np.hypot(x_mm, y_mm)
    ang_deg = (np.degrees(np.arctan2(x_mm, -y_mm)) + 9.0) % 360.0
    sec_idx = (ang_deg // 18.0).astype(int) % 20

    c_map = _build_board_class_map(board, r_mm, ang_deg)
    on_wire = _build_wire_mask(board, r_mm, ang_deg)
    expected_lab, expected_bgr = _reconstruct_expected_board(topdown, c_map, sec_idx, on_wire, valid_mask)

    lab = cv2.cvtColor(topdown, cv2.COLOR_BGR2LAB).astype(np.float32)
    dL = lab[..., 0] - expected_lab[..., 0]
    da = lab[..., 1] - expected_lab[..., 1]
    db = lab[..., 2] - expected_lab[..., 2]
    dC = np.hypot(da, db)

    # Cast shadow separation: darker on bright sectors without chromatic shift
    is_shadow = (dL < -20.0) & (dC < 14.0)

    anomaly = np.hypot(dL, dC * 1.5)
    anomaly[is_shadow] = 0.0
    anomaly[~valid_mask] = 0.0

    # Thin structure filters:
    # 1. 9x9 kernel for metallic barrel & shaft ridges
    gray = cv2.cvtColor(topdown, cv2.COLOR_BGR2GRAY)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat9 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k9).astype(np.float32)
    blackhat9 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k9).astype(np.float32)
    thin9 = tophat9 + blackhat9

    # Angle search on specular bright ridges (invariant to flight color and cast shadow)
    tophat9_angle = tophat9.copy()
    tophat9_angle[is_shadow] = 0.0
    tophat9_angle[on_wire] = 0.0
    tophat9_angle[~valid_mask] = 0.0

    # Axial 1D barrel trace (transverse wires NOT zeroed out so wires don't truncate barrel)
    thin9_axial = thin9.copy()
    thin9_axial[~valid_mask] = 0.0

    # Wire-suppressed variant used to bridge gaps in the barrel without following ring/radial wires
    thin9_nowire = tophat9_angle.copy()

    # 2. 5x5 kernel for steel needle tip tracing (8 mm rings yield bit-exact 0.0)
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat5 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k5).astype(np.float32)
    blackhat5 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k5).astype(np.float32)
    thin5 = tophat5 + 0.35 * blackhat5

    # Needle trace along axial ray (transverse wires NOT zeroed out so needle reaches sisal)
    thin5_axial = thin5.copy()
    thin5_axial[is_shadow] = 0.0
    thin5_axial[(r_mm >= 180.0) & (r_mm <= 220.0)] = 0.0
    thin5_axial[~valid_mask] = 0.0

    # Tophat5 with wires, shadows, and invalid regions zeroed out (strictly 0.0 on synthetic boards and wires)
    tophat5_no_wire = tophat5.copy()
    tophat5_no_wire[on_wire] = 0.0
    tophat5_no_wire[is_shadow] = 0.0
    tophat5_no_wire[~valid_mask] = 0.0

    # Inscribed radius candidate extraction:
    # A. Playable board & catch ring (r <= 220 mm)
    raw_board_mask = (anomaly > 36.0) & valid_mask & (r_mm <= 220.0)
    dist_mm = cv2.distanceTransform(raw_board_mask.astype(np.uint8), cv2.DIST_L2, 5) / s
    k5_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(dist_mm, k5_ellipse)
    is_local_max = (dist_mm == dil) & (dist_mm > 0)
    peaks_inner = is_local_max & (
        ((r_mm <= 170.0) & (dist_mm >= 3.6)) |
        ((r_mm > 170.0) & (dist_mm >= 8.5))
    )

    py_idx, px_idx = np.where(peaks_inner)
    cands_with_score = [
        (float(dist_mm[py, px]), np.array([x_mm[py, px], y_mm[py, px]]))
        for py, px in zip(py_idx, px_idx)
    ]

    # B. Outer surround (170 < r <= 245 mm): bright white flights
    outer_flight_mask = (r_mm > 170.0) & (r_mm <= 245.0) & (lab[..., 0] > 175.0) & valid_mask
    dist_out = cv2.distanceTransform(outer_flight_mask.astype(np.uint8), cv2.DIST_L2, 5) / s
    dil_out = cv2.dilate(dist_out, k5_ellipse)
    is_local_max_out = (dist_out == dil_out) & (dist_out > 0)
    peaks_outer = is_local_max_out & (dist_out >= 5.5) & (dist_out <= 26.0)
    py_o, px_o = np.where(peaks_outer)
    for py, px in zip(py_o, px_o):
        cands_with_score.append((float(dist_out[py, px]), np.array([x_mm[py, px], y_mm[py, px]])))

    # Sort descending by distance from origin so the flight tail is prioritized over the shaft
    cands_with_score.sort(key=lambda item: float(np.hypot(item[1][0], item[1][1])), reverse=True)

    # Deduplicate candidate peaks with 18 mm NMS
    kept_peaks = []
    for _, c_mm in cands_with_score:
        if not any(np.linalg.norm(c_mm - kp) < 18.0 for kp in kept_peaks):
            kept_peaks.append(c_mm)

    angles = np.arange(0, 360, 1.0, dtype=np.float32)
    rads = np.radians(angles)
    all_dx = np.cos(rads)
    all_dy = np.sin(rads)

    ts_coarse = np.linspace(12.0, 85.0, 41, dtype=np.float32)
    ws_search = np.array([-9.0, -7.0, -2.0, 0.0, 2.0, 7.0, 9.0], dtype=np.float32)
    fine_ts = np.linspace(-15.0, 160.0, 351, dtype=np.float32)

    found_candidates = []
    for c_mm in kept_peaks:
        c_r = float(np.hypot(c_mm[0], c_mm[1]))
        dot = c_mm[0] * all_dx + c_mm[1] * all_dy
        t_star = np.clip(-dot, 0.0, 130.0)
        rx = c_mm[0] + t_star * all_dx
        ry = c_mm[1] + t_star * all_dy
        min_r = np.hypot(rx, ry)

        req_drop = 8.0 if c_r >= 35.0 else 4.0
        if c_r > 175.0:
            inward_proj = dot / c_r
            valid_angles = np.where((min_r < c_r - req_drop) & (inward_proj < -0.58))[0]
        else:
            valid_angles = np.where(min_r < c_r - req_drop)[0]

        if len(valid_angles) == 0:
            continue

        sel_dx = all_dx[valid_angles]
        sel_dy = all_dy[valid_angles]
        M_ang = len(valid_angles)
        Tg = ts_coarse[None, :, None]
        Wg = ws_search[None, None, :]
        Dx = sel_dx[:, None, None]
        Dy = sel_dy[:, None, None]
        Nx = -Dy
        Ny = Dx

        Xg = c_mm[0] + Tg * Dx + Wg * Nx
        Yg = c_mm[1] + Tg * Dy + Wg * Ny
        Ug = (Xg * s + TOPDOWN_SIZE / 2.0).astype(np.float32).reshape(M_ang, -1)
        Vg = (Yg * s + TOPDOWN_SIZE / 2.0).astype(np.float32).reshape(M_ang, -1)

        remap_vals = cv2.remap(tophat9_angle, Ug, Vg, cv2.INTER_LINEAR).reshape(M_ang, len(ts_coarse), len(ws_search))
        C_c = remap_vals[:, :, [2, 3, 4]].max(axis=2)
        F_c = remap_vals[:, :, [0, 1, 5, 6]].mean(axis=2)
        S_c = np.maximum(0.0, C_c - 0.85 * F_c)
        coarse_scores = S_c.mean(axis=1)

        best_idx = int(np.argmax(coarse_scores))
        best_score = float(coarse_scores[best_idx])
        if best_score < 18.0:
            continue

        best_ang = angles[valid_angles[best_idx]]
        fine_angles = best_ang + np.linspace(-16.0, 16.0, 129, dtype=np.float32)
        f_rads = np.radians(fine_angles)
        f_dx = np.cos(f_rads)
        f_dy = np.sin(f_rads)

        M_f = len(fine_angles)
        Xg = c_mm[0] + Tg * f_dx[:, None, None] + Wg * (-f_dy[:, None, None])
        Yg = c_mm[1] + Tg * f_dy[:, None, None] + Wg * (f_dx[:, None, None])
        Ug = (Xg * s + TOPDOWN_SIZE / 2.0).astype(np.float32).reshape(M_f, -1)
        Vg = (Yg * s + TOPDOWN_SIZE / 2.0).astype(np.float32).reshape(M_f, -1)

        f_vals = cv2.remap(tophat9_angle, Ug, Vg, cv2.INTER_LINEAR).reshape(M_f, len(ts_coarse), len(ws_search))
        C_f = f_vals[:, :, [2, 3, 4]].max(axis=2)
        F_f = f_vals[:, :, [0, 1, 5, 6]].mean(axis=2)
        S_f = np.maximum(0.0, C_f - 0.85 * F_f)
        f_scores = S_f.mean(axis=1)

        opt_f_idx = int(np.argmax(f_scores))
        opt_ang = float(fine_angles[opt_f_idx])
        fine_best_score = float(f_scores[opt_f_idx])
        if fine_best_score < 20.0:
            continue

        win_rad = math.radians(opt_ang)
        w_dx, w_dy = math.cos(win_rad), math.sin(win_rad)
        w_nx, w_ny = -w_dy, w_dx

        # Snap to barrel ridge center laterally within [-2.5, 2.5] mm
        ws_lat = np.linspace(-2.5, 2.5, 41, dtype=np.float32)
        t_samples = np.linspace(25.0, 45.0, 11, dtype=np.float32)
        Ug_lat = ((c_mm[0] + t_samples[:, None] * w_dx + ws_lat[None, :] * w_nx) * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
        Vg_lat = ((c_mm[1] + t_samples[:, None] * w_dy + ws_lat[None, :] * w_ny) * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
        vals_lat = cv2.remap(tophat5, Ug_lat, Vg_lat, cv2.INTER_LINEAR)
        means = vals_lat.mean(axis=0)
        plateau = ws_lat[means >= 0.90 * np.max(means)]
        w_lat = float(plateau.mean())
        c_mm_corr = c_mm + w_lat * np.array([w_nx, w_ny])

        S_nw_cur = [None]

        def _profile(c_corr: np.ndarray, ang_deg_: float):
            """Axial barrel profile along (c_corr, ang_deg_); None when no barrel."""
            rad_ = math.radians(ang_deg_)
            w_dx, w_dy = math.cos(rad_), math.sin(rad_)
            w_nx, w_ny = -w_dy, w_dx
            ws_fine = np.array([-10.0, -7.5, -2.0, 0.0, 2.0, 7.5, 10.0], dtype=np.float32)
            X_fine = c_corr[0] + fine_ts[:, None] * w_dx + ws_fine[None, :] * w_nx
            Y_fine = c_corr[1] + fine_ts[:, None] * w_dy + ws_fine[None, :] * w_ny
            U_fine = (X_fine * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
            V_fine = (Y_fine * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
            vals_fine = cv2.remap(thin9_axial, U_fine, V_fine, cv2.INTER_LINEAR)
            C_line = vals_fine[:, [2, 3, 4]].max(axis=1)
            F_line = vals_fine[:, [0, 1, 5, 6]].mean(axis=1)
            S_line = np.maximum(0.0, C_line - 0.85 * F_line)
            S_sm = np.convolve(S_line, np.ones(3) / 3.0, mode="same")
            vals_nw = cv2.remap(thin9_nowire, U_fine, V_fine, cv2.INTER_LINEAR)
            S_nw_line = np.maximum(
                0.0, vals_nw[:, [2, 3, 4]].max(axis=1) - 0.85 * vals_nw[:, [0, 1, 5, 6]].mean(axis=1)
            )
            S_nw_cur[0] = np.convolve(S_nw_line, np.ones(3) / 3.0, mode="same")
            zone = np.where((fine_ts >= 10.0) & (fine_ts <= 65.0) & (S_sm >= 20.0))[0]
            if len(zone) == 0:
                return None
            st = int(zone[0])
            pk = st + int(np.argmax(S_sm[st:min(len(S_sm), st + 100)]))
            pv = float(S_sm[pk])
            if pv < 25.0 or fine_ts[pk] > 85.0:
                return None
            th = max(14.0, 0.28 * pv)
            dr = None
            for k in range(pk + 8, len(fine_ts) - 8):
                if np.all(S_sm[k : k + 8] < th):
                    dr = k
                    break
            return S_sm, st, pk, pv, dr

        c_mm_corr = c_mm_corr.copy()
        prof = _profile(c_mm_corr, opt_ang)
        if prof is None:
            continue
        # Axis refinement: the flight-centred angle search can be several degrees off the metal barrel, which
        # makes the profile miss (or truncate) the barrel far from the flight. Fit the barrel ridge's lateral
        # offset against the axial coordinate and re-aim the axis along it.
        for _ in range(2):
            S_smooth, start_idx, peak_idx, peak_val, drop_idx = prof
            t_lo = float(fine_ts[start_idx])
            t_hi = float(fine_ts[drop_idx]) if drop_idx is not None else float(fine_ts[peak_idx]) + 30.0
            refined = _refine_axis(thin9_axial, c_mm_corr, opt_ang, t_lo, t_hi, s)
            if refined is None:
                break
            new_c, new_ang = refined
            new_prof = _profile(new_c, new_ang)
            if new_prof is None:
                break
            c_mm_corr, opt_ang, prof = new_c, new_ang, new_prof
        S_smooth, start_idx, peak_idx, peak_val, drop_idx = prof
        win_rad = math.radians(opt_ang)
        w_dx, w_dy = math.cos(win_rad), math.sin(win_rad)
        w_nx, w_ny = -w_dy, w_dx
        t_peak = fine_ts[peak_idx]

        # Shaft Plateau Bridging: check for forward metallic barrel resurgence across a translucent/plastic shaft gap.
        # Only applies to darts originating in the outer flight zone (c_r > 175.0) where the flight drops early
        # (current_barrel_len < 45.0) in the outer single bed (r_drop > 120.0).
        resurgence_idx = None
        if drop_idx is not None:
            current_barrel_len = fine_ts[drop_idx] - fine_ts[start_idx]
            r_drop = float(np.hypot(c_mm_corr[0] + fine_ts[drop_idx] * w_dx, c_mm_corr[1] + fine_ts[drop_idx] * w_dy))
            if c_r > 175.0 and current_barrel_len < 45.0 and r_drop > 120.0:
                for k in range(drop_idx + 6, min(len(fine_ts) - 16, drop_idx + 60)):
                    if np.all(S_smooth[k : k + 12] >= 22.0):
                        X_res = c_mm_corr[0] + fine_ts[k : k + 12] * w_dx
                        Y_res = c_mm_corr[1] + fine_ts[k : k + 12] * w_dy
                        U_res = (X_res * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
                        V_res = (Y_res * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
                        res_vals = cv2.remap(tophat5_no_wire, U_res, V_res, cv2.INTER_LINEAR)
                        if float(np.mean(res_vals)) >= 8.0:
                            resurgence_idx = k
                            break

                if resurgence_idx is not None:
                    new_peak_idx = resurgence_idx + int(np.argmax(S_smooth[resurgence_idx : min(len(S_smooth), resurgence_idx + 80)]))
                    new_peak_val = float(S_smooth[new_peak_idx])
                    new_thresh = max(14.0, 0.28 * new_peak_val)
                    drop_idx = None
                    for k in range(new_peak_idx + 6, len(fine_ts) - 8):
                        if np.all(S_smooth[k : k + 8] < new_thresh):
                            drop_idx = k
                            break
                    peak_idx = new_peak_idx
                    peak_val = max(peak_val, new_peak_val)

        # Gap bridging: grip -> knurl -> taper -> needle transitions and translucent shafts leave short gaps in
        # the barrel profile. Continue the barrel across gaps when wire-suppressed metal evidence resumes.
        bridged = False
        S_nw = S_nw_cur[0]
        for _ in range(2):
            if drop_idx is None:
                break
            new_drop = _bridge_gap(S_nw, fine_ts, start_idx, drop_idx)
            if new_drop is None:
                break
            drop_idx = new_drop if new_drop >= 0 else None
            bridged = True
            if drop_idx is None:
                break

        if drop_idx is None:
            t_barrel_end = fine_ts[min(peak_idx + 60, len(fine_ts) - 1)]
        elif resurgence_idx is not None or bridged:
            t_barrel_end = fine_ts[drop_idx]
        else:
            w_s = max(peak_idx, drop_idx - 6)
            w_e = min(len(fine_ts), drop_idx + 6)
            grad = np.gradient(S_smooth[w_s:w_e])
            t_barrel_end = fine_ts[w_s + int(np.argmin(grad))]

        barrel_len = t_barrel_end - fine_ts[start_idx]
        if barrel_len < 12.0:
            continue

        # Closest approach to board origin (0, 0) for darts hitting the bullseye
        is_bull_snap = False
        t_closest = -float(c_mm_corr[0] * w_dx + c_mm_corr[1] * w_dy)
        r_closest = float(np.hypot(c_mm_corr[0] + t_closest * w_dx, c_mm_corr[1] + t_closest * w_dy))
        if r_closest <= 3.5 and 50.0 <= t_closest <= 95.0:
            t_barrel_end = t_closest
            is_bull_snap = True

        r_barrel_end = float(np.hypot(c_mm_corr[0] + t_barrel_end * w_dx, c_mm_corr[1] + t_barrel_end * w_dy))

        # Needle extension: trace the steel point beyond the barrel end (needle, taper, or knurl/mesh sections that
        # the barrel profile missed) while wire-suppressed 5x5 specular evidence continues along the axis. Gaps up
        # to NEEDLE_GAP_MM are bridged (wire crossings are zeroed in the evidence map).
        t_tip = float(t_barrel_end)
        if not is_bull_snap:
            ts_needle = t_barrel_end + np.arange(0.0, NEEDLE_MAX_MM + 0.01, 0.5, dtype=np.float32)
            ws_needle = np.array([-4.5, -1.0, 0.0, 1.0, 4.5], dtype=np.float32)
            X_n = c_mm_corr[0] + ts_needle[:, None] * w_dx + ws_needle[None, :] * w_nx
            Y_n = c_mm_corr[1] + ts_needle[:, None] * w_dy + ws_needle[None, :] * w_ny
            U_n = (X_n * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
            V_n = (Y_n * s + TOPDOWN_SIZE / 2.0).astype(np.float32)
            vals_n = cv2.remap(tophat5_no_wire, U_n, V_n, cv2.INTER_LINEAR)
            C_n = vals_n[:, [1, 2, 3]].max(axis=1)
            F_n = vals_n[:, [0, 4]].mean(axis=1)
            S_n = np.maximum(0.0, C_n - 0.85 * F_n)
            good = S_n >= NEEDLE_MIN_EVIDENCE
            last_good = None
            run_bad = 0
            for idx in range(len(ts_needle)):
                if good[idx]:
                    last_good = idx
                    run_bad = 0
                else:
                    run_bad += 1
                    limit = NEEDLE_START_MM if last_good is None else NEEDLE_GAP_MM
                    if run_bad * 0.5 > limit:
                        break
            if last_good is not None:
                t_tip = float(ts_needle[last_good])

        tip_mm = np.array([c_mm_corr[0] + t_tip * w_dx, c_mm_corr[1] + t_tip * w_dy])
        tip_r = float(np.hypot(tip_mm[0], tip_mm[1]))
        if tip_r > 212.0:
            continue

        # Perimeter candidates must travel inward toward the board
        if c_r >= 175.0:
            if c_r - tip_r < 23.5:
                continue

        bwd_mask = (fine_ts <= 10.0) & (fine_ts >= -15.0)
        tail_idx = np.where(bwd_mask)[0]
        t_tail = fine_ts[tail_idx[0]] if len(tail_idx) > 0 else 0.0
        tail_mm = np.array([c_mm_corr[0] + t_tail * w_dx, c_mm_corr[1] + t_tail * w_dy])

        tip_px = project(board.H, [tip_mm])[0]
        tail_px = project(board.H, [tail_mm])[0]
        score, sector, mult, label = score_point_detailed(board.H, tip_px, board.rings)
        conf = 0.95 if is_bull_snap else float(np.clip(peak_val / 150.0, 0.50, 0.99))

        dart = DartDetection(
            tip_px=tip_px,
            tail_px=tail_px,
            tip_mm=tip_mm,
            sector=sector,
            multiplier=mult,
            score=score,
            label=label,
            confidence=conf,
        )
        dart.c_mm = c_mm_corr
        dart.tail_mm = tail_mm
        dart.barrel_mm = tip_mm - 25.0 * np.array([w_dx, w_dy])
        dart.ray_angle = opt_ang
        dart.peak_val = peak_val
        found_candidates.append(dart)

    # Merge candidates that trace the same physical dart (several flight/shaft peaks share one axis) and keep
    # the member that reaches furthest toward the board along that axis.
    found_candidates.sort(key=lambda d: d.confidence, reverse=True)
    groups: list[list[DartDetection]] = []
    for d in found_candidates:
        for grp in groups:
            if _same_axis(d, grp[0]):
                grp.append(d)
                break
        else:
            groups.append([d])
    found_candidates = []
    for grp in groups:
        lead = grp[0]
        l_dir = np.array([math.cos(math.radians(lead.ray_angle)), math.sin(math.radians(lead.ray_angle))])
        best = max(grp, key=lambda m: float(np.dot(m.tip_mm - lead.c_mm, l_dir)))
        best.confidence = max(m.confidence for m in grp)
        found_candidates.append(best)
    found_candidates.sort(key=lambda d: d.confidence, reverse=True)
    kept_darts = []
    for d in found_candidates:
        if any(np.linalg.norm(d.tip_mm - k.tip_mm) < 6.0 for k in kept_darts):
            continue
        if any(np.linalg.norm(d.barrel_mm - k.barrel_mm) < 4.0 for k in kept_darts):
            continue

        is_dup = False
        for k in kept_darts:
            tip_dist = float(np.linalg.norm(d.tip_mm - k.tip_mm))
            if tip_dist < 6.0:
                is_dup = True
                break
            if np.linalg.norm(d.barrel_mm - k.barrel_mm) < 4.0:
                is_dup = True
                break
            tail_dist = float(np.linalg.norm(d.c_mm - k.c_mm))
            if tail_dist < 32.0:
                is_dup = True
                break
            ang_diff = abs((d.ray_angle - k.ray_angle + 180.0) % 360.0 - 180.0)
            if ang_diff < 18.0 and tip_dist < 25.0:
                k_rad = math.radians(k.ray_angle)
                k_dir = np.array([math.cos(k_rad), math.sin(k_rad)])
                p1 = k.c_mm - 30.0 * k_dir
                p2 = k.tip_mm + 15.0 * k_dir
                if pt_to_segment_dist(d.barrel_mm, p1, p2) < 5.0 or pt_to_segment_dist(d.tip_mm, p1, p2) < 5.0:
                    is_dup = True
                    break
        if is_dup:
            continue

        kept_darts.append(d)
        if len(kept_darts) >= max_darts:
            break

    diag = None
    if diagnostics:
        diag = DartDiagnostics(
            topdown_anomaly=anomaly,
            topdown_ridges=thin9_axial,
            topdown_expected=expected_bgr,
        )

    return kept_darts, diag
