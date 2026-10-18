import cv2
import numpy as np

from dartcv.geometry import (CALIB_ANGLES, RADII, SECTORS, compute_homography, polar,
                             project, score_from_image_point, score_from_model_point)
from dartcv.overlay import draw_overlay, model_to_topdown, topdown_size_px, warp_topdown

S = 4.0  # px per mm for the synthetic frontal render


def render_frontal(size=topdown_size_px()):
    """Correctly oriented board (20 centred at top) in a size x size image."""
    img = np.full((size, size, 3), 128, np.uint8)
    m2i = np.float64([[S, 0, size / 2], [0, S, size / 2], [0, 0, 1]])
    ys, xs = np.mgrid[0:size, 0:size]
    dx, dy = (xs - size / 2) / S, (ys - size / 2) / S
    r = np.hypot(dx, dy)
    ang = (np.degrees(np.arctan2(dx, -dy)) + 9) % 360
    idx = (ang // 18).astype(int) % 20
    even = idx % 2 == 0
    base = np.where(even[..., None], (20, 20, 20), (235, 235, 235)).astype(np.uint8)
    ring = np.where(even[..., None], (40, 40, 210), (60, 160, 40)).astype(np.uint8)  # BGR
    img[r <= RADII["double_outer"]] = base[r <= RADII["double_outer"]]
    for lo, hi in ((RADII["treble_inner"], RADII["treble_outer"]),
                   (RADII["double_inner"], RADII["double_outer"])):
        m = (r > lo) & (r <= hi)
        img[m] = ring[m]
    img[r <= RADII["outer_bull"]] = (60, 160, 40)
    img[r <= RADII["bull"]] = (40, 40, 210)
    return img, m2i


def synthetic_view():
    board, m2i = render_frontal()
    n = board.shape[0]
    persp = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [n, 0], [n, n], [0, n]]),
        np.float32([[120, 90], [700, 40], [760, 640], [60, 560]]))
    canvas = cv2.warpPerspective(board, persp, (820, 700), borderValue=(128, 128, 128))
    return canvas, persp @ m2i, board, m2i


def test_recovers_homography_from_four_points():
    _, H_true, _, _ = synthetic_view()
    pts = project(H_true, [polar(RADII["double_outer"], a) for a in CALIB_ANGLES])
    H = compute_homography(pts)
    probe = [polar(r, a) for r in (10, 100, 165) for a in (0, 45, 200)]
    assert np.abs(project(H, probe) - project(H_true, probe)).max() < 1e-2


def test_topdown_matches_frontal_render():
    img, H_true, board, m2i = synthetic_view()
    top = warp_topdown(img, H_true)
    # Re-render expected top-down by mapping the frontal board into the output frame.
    size = topdown_size_px()
    expected = cv2.warpPerspective(board, model_to_topdown() @ np.linalg.inv(m2i), (size, size),
                                   borderValue=(128, 128, 128))
    inside = np.zeros((size, size), bool)
    cv2.circle(inside.view(np.uint8), (size // 2, size // 2),
               int((RADII["double_outer"] - 3) * 4), 1, -1)
    diff = np.abs(top.astype(int) - expected.astype(int)).max(axis=2)
    assert (diff[inside] > 60).mean() < 0.05  # only wire-edge pixels may differ


def test_overlay_runs():
    img, H, _, _ = synthetic_view()
    assert draw_overlay(img, H).shape == img.shape


def test_scoring():
    assert score_from_model_point(0, 0) == 50
    assert score_from_model_point(10, 0) == 25
    assert score_from_model_point(*polar(103, 0)) == 60
    assert score_from_model_point(*polar(166, 0)) == 40
    assert score_from_model_point(*polar(130, 18)) == 1
    assert score_from_model_point(*polar(130, 90)) == SECTORS[5]
    assert score_from_model_point(*polar(200, 0)) == 0
    _, H, _, _ = synthetic_view()
    assert score_from_image_point(H, project(H, [polar(103, 0)])[0]) == 60


def test_auto_detection_recovers_homography():
    from dartcv.detect import detect_board
    img, H_true, _, _ = synthetic_view()
    det = detect_board(img)
    probe = [polar(r, a) for r in (50, 100, 165) for a in (0, 90, 200, 300)]
    err = np.linalg.norm(project(det.H, probe) - project(H_true, probe), axis=1).max()
    assert err < 3.0  # pixels


def test_auto_detection_with_tilted_camera():
    from dartcv.detect import detect_board
    img, H_true, _, _ = synthetic_view()
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
    R3 = np.vstack([R, [0, 0, 1]])
    rot = cv2.warpAffine(img, R, (w, h), borderValue=(128, 128, 128))
    det = detect_board(rot, upright_deg=-12)
    H_t = R3 @ H_true
    probe = [polar(r, a) for r in (50, 100, 165) for a in (0, 90, 200, 300)]
    err = np.linalg.norm(project(det.H, probe) - project(H_t, probe), axis=1).max()
    assert err < 3.0
