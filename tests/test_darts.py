import os
import cv2
import numpy as np
import pytest

from test_homography import render_frontal
from dartcv.detect import Detection, analyze_image
from dartcv.darts import detect_darts
from dartcv.geometry import polar, project


def add_synthetic_dart(
    img: np.ndarray,
    m2i: np.ndarray,
    tip_mm: tuple[float, float],
    tail_mm: tuple[float, float],
    flight_color: tuple[int, int, int] = (240, 240, 240),
) -> None:
    """Render a realistic dart onto a synthetic board canvas."""
    tip_px = project(m2i, [tip_mm])[0]
    tail_px = project(m2i, [tail_mm])[0]
    vec = tail_px - tip_px
    L = np.linalg.norm(vec)
    u = vec / L
    n = np.array([-u[1], u[0]])

    # Cast shadow slightly offset
    s_tip = tip_px + np.array([10.0, 15.0])
    s_tail = tail_px + np.array([30.0, 45.0])
    cv2.line(img, tuple(s_tip.astype(int)), tuple(s_tail.astype(int)), (20, 20, 20), 16)

    # Steel needle (first 18% of length from tip)
    p_barrel_start = tip_px + u * (L * 0.18)
    cv2.line(img, tuple(tip_px.astype(int)), tuple(p_barrel_start.astype(int)), (170, 170, 170), 8)

    # Metallic barrel (next 42% of length)
    p_barrel_end = tip_px + u * (L * 0.60)
    cv2.line(img, tuple(p_barrel_start.astype(int)), tuple(p_barrel_end.astype(int)), (200, 205, 210), 20)

    # Shaft (next 15% of length)
    p_flight_start = tip_px + u * (L * 0.75)
    cv2.line(img, tuple(p_barrel_end.astype(int)), tuple(p_flight_start.astype(int)), (30, 30, 30), 10)

    # Flight (last 25% of length, wide fin)
    flight_w = 45.0
    f_p1 = p_flight_start + n * (flight_w * 0.5)
    f_p2 = tail_px + n * flight_w
    f_p3 = tail_px - n * flight_w
    f_p4 = p_flight_start - n * (flight_w * 0.5)
    pts = np.array([f_p1, f_p2, f_p3, f_p4], dtype=np.int32)
    cv2.fillPoly(img, [pts], flight_color)


def test_zero_darts_on_clean_board():
    board_img, m2i = render_frontal()
    det_board = Detection(
        H=m2i,
        H_initial=m2i,
        ellipse=((780, 780), (780, 780), 0),
        rotation_deg=0.0,
        ecc_score=1.0,
    )
    darts, diag = detect_darts(board_img, det_board, diagnostics=True)
    assert len(darts) == 0
    assert diag is not None
    assert diag.topdown_anomaly is not None


def test_single_dart_treble_20():
    board_img, m2i = render_frontal()
    det_board = Detection(
        H=m2i,
        H_initial=m2i,
        ellipse=((780, 780), (780, 780), 0),
        rotation_deg=0.0,
        ecc_score=1.0,
    )
    tip_true = polar(103.0, 0.0)  # T20 centre (r=103 mm, ang=0 deg)
    tail_true = polar(175.0, 12.0)
    add_synthetic_dart(board_img, m2i, tip_true, tail_true, flight_color=(0, 0, 220))

    darts, _ = detect_darts(board_img, det_board, max_darts=1)
    assert len(darts) == 1
    d = darts[0]
    assert d.label == "T20"
    assert d.score == 60
    assert d.multiplier == 3
    assert d.sector == 20
    err = np.linalg.norm(d.tip_mm - np.array(tip_true))
    assert err < 2.5


@pytest.mark.parametrize("flight_color", [(240, 240, 240), (20, 20, 20), (0, 0, 220)])
def test_single_dart_flight_colors(flight_color):
    board_img, m2i = render_frontal()
    det_board = Detection(
        H=m2i,
        H_initial=m2i,
        ellipse=((780, 780), (780, 780), 0),
        rotation_deg=0.0,
        ecc_score=1.0,
    )
    tip_true = polar(103.0, 0.0)
    tail_true = polar(175.0, 12.0)
    add_synthetic_dart(board_img, m2i, tip_true, tail_true, flight_color=flight_color)

    darts, _ = detect_darts(board_img, det_board, max_darts=1)
    assert len(darts) == 1
    d = darts[0]
    assert d.score == 60
    assert d.label == "T20"
    err = np.linalg.norm(d.tip_mm - np.array(tip_true))
    assert err < 2.5


def test_multi_darts_two_and_three():
    board_img, m2i = render_frontal()
    det_board = Detection(
        H=m2i,
        H_initial=m2i,
        ellipse=((780, 780), (780, 780), 0),
        rotation_deg=0.0,
        ecc_score=1.0,
    )
    # Dart 1 in T20 (score 60)
    tip1 = polar(103.0, 0.0)
    tail1 = polar(175.0, 12.0)
    add_synthetic_dart(board_img, m2i, tip1, tail1, flight_color=(0, 0, 220))

    # Dart 2 in D16 (score 32, ang=234 deg)
    tip2 = polar(166.0, 234.0)
    tail2 = polar(195.0, 250.0)
    add_synthetic_dart(board_img, m2i, tip2, tail2, flight_color=(230, 230, 230))

    darts, _ = detect_darts(board_img, det_board, max_darts=2)
    assert len(darts) == 2
    total = sum(d.score for d in darts)
    assert total == 92


def test_dart_miss_outside_double():
    board_img, m2i = render_frontal()
    det_board = Detection(
        H=m2i,
        H_initial=m2i,
        ellipse=((780, 780), (780, 780), 0),
        rotation_deg=0.0,
        ecc_score=1.0,
    )
    # Dart landing in catch ring outside double wire (r=180 mm)
    tip_miss = polar(180.0, 54.0)
    tail_miss = polar(205.0, 65.0)
    add_synthetic_dart(board_img, m2i, tip_miss, tail_miss, flight_color=(0, 200, 200))

    darts, _ = detect_darts(board_img, det_board, max_darts=1)
    if len(darts) > 0:
        d = darts[0]
        assert d.label == "MISS"
        assert d.score == 0


def test_real_single_dart_image():
    img_path = "data/images/single_dart/PXL_20261018_152634348.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"Image {img_path} not found")

    img = cv2.imread(img_path)
    res = analyze_image(img, upright_deg=0.0, refine=True, diagnostics=True, max_darts=1)
    assert res.board is not None
    assert len(res.darts) == 1
    d = res.darts[0]
    assert d.label == "S4"
    assert d.score == 4
    assert d.sector == 4
    assert d.multiplier == 1
    assert "board_ms" in res.timings_ms
    assert "darts_ms" in res.timings_ms
    assert "total_ms" in res.timings_ms
    assert res.total_score == 4
