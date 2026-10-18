import cv2
import numpy as np
import pytest

from dartcv.geometry import (
    CALIB_ANGLES,
    RADII,
    SECTORS,
    compute_homography,
    polar,
    project,
    score_from_image_point,
    score_from_model_point,
)
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
    ring = np.where(even[..., None], (40, 40, 210), (60, 160, 40)).astype(
        np.uint8
    )  # BGR
    img[r <= RADII["double_outer"]] = base[r <= RADII["double_outer"]]
    for lo, hi in (
        (RADII["treble_inner"], RADII["treble_outer"]),
        (RADII["double_inner"], RADII["double_outer"]),
    ):
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
        np.float32([[120, 90], [700, 40], [760, 640], [60, 560]]),
    )
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
    expected = cv2.warpPerspective(
        board,
        model_to_topdown() @ np.linalg.inv(m2i),
        (size, size),
        borderValue=(128, 128, 128),
    )
    inside = np.zeros((size, size), bool)
    cv2.circle(
        inside.view(np.uint8),
        (size // 2, size // 2),
        int((RADII["double_outer"] - 3) * 4),
        1,
        -1,
    )
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


def test_cli_defaults_to_raw_board_directory(tmp_path, monkeypatch):
    from dartcv.cli import main

    monkeypatch.chdir(tmp_path)
    image_dir = tmp_path / "data" / "images" / "raw_board"
    image_dir.mkdir(parents=True)
    img, *_ = synthetic_view()
    assert cv2.imwrite(str(image_dir / "board.jpg"), img)

    output_dir = tmp_path / "out"
    main(["--out", str(output_dir)])

    assert {path.name for path in (output_dir / "board").iterdir()} == {
        "overlay.png",
        "topdown.png",
        "topdown_overlay.png",
        "01_colour_mask.png",
        "02_outer_ellipse.png",
        "03_initial_alignment.png",
        "04_ecc_edges.png",
        "05_calibration.png",
        "06_bull_alignment.png",
        "07_ring_alignment.png",
    }
    for path in (output_dir / "board").iterdir():
        assert cv2.imread(str(path)) is not None


def test_diagnostic_overlays_preserve_source_and_mark_calibration_wires():
    from dartcv.detect import detect_board
    from dartcv.diagnostics import (
        CALIB_COLOURS,
        calibration_overlay,
        detection_overlays,
    )

    img, H, _, _ = synthetic_view()
    original = img.copy()
    det = detect_board(img, diagnostics=True)
    outputs = detection_overlays(img, det)
    assert len(outputs) == 7
    assert det.ecc_score is not None
    for name, output in outputs.items():
        if name == "04_ecc_edges.png":
            expected = (topdown_size_px(), topdown_size_px(), 3)
        elif name == "06_bull_alignment.png":
            expected = (805, 1440, 3)
        elif name == "07_ring_alignment.png":
            expected = (1610, 1440, 3)
        else:
            expected = img.shape
        assert output.shape == expected
        assert output.dtype == np.uint8
    marked = calibration_overlay(img, H)
    for angle, colour in zip(CALIB_ANGLES, CALIB_COLOURS):
        x, y = np.round(project(H, [polar(RADII["double_outer"], angle)])[0]).astype(
            int
        )
        np.testing.assert_array_equal(marked[y, x], colour)
    np.testing.assert_array_equal(img, original)


def test_diagnostics_keep_original_image_coordinates_after_downscaling():
    from dartcv.detect import detect_board
    from dartcv.diagnostics import detection_overlays

    img, *_ = synthetic_view()
    img = cv2.resize(img, None, fx=2, fy=2)
    det = detect_board(img, refine=False, diagnostics=True)
    debug = det.diagnostics
    assert debug is not None
    assert max(debug.colour_mask.shape) == 1000
    assert debug.boundary_points.shape[0] == debug.ellipse_inliers.shape[0]
    scale = debug.colour_mask.shape[1] / img.shape[1]
    points = np.round(debug.boundary_points * scale).astype(int)
    assert np.all(debug.colour_mask[points[:, 1], points[:, 0]] == 255)
    assert detection_overlays(img, det)["02_outer_ellipse.png"].shape == img.shape


@pytest.mark.parametrize("fails", [False, True])
def test_diagnostic_ecc_inputs_survive_failed_or_rejected_refinement(
    monkeypatch, fails
):
    from dartcv.detect import _edge_map, detect_board
    from dartcv.diagnostics import detection_overlays

    def fake_ecc(template, image, warp, *args):
        mask = args[-2]
        assert mask.shape == image.shape
        assert mask[0, 0] == 0
        assert mask[mask.shape[0] // 2, mask.shape[1] // 2] == 255
        if fails:
            raise cv2.error("forced ECC failure")
        return 0.1, warp  # below the acceptance threshold

    monkeypatch.setattr(cv2, "findTransformECC", fake_ecc)
    img, *_ = synthetic_view()
    det = detect_board(img, diagnostics=True)
    assert det.ecc_score is None
    debug = det.diagnostics
    assert debug is not None
    np.testing.assert_array_equal(debug.H_before_rings, det.H_initial)
    # Colour-ring refinement remains available even when ECC falls back.
    assert len(det.rings) == 4
    np.testing.assert_allclose(
        project(det.H, [[0, 0]]), project(det.H_initial, [[0, 0]]), atol=1e-4
    )
    gray = cv2.cvtColor(warp_topdown(img, det.H_initial), cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    np.testing.assert_array_equal(
        debug.ecc_edges, _edge_map(gray, 0.125 if fails else 0.5)
    )
    assert "04_ecc_edges.png" in detection_overlays(img, det)


def test_cli_supplied_points_produces_calibration_overlay(tmp_path):
    from dartcv.cli import main

    img, H, _, _ = synthetic_view()
    source = tmp_path / "board.png"
    assert cv2.imwrite(str(source), img)
    points = project(H, [polar(RADII["double_outer"], a) for a in CALIB_ANGLES])
    output_dir = tmp_path / "out"
    main([str(source), "--out", str(output_dir), "--points", *map(str, points.ravel())])
    assert {path.name for path in output_dir.iterdir()} == {
        "overlay.png",
        "topdown.png",
        "topdown_overlay.png",
        "05_calibration.png",
    }


def test_diagnostic_capture_is_optional():
    from dartcv.detect import detect_board
    from dartcv.diagnostics import detection_overlays

    img, *_ = synthetic_view()
    det = detect_board(img, refine=False)
    assert det.diagnostics is None
    with pytest.raises(ValueError, match="diagnostics=True"):
        detection_overlays(img, det)


def test_bull_measurement_is_independent_of_predicted_centre():
    from dartcv.detect import _detect_bull

    img, H, _, _ = synthetic_view()
    shifted = np.array([[1, 0, 12], [0, 1, -8], [0, 0, 1]]) @ H
    bull = _detect_bull(img, shifted)
    assert bull is not None
    assert np.linalg.norm(bull.centre - project(H, [[0, 0]])[0]) < 1.0
    assert np.linalg.norm(bull.centre - project(shifted, [[0, 0]])[0]) > 10


def test_bull_anchor_preserves_outer_double_conic():
    from dartcv.detect import _anchor_bull

    _, H, _, _ = synthetic_view()
    target = project(H, [[18, -12]])[0]
    anchored = _anchor_bull(H, target)
    np.testing.assert_allclose(project(anchored, [[0, 0]])[0], target, atol=1e-4)
    circle = [polar(RADII["double_outer"], a) for a in range(0, 360, 5)]
    mapped_back = project(np.linalg.inv(H), project(anchored, circle))
    np.testing.assert_allclose(
        np.linalg.norm(mapped_back, axis=1), RADII["double_outer"], atol=1e-3
    )
    # A translation would move this conic; an interior projective correction must not.
    assert np.linalg.norm(anchored[2, :2] - H[2, :2] / H[2, 2]) > 1e-6
    with pytest.raises(ValueError, match="too far"):
        _anchor_bull(H, project(H, [[160, 0]])[0])


def test_bull_constraint_survives_ecc_failure_under_strong_perspective(monkeypatch):
    from dartcv.detect import detect_board

    board, m2i = render_frontal()
    n = board.shape[0]
    warp = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [n, 0], [n, n], [0, n]]),
        np.float32([[240, 90], [600, 100], [790, 650], [40, 590]]),
    )
    img = cv2.warpPerspective(board, warp, (840, 720), borderValue=(128, 128, 128))
    truth = project(warp @ m2i, [[0, 0]])[0]

    def fail_ecc(*args):
        raise cv2.error("forced ECC failure")

    monkeypatch.setattr(cv2, "findTransformECC", fail_ecc)
    det = detect_board(img, diagnostics=True)
    assert det.bull is not None
    assert det.ecc_score is None
    assert np.linalg.norm(np.array(det.ellipse[0]) - truth) > 10
    assert np.linalg.norm(det.bull.centre - truth) < 1.0
    assert np.linalg.norm(project(det.H, [[0, 0]])[0] - truth) < 1.0


def test_bull_rejects_red_without_green_surround():
    from dartcv.detect import _detect_bull

    img, H, _, _ = synthetic_view()
    img[:] = 128
    centre = tuple(np.round(project(H, [[0, 0]])[0]).astype(int))
    cv2.circle(img, centre, 12, (0, 0, 200), -1)
    assert _detect_bull(img, H) is None


def test_occluded_bull_falls_back_without_inventing_measurement():
    from dartcv.detect import detect_board
    from dartcv.diagnostics import bull_alignment_overlay

    img, H, _, _ = synthetic_view()
    centre = tuple(np.round(project(H, [[0, 0]])[0]).astype(int))
    cv2.circle(img, centre, 45, (30, 30, 30), -1)
    det = detect_board(img, diagnostics=True)
    assert det.bull is None
    assert np.isfinite(det.H).all()
    assert bull_alignment_overlay(img, det).shape == img.shape
