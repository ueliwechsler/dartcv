import cv2
import numpy as np
from test_homography import render_frontal, synthetic_view

from dartcv.geometry import RADII, contour_radius, project
from dartcv.overlay import draw_overlay, model_to_topdown, topdown_overlay
from dartcv.rings import _fit_boundary, refine_coloured_rings


def coloured_bands_with_metal():
    """Known paint dimensions deliberately differ from the nominal wire radii."""
    img, H = render_frontal()
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    dx, dy = (xx - img.shape[1] / 2) / 4, (yy - img.shape[0] / 2) / 4
    radii = np.hypot(dx, dy)
    even = (((np.degrees(np.arctan2(dx, -dy)) + 9) % 360) // 18).astype(int) % 2 == 0
    colours = np.where(even[..., None], (40, 40, 210), (60, 160, 40)).astype(np.uint8)
    truth = {
        "treble_inner": 100.5,
        "treble_outer": 106.5,
        "double_inner": 163.8,
        "double_outer": 171.2,
    }
    for band in ("treble", "double"):
        inner, outer = f"{band}_inner", f"{band}_outer"
        grey = (radii > RADII[inner] - 3) & (radii < RADII[outer] + 4)
        img[grey] = (190, 190, 190)
        paint = (radii > truth[inner]) & (radii <= truth[outer])
        img[paint] = colours[paint]
    return img, H, truth


def test_ring_geometry_recovers_scale_without_displacing_bull():
    img, truth, _, _ = synthetic_view()
    initial = truth @ np.diag([0.96, 0.96, 1])
    H, rings = refine_coloured_rings(img, initial)
    assert len(rings) == 4
    np.testing.assert_allclose(
        project(H, [[0, 0]]), project(truth, [[0, 0]]), atol=1e-4
    )
    for name, ring in rings.items():
        measured_radii = np.linalg.norm(
            project(np.linalg.inv(truth), ring.contour), axis=1
        )
        assert np.median(np.abs(measured_radii - RADII[name])) < 0.4
        assert len(ring.samples) >= 60
        assert ring.median_error_px < 1.0


def test_measured_paint_edges_exclude_surrounding_metal_and_allow_actual_widths():
    img, H, truth = coloured_bands_with_metal()
    original = img.copy()
    refined, rings = refine_coloured_rings(img, H)
    assert len(rings) == 4
    for name, ring in rings.items():
        radii = contour_radius(H, ring.contour, np.arange(360))
        assert np.max(np.abs(radii - truth[name])) < 0.3
        assert ring.median_error_px < 1.0
    np.testing.assert_array_equal(img, original)
    np.testing.assert_allclose(
        project(refined, [[0, 0]]), project(H, [[0, 0]]), atol=1e-4
    )
    assert (
        abs(
            np.median(contour_radius(H, rings["treble_outer"].contour, np.arange(360)))
            - np.median(
                contour_radius(H, rings["treble_inner"].contour, np.arange(360))
            )
            - 6
        )
        < 0.3
    )


def test_ring_fit_uses_distributed_support_and_survives_local_occlusion():
    img, H, truth = coloured_bands_with_metal()
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    angle = (
        np.degrees(np.arctan2(xx - img.shape[1] / 2, -(yy - img.shape[0] / 2))) % 360
    )
    img[(angle > 20) & (angle < 55)] = 120
    _, rings = refine_coloured_rings(img, H)
    assert len(rings) == 4
    for name, ring in rings.items():
        assert (
            np.median(
                np.abs(contour_radius(H, ring.contour, np.arange(360)) - truth[name])
            )
            < 0.3
        )
    # A small remaining arc cannot independently establish an entire ellipse.
    img[(angle > 55) & (angle < 340)] = 120
    refined, rings = refine_coloured_rings(img, H)
    assert rings == {}
    np.testing.assert_array_equal(refined, H)


def test_missing_colour_does_not_snap_to_grey_contrast_edges():
    img, H, _ = coloured_bands_with_metal()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    monochrome = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    refined, rings = refine_coloured_rings(monochrome, H)
    assert rings == {}
    np.testing.assert_array_equal(refined, H)


def test_one_missing_band_keeps_other_measurements_without_inventing_edges():
    img, H, truth = coloured_bands_with_metal()
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    radius = np.hypot(xx - img.shape[1] / 2, yy - img.shape[0] / 2) / 4
    img[(radius > 90) & (radius < 115)] = 128
    refined, rings = refine_coloured_rings(img, H)
    assert set(rings) == {"double_inner", "double_outer"}
    np.testing.assert_array_equal(refined, H)
    for name, ring in rings.items():
        assert (
            np.max(
                np.abs(contour_radius(H, ring.contour, np.arange(360)) - truth[name])
            )
            < 0.3
        )


def test_robust_boundary_fit_rejects_colour_outliers():
    H = model_to_topdown()
    angles = np.linspace(0, 2 * np.pi, 260, endpoint=False)
    radius = np.full(260, 107.0)
    radius[::9] += 5
    points = project(
        H, np.column_stack([radius * np.sin(angles), -radius * np.cos(angles)])
    )
    ring = _fit_boundary(points, H, 107)
    assert ring is not None
    assert not ring.inliers[::9].any()
    np.testing.assert_allclose(
        contour_radius(H, ring.contour, np.arange(360)), 107, atol=0.01
    )


def test_overlay_uses_measured_contours_in_both_coordinate_frames():
    img, H, _, _ = synthetic_view()
    _, rings = refine_coloured_rings(img, H)
    contours = {name: ring.contour for name, ring in rings.items()}
    overlay = draw_overlay(img, H, ring_contours=contours, numbers=False, thickness=3)
    topdown = topdown_overlay(
        img, H, ring_contours=contours, numbers=False, thickness=3
    )
    M = model_to_topdown() @ np.linalg.inv(H)
    for ring in rings.values():
        for contour, view in (
            (ring.contour, overlay),
            (project(M, ring.contour), topdown),
        ):
            x, y = np.round(contour[123]).astype(int)
            assert view[y, x, 1] > 220
