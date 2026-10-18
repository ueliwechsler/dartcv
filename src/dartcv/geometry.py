"""Dartboard model geometry (mm) and homography helpers.

Model coordinates: origin at board centre, x right, y down (image-like),
angles measured clockwise from the top (the centre of the 20 sector).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Standard (BDO/WDF) radii in mm.
RADII = {
    "bull": 6.35,
    "outer_bull": 15.9,
    "treble_inner": 99.0,
    "treble_outer": 107.0,
    "double_inner": 162.0,
    "double_outer": 170.0,
}

# Wire/outer-double intersections used for calibration, in the order the user
# must click them: 20|1, 6|10, 3|19, 11|14 (clockwise from top, 90 deg apart).
CALIB_ANGLES = (9.0, 99.0, 189.0, 279.0)
CALIB_NAMES = ("20|1", "6|10", "3|19", "11|14")


def polar(radius_mm: float, angle_deg: float) -> tuple[float, float]:
    """Model point (mm) for a radius and a clockwise-from-top angle."""
    a = math.radians(angle_deg)
    return radius_mm * math.sin(a), -radius_mm * math.cos(a)


def calibration_model_points() -> np.ndarray:
    return np.float32([polar(RADII["double_outer"], a) for a in CALIB_ANGLES])


def compute_homography(image_pts) -> np.ndarray:
    """Homography mapping model (mm) -> image pixels from 4+ clicked points.

    The first four points must be in CALIB_ANGLES order; extra points are not
    supported because they would need their own model coordinates.
    """
    pts = np.float32(image_pts)
    if pts.shape != (4, 2):
        raise ValueError("Exactly 4 image points (20|1, 6|10, 3|19, 11|14) required.")
    return cv2.getPerspectiveTransform(calibration_model_points(), pts)


def project(H: np.ndarray, pts) -> np.ndarray:
    """Apply a homography to Nx2 points."""
    p = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def contour_radius(H: np.ndarray, contour: np.ndarray, angles):
    """Radius of an image-space ring contour along model-space ray angles."""
    points = project(np.linalg.inv(H), contour)
    bearings = np.degrees(np.arctan2(points[:, 0], -points[:, 1])) % 360
    return np.interp(angles, bearings, np.linalg.norm(points, axis=1), period=360)


def score_from_model_point(x_mm: float, y_mm: float) -> int:
    """Score of a point in model (mm) coordinates."""
    r = math.hypot(x_mm, y_mm)
    if r <= RADII["bull"]:
        return 50
    if r <= RADII["outer_bull"]:
        return 25
    if r > RADII["double_outer"]:
        return 0
    angle = (math.degrees(math.atan2(x_mm, -y_mm)) + 9.0) % 360.0
    n = SECTORS[int(angle // 18) % 20]
    if RADII["treble_inner"] < r <= RADII["treble_outer"]:
        return 3 * n
    if r > RADII["double_inner"]:
        return 2 * n
    return n


def score_from_image_point(H: np.ndarray, xy) -> int:
    """Score of a pixel (e.g. a dart tip) given the model->image homography."""
    x, y = project(np.linalg.inv(H), [xy])[0]
    return score_from_model_point(float(x), float(y))
