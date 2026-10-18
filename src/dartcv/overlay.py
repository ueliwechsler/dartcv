"""Draw the dartboard model on an image and rectify to a top-down view."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import RADII, SECTORS, polar, project

TOPDOWN_SCALE = 4  # output pixels per mm
_MARGIN_MM = 25.0  # room for the number ring


def topdown_size_px() -> int:
    return int(2 * (RADII["double_outer"] + _MARGIN_MM) * TOPDOWN_SCALE)


def model_to_topdown() -> np.ndarray:
    """Matrix mapping model mm -> top-down output pixels."""
    c = topdown_size_px() / 2
    s = TOPDOWN_SCALE
    return np.float64([[s, 0, c], [0, s, c], [0, 0, 1]])


def draw_overlay(img, H, color=(0, 255, 0), thickness=1, numbers=True):
    """Draw rings, wires and sector numbers through homography H (model->image)."""
    out = img.copy()
    angles = np.linspace(0, 360, 361)
    for r in RADII.values():
        poly = project(H, [polar(r, a) for a in angles])
        cv2.polylines(out, [np.round(poly).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    for k in range(20):
        a = 9 + 18 * k
        p = project(H, [polar(RADII["outer_bull"], a), polar(RADII["double_outer"], a)])
        p1, p2 = (tuple(int(round(v)) for v in q) for q in p)
        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
    if numbers:
        for i, n in enumerate(SECTORS):
            x, y = project(H, [polar(RADII["double_outer"] + 13, i * 18)])[0]
            text = str(n)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(out, text, (int(x - w / 2), int(y + h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def warp_topdown(img, H):
    """Rectify the image so the board appears frontal (20 at the top)."""
    size = topdown_size_px()
    M = model_to_topdown() @ np.linalg.inv(H)  # image -> top-down
    return cv2.warpPerspective(img, M, (size, size), flags=cv2.INTER_LINEAR)


def topdown_overlay(img, H, **kw):
    """Top-down image with the model drawn on it."""
    M = model_to_topdown()
    return draw_overlay(warp_topdown(img, H), M, **kw)
