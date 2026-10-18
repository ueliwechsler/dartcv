"""2D dartboard rendering utilities for CV prototyping.

This module generates a clean synthetic dartboard that can be used as a
reference template for board detection and later scoring logic.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
except Exception:  # pragma: no cover
    plt = None
    Circle = None
    Wedge = None


STANDARD_DARTBOARD_ORDER = [
    20,
    1,
    18,
    4,
    13,
    6,
    10,
    15,
    2,
    17,
    3,
    19,
    7,
    16,
    8,
    11,
    14,
    9,
    12,
    5,
]


LABEL_RING_OUTER = 1.25
LABEL_RADIUS = 1.12


def polar_to_cart(
    cx: float, cy: float, radius: float, angle_deg: float
) -> tuple[float, float]:
    """Convert polar coordinates to cartesian coordinates."""
    theta = math.radians(angle_deg)
    return cx + radius * math.cos(theta), cy + radius * math.sin(theta)


def _ring_bounds() -> dict:
    """Return the standard radial bands of a proper dartboard.

    The board is built as a sequence of radial bands from center outward:
    inner bull -> outer bull -> single -> triple -> single -> double.
    The triple and double rings are narrow and have similar radial thickness.
    """
    return {
        "inner_bull": 0.06,
        "outer_bull": 0.12,
        "single_1_end": 0.58,
        "triple_end": 0.63,
        "single_2_end": 0.95,
        "double_end": 1.0,
    }


def _sector_colors(i: int) -> str:
    # Standard dartboard: 20 is black, then every second sector is black.
    return "#1a1a1a" if i % 2 == 0 else "#f5f5f5"


def _ring_color_for_sector(i: int) -> str:
    # Sector 20 is red, 1 is green, alternating thereafter.
    return "#d32f2f" if i % 2 == 0 else "#1b9e4a"


def _sector_label_position(
    cx: float, cy: float, radius: float, angle_deg: float
) -> tuple[float, float]:
    return polar_to_cart(cx, cy, radius, angle_deg)

def render_svg(width: int = 800, height: int = 800, radius: float | None = None) -> str:
    """Return a proper standard dartboard SVG template."""
    if radius is None:
        radius = min(width, height) * 0.38

    cx = width / 2.0
    cy = height / 2.0
    rings = _ring_bounds()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{radius * LABEL_RING_OUTER}" fill="black"/>',
    ]

    for i in range(20):
        center_angle = 90 - i * 18
        start = center_angle - 9
        end = center_angle + 9
        color = _sector_colors(i)

        # single area from outer bull to triple ring
        parts.append(
            f'<path d="{_wedge_svg_path(cx, cy, radius * rings["outer_bull"], radius * rings["single_1_end"], start, end)}" '
            f'fill="{color}" stroke="black" stroke-width="2"/>'
        )

        # triple ring: small and narrow, same width as double ring
        ring_color = _ring_color_for_sector(i)
        parts.append(
            f'<path d="{_wedge_svg_path(cx, cy, radius * rings["single_1_end"], radius * rings["triple_end"], start, end)}" '
            f'fill="{ring_color}" stroke="black" stroke-width="2"/>'
        )

        # second single area before the double ring
        parts.append(
            f'<path d="{_wedge_svg_path(cx, cy, radius * rings["triple_end"], radius * rings["single_2_end"], start, end)}" '
            f'fill="{color}" stroke="black" stroke-width="2"/>'
        )

        # double ring: small and narrow, same width as triple ring
        parts.append(
            f'<path d="{_wedge_svg_path(cx, cy, radius * rings["single_2_end"], radius * rings["double_end"], start, end)}" '
            f'fill="{ring_color}" stroke="black" stroke-width="2"/>'
        )

    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{radius * rings["outer_bull"]}" fill="#4caf50" stroke="black" stroke-width="3"/>'
    )
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{radius * rings["inner_bull"]}" fill="#d32f2f" stroke="black" stroke-width="3"/>'
    )
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="black" stroke-width="4"/>'
    )

    for i in range(20):
        center_angle = 90 - i * 18
        number = STANDARD_DARTBOARD_ORDER[i]
        lx, ly = _sector_label_position(cx, cy, radius * LABEL_RADIUS, center_angle)
        parts.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="30" font-family="Arial" fill="#ffffff" font-weight="bold">{number}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _wedge_svg_path(
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
    start_deg: float,
    end_deg: float,
) -> str:
    pts = []
    steps = 180
    for step in range(steps + 1):
        theta = start_deg + (end_deg - start_deg) * step / steps
        x, y = polar_to_cart(cx, cy, r_outer, theta)
        pts.append(f"{x:.2f},{y:.2f}")
    for step in range(steps, -1, -1):
        theta = start_deg + (end_deg - start_deg) * step / steps
        x, y = polar_to_cart(cx, cy, r_inner, theta)
        pts.append(f"{x:.2f},{y:.2f}")
    return "M " + " L ".join(pts) + " Z"


def render_dartboard(
    ax=None, figsize: tuple[int, int] = (6, 6), radius: float = 1.0, show: bool = True
):
    """Render a proper standard dartboard on matplotlib axes."""
    if plt is None:
        raise RuntimeError("matplotlib is required for render_dartboard().")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")

    rings = _ring_bounds()
    ax.add_patch(Circle((0, 0), radius * LABEL_RING_OUTER, facecolor="black", edgecolor="black", zorder=0))
    for i in range(20):
        center_angle = 90 - i * 18
        start = center_angle - 9
        end = center_angle + 9
        color = _sector_colors(i)

        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=radius * rings["single_1_end"],
                theta1=start,
                theta2=end,
                width=radius * (rings["single_1_end"] - rings["outer_bull"]),
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        ring_color = _ring_color_for_sector(i)
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=radius * rings["triple_end"],
                theta1=start,
                theta2=end,
                width=radius * (rings["triple_end"] - rings["single_1_end"]),
                facecolor=ring_color,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=radius * rings["single_2_end"],
                theta1=start,
                theta2=end,
                width=radius * (rings["single_2_end"] - rings["triple_end"]),
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=radius * rings["double_end"],
                theta1=start,
                theta2=end,
                width=radius * (rings["double_end"] - rings["single_2_end"]),
                facecolor=ring_color,
                edgecolor="black",
                linewidth=1.0,
            )
        )

    for i in range(20):
        center_angle = 90 - i * 18
        number = STANDARD_DARTBOARD_ORDER[i]
        x, y = _sector_label_position(0, 0, radius * LABEL_RADIUS, center_angle)
        ax.text(x, y, str(number), ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=10)

    ax.add_patch(
        Circle(
            (0, 0),
            radius * rings["outer_bull"],
            facecolor="#4caf50",
            edgecolor="black",
            linewidth=2,
        )
    )
    ax.add_patch(
        Circle(
            (0, 0),
            radius * rings["inner_bull"],
            facecolor="#d32f2f",
            edgecolor="black",
            linewidth=2,
        )
    )
    ax.add_patch(
        Circle((0, 0), radius, facecolor="none", edgecolor="black", linewidth=2)
    )

    if show:
        plt.show()

    return fig, ax


def point_to_score(x: float, y: float) -> int:
    """Convert a point relative to the board center into a rough dart score.

    This is a simple geometric approximation for a normalized board. It is useful
    as a first prototype for later image-based scoring logic.
    """
    r = math.hypot(x, y)
    angle_deg = math.degrees(math.atan2(y, x))
    if angle_deg < 0:
        angle_deg += 360

    # Normalize angle to a sector index in [0, 20)
    sector = int((angle_deg + 9) // 18) % 20
    # Standard dartboard numbering around the clock: 20 at 0°, 1 at 18°, ...
    score_map = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    sector_value = score_map[sector]

    rings = _ring_bounds()
    if r <= radius_value(rings["inner_bull"]):
        return 50
    if r <= radius_value(rings["outer_bull"]):
        return 25
    if r <= radius_value(rings["single_1_end"]):
        return sector_value
    if r <= radius_value(rings["triple_end"]):
        return sector_value * 3
    if r <= radius_value(rings["single_2_end"]):
        return sector_value
    if r <= radius_value(rings["double_end"]):
        return sector_value * 2
    return 0


def radius_value(normalized_radius: float) -> float:
    return normalized_radius


def render_opencv(width: int = 800, height: int = 800):
    """Render the dartboard as a NumPy image if OpenCV is installed."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is not installed; install 'opencv-python' to use render_opencv()."
        ) from exc

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cx = width // 2
    cy = height // 2
    r = min(width, height) * 0.44

    for i in range(20):
        start = -90 + i * 18
        end = start + 18
        color = (20, 20, 20) if i % 2 == 0 else (245, 245, 245)
        ring_color = (211, 47, 47) if i % 2 == 0 else (27, 158, 74)

        pts = _wedge_polygon(cx, cy, r * 0.12, r * 0.58, start, end)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

        pts = _wedge_polygon(cx, cy, r * 0.58, r * 0.63, start, end)
        cv2.fillPoly(img, [pts], ring_color)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

        pts = _wedge_polygon(cx, cy, r * 0.63, r * 0.95, start, end)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

        pts = _wedge_polygon(cx, cy, r * 0.95, r, start, end)
        cv2.fillPoly(img, [pts], ring_color)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

    cv2.circle(img, (cx, cy), int(r * 0.12), (76, 175, 80), thickness=-1)
    cv2.circle(img, (cx, cy), int(r * 0.06), (211, 47, 47), thickness=-1)
    cv2.circle(img, (cx, cy), int(r), (0, 0, 0), thickness=4)
    return img


def _wedge_polygon(
    cx: int,
    cy: int,
    r_inner: float,
    r_outer: float,
    start_deg: float,
    end_deg: float,
    steps: int = 200,
):
    pts = []
    for i in range(steps + 1):
        theta = start_deg + (end_deg - start_deg) * i / steps
        x = cx + r_outer * math.cos(math.radians(theta))
        y = cy + r_outer * math.sin(math.radians(theta))
        pts.append((int(round(x)), int(round(y))))
    for i in range(steps, -1, -1):
        theta = start_deg + (end_deg - start_deg) * i / steps
        x = cx + r_inner * math.cos(math.radians(theta))
        y = cy + r_inner * math.sin(math.radians(theta))
        pts.append((int(round(x)), int(round(y))))
    return np.array(pts, dtype=np.int32)
