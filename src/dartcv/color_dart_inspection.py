"""Explicit, side-effect-free visualization of board-preparation runs."""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib.figure import Figure

from .color_dart_pipeline import (
    BOARD_RADII_MM,
    BoardGeometry,
    BoardPreparationResult,
    Image,
    Matrix3,
    OrientationEstimate,
    StageOutcome,
    _validate_image,
)


def draw_board_geometry(image: Image, model_to_image: Matrix3) -> Image:
    """Draw nominal rings/spokes, not measured wires or confirmed sector labels."""
    _validate_image(image)
    transform = BoardGeometry(model_to_image, detector_rotation_deg=0.0).model_to_source
    output = image.copy()
    thickness = max(1, round(min(image.shape[:2]) / 700))

    def draw(points: np.ndarray, *, closed: bool) -> None:
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ transform.T
        denominators = homogeneous[:, 2]
        if np.any(np.abs(denominators) < 1e-12) or (
            np.any(denominators > 0) and np.any(denominators < 0)
        ):
            raise ValueError("Board geometry crosses the projective horizon.")
        projected = homogeneous[:, :2] / denominators[:, None]
        if not np.isfinite(projected).all() or np.any(np.abs(projected) > 2**30):
            raise ValueError("Board geometry projects outside supported pixel coordinates.")
        cv2.polylines(
            output,
            [np.rint(projected).astype(np.int32).reshape(-1, 1, 2)],
            closed,
            (255, 0, 255),
            thickness,
            cv2.LINE_AA,
        )

    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    for radius in BOARD_RADII_MM.values():
        draw(
            radius * np.column_stack((np.sin(angles), -np.cos(angles))),
            closed=True,
        )
    for angle_deg in np.arange(9.0, 360.0, 18.0):
        angle = np.deg2rad(angle_deg)
        radii = np.array([BOARD_RADII_MM["outer_bull"], BOARD_RADII_MM["double_outer"]])
        draw(np.column_stack((radii * np.sin(angle), -radii * np.cos(angle))), closed=False)
    return output


def plot_board_preparation(
    result: BoardPreparationResult, *, max_display_px: int = 1400
) -> Figure:
    """Return a six-panel inspection figure; never show or save automatically."""
    if not result.diagnostics_enabled:
        raise ValueError("Run prepare_board(..., diagnostics=True) before plotting.")
    if (
        isinstance(max_display_px, bool)
        or not isinstance(max_display_px, int)
        or max_display_px < 2
    ):
        raise ValueError("Display size must be an integer of at least 2 pixels.")
    if not result.steps or "image" not in result.steps[0].input_artifacts:
        raise ValueError("The preparation result has no captured source image.")

    def preview(image: Image) -> tuple[Image, Matrix3]:
        height, width = image.shape[:2]
        scale = min(1.0, max_display_px / max(height, width))
        output_width = max(1, round(width * scale))
        output_height = max(1, round(height * scale))
        resized = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
        transform = np.diag([output_width / width, output_height / height, 1.0])
        return resized, transform

    source = result.steps[0].input_artifacts["image"]
    source_preview, source_scale = preview(source)
    panels: list[tuple[str, np.ndarray | None]] = [
        ("Input (BGR converted to RGB for display)", source_preview),
        (
            "Source geometry (nominal rings and spokes)",
            draw_board_geometry(source_preview, source_scale @ result.board.model_to_source)
            if result.board is not None
            else None,
        ),
        (
            "Red/green mask (detector working resolution)",
            result.steps[0].output_artifacts.get("colour_mask"),
        ),
    ]
    if result.board_view is not None:
        view = result.board_view
        normalized, view_scale = preview(view.image)
        panels.extend(
            [
                ("Normalized board (uncorrected 20-sector alignment)", normalized),
                (
                    "Normalized geometry (check ring/spoke alignment)",
                    draw_board_geometry(normalized, view_scale @ view.model_to_view),
                ),
                ("Nominal board mask (not observed image coverage)", view.board_mask),
            ]
        )
    else:
        panels.extend(
            (title, None)
            for title in ("Normalized board", "Normalized geometry", "Nominal board mask")
        )

    figure = Figure(figsize=(16, 10), layout="constrained")
    axes = figure.subplots(2, 3).flat
    for axis, (title, image) in zip(axes, panels, strict=True):
        if image is not None:
            if image.ndim == 3:
                axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
        axis.set_title(title)
        axis.set_axis_off()
    if result.failure is not None:
        figure.suptitle(
            f"{result.failure.stage}: {result.failure.message}", color="red"
        )
    else:
        figure.suptitle("Board preparation only: no dart recognition or confirmed orientation")
    return figure


def plot_orientation(result: StageOutcome[OrientationEstimate]) -> Figure:
    """Inspect template, unwrapped candidates, response and wider input view."""
    required = ("number_ring", "number_strip", "printed20_template", "angular_response")
    if any(name not in result.artifacts for name in required):
        raise ValueError("Run the orientation stage with diagnostics=True before plotting.")
    figure = Figure(figsize=(15, 9), layout="constrained")
    axes = figure.subplots(2, 2)
    for axis, name, title in (
        (axes[0, 0], "printed20_template", "Explicit reference 20 (board-style dependent)"),
        (axes[0, 1], "number_strip", "Unwrapped ring: outer edge at top"),
        (axes[1, 1], "number_ring", "Wider input (not orientation-corrected)"),
    ):
        axis.imshow(cv2.cvtColor(result.artifacts[name], cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.set_axis_off()
    response = result.artifacts["angular_response"]
    angles = np.arange(len(response)) * 360.0 / len(response) - 180.0
    axes[1, 0].plot(angles, response)
    axes[1, 0].set(xlabel="Clockwise angle from model top (degrees)", ylabel="Correlation",
                   title="Distinct candidate peaks (not probabilities)", ylim=(-0.1, 1.05))
    candidates = result.artifacts.get("orientation_candidates")
    if candidates is not None:
        for rank, (angle, strength) in enumerate(candidates, start=1):
            axes[1, 0].plot(angle, strength, "ro")
            axes[1, 0].annotate(str(rank), (angle, strength))
            column = (angle + 180.0) * len(response) / 360.0
            axes[0, 1].axvline(column, color="red", linewidth=1)
    if result.value is None:
        figure.suptitle(f"Orientation unavailable: {result.failure}", color="red")
    else:
        value = result.value
        label = "confident" if value.confident else "UNCERTAIN best guess"
        transform = result.artifacts.get("number_ring_model_to_view")
        if transform is not None:
            angle = np.deg2rad(-value.correction_deg)
            point = transform @ np.array([215 * np.sin(angle), -215 * np.cos(angle), 1])
            point = point[:2] / point[2]
            axes[1, 1].annotate(
                "best 20", point, xytext=(12, 20), textcoords="offset points",
                color="red", arrowprops={"arrowstyle": "->", "color": "red"},
            )
        figure.suptitle(
            f"{label}: clockwise correction {value.correction_deg:.2f} degrees; "
            f"match strength {value.confidence:.3f}"
        )
    return figure
