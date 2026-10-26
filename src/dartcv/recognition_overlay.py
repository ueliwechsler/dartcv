"""Draw recognition observations on a copy of a BGR image, without inference."""

from collections.abc import Sequence

import cv2
import numpy as np

from .recognition import DartBox, DartTip, NumberReading


def annotate(
    base: np.ndarray,
    boxes: Sequence[DartBox] = (),
    numbers: Sequence[NumberReading] = (),
    tips: Sequence[DartTip] = (),
    scores: Sequence[int] | None = None,
) -> np.ndarray:
    """Render boxes, observed text, and tips; scores must align one-to-one."""
    if scores is not None and len(scores) != len(tips):
        raise ValueError("scores must have one entry per tip")
    out = base.copy()
    scale = max(1, round(max(out.shape[:2]) / 1600))

    def label(text, xy, colour):
        pos = (int(xy[0]), int(xy[1]))
        cv2.putText(
            out,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * scale,
            (0, 0, 0),
            4 * scale,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * scale,
            colour,
            2 * scale,
            cv2.LINE_AA,
        )

    for box in boxes:
        x0, y0, x1, y1 = (int(round(value)) for value in box.xyxy)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2 * scale)
        label(
            f"dart {box.id} {box.confidence:.2f}",
            (x0, max(y0 - 8 * scale, 0)),
            (0, 200, 255),
        )
    for number in numbers:
        polygon = np.asarray(number.polygon, float)
        cv2.polylines(
            out,
            [np.round(polygon).astype(np.int32)],
            True,
            (255, 200, 0),
            2 * scale,
            cv2.LINE_AA,
        )
        label(number.text, polygon.min(axis=0), (255, 200, 0))
    for index, tip in enumerate(tips):
        x_px, y_px = (int(round(value)) for value in tip.xy)
        cv2.drawMarker(
            out,
            (x_px, y_px),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            30 * scale,
            3 * scale,
            cv2.LINE_AA,
        )
        if scores is not None:
            label(
                str(scores[index]), (x_px + 12 * scale, y_px - 12 * scale), (0, 0, 255)
            )
    return out
