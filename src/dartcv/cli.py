"""CLI: click 4 calibration points, write overlay and top-down images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .detect import DetectionError, detect_board
from .geometry import CALIB_NAMES, compute_homography
from .overlay import draw_overlay, topdown_overlay, warp_topdown


def pick_points(img) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    win = "dartcv: click calibration points (Backspace=undo, Esc=quit)"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while len(pts) < 4:
        view = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 5, (0, 0, 255), -1)
            cv2.putText(view, CALIB_NAMES[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        msg = f"Click wire {CALIB_NAMES[len(pts)]} on the outer edge of the double ring"
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            sys.exit("aborted")
        if key in (8, 127) and pts:
            pts.pop()
    cv2.destroyAllWindows()
    return pts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="dartcv")
    ap.add_argument("image")
    ap.add_argument("--points", type=float, nargs=8, metavar="N",
                    help="x1 y1 ... x4 y4 for 20|1, 6|10, 3|19, 11|14 (skips clicking)")
    ap.add_argument("--manual", action="store_true", help="click calibration points instead of auto-detecting")
    ap.add_argument("--upright", type=float, default=0.0,
                    help="angle (deg, clockwise) at which the 20 appears in the image; resolves the 36-deg ambiguity")
    ap.add_argument("--out", default="out", help="output directory")
    a = ap.parse_args(argv)

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image} (HEIC? convert first: sips -s format jpeg in.HEIC --out out.jpg)")
    if a.points:
        H = compute_homography(list(zip(a.points[0::2], a.points[1::2])))
    elif a.manual:
        H = compute_homography(pick_points(img))
    else:
        try:
            det = detect_board(img, upright_deg=a.upright)
        except DetectionError as exc:
            sys.exit(f"auto-detection failed ({exc}); retry with --manual")
        H = det.H
        print(f"detected board (ECC score: {det.ecc_score})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "overlay.png"), draw_overlay(img, H))
    cv2.imwrite(str(out / "topdown.png"), warp_topdown(img, H))
    cv2.imwrite(str(out / "topdown_overlay.png"), topdown_overlay(img, H))
    print(f"wrote overlay.png, topdown.png, topdown_overlay.png to {out}/")
