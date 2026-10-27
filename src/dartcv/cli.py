"""CLI: detect or calibrate a board and write results and pipeline overlays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .detect import DetectionError, detect_board
from .diagnostics import calibration_overlay, detection_overlays
from .geometry import CALIB_NAMES, compute_homography
from .overlay import draw_overlay, topdown_overlay, warp_topdown
try:
    from .darts import detect_darts
except ImportError:
    detect_darts = None

DEFAULT_IMAGE_PATH = Path("data/images/raw_board")
IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
}


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
            cv2.putText(
                view,
                CALIB_NAMES[i],
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        msg = f"Click wire {CALIB_NAMES[len(pts)]} on the outer edge of the double ring"
        cv2.putText(
            view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            sys.exit("aborted")
        if key in (8, 127) and pts:
            pts.pop()
    cv2.destroyAllWindows()
    return pts


def _process_image(image_path: Path, output_dir: Path, args) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        print(
            f"cannot read {image_path} (HEIC? convert first: sips -s format jpeg in.HEIC --out out.jpg)",
            file=sys.stderr,
        )
        return False

    det = None
    if args.points:
        H = compute_homography(list(zip(args.points[0::2], args.points[1::2])))
    elif args.manual:
        H = compute_homography(pick_points(img))
    else:
        try:
            det = detect_board(img, upright_deg=args.upright, diagnostics=True)
        except DetectionError as exc:
            print(
                f"auto-detection failed for {image_path} ({exc}); retry with --manual",
                file=sys.stderr,
            )
            return False
        H = det.H
        bull_status = (
            "independently detected"
            if det.bull is not None
            else "not found; unconstrained"
        )
        print(
            f"{image_path}: detected board (ECC score: {det.ecc_score}; "
            f"bull: {bull_status}; coloured ring edges: {len(det.rings)}/4)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ring_contours = (
        {} if det is None else {name: ring.contour for name, ring in det.rings.items()}
    )
    outputs = {
        "overlay.png": draw_overlay(img, H, ring_contours=ring_contours),
        "topdown.png": warp_topdown(img, H),
        "topdown_overlay.png": topdown_overlay(img, H, ring_contours=ring_contours),
    }
    if det is not None:
        outputs.update(detection_overlays(img, det))
    else:
        outputs["05_calibration.png"] = calibration_overlay(img, H, supplied=True)

    if (getattr(args, "detect_darts", False) or getattr(args, "score", False)) and det is not None and detect_darts is not None:
        darts, diag = detect_darts(img, det, diagnostics=True)
        total_score = sum(d.score for d in darts)
        print(f"{image_path}: detected {len(darts)} dart(s), total score: {total_score}")
        for idx, d in enumerate(darts, 1):
            print(f"  Dart #{idx}: {d.label} ({d.score} pts) at px=({d.tip_px[0]:.1f}, {d.tip_px[1]:.1f}) conf={d.confidence:.2f}")

        darts_img = img.copy()
        for idx, d in enumerate(darts, 1):
            pt = (int(round(d.tip_px[0])), int(round(d.tip_px[1])))
            cv2.drawMarker(darts_img, pt, (0, 255, 0), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
            cv2.putText(
                darts_img,
                f"#{idx} {d.label} ({d.score})",
                (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        outputs["08_darts_overlay.png"] = draw_overlay(darts_img, H, ring_contours=ring_contours)

    for name, output in outputs.items():
        if not cv2.imwrite(str(output_dir / name), output):
            print(f"could not write {output_dir / name}", file=sys.stderr)
            return False
    print(f"{image_path}: wrote {', '.join(outputs)} to {output_dir}/")
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="dartcv")
    ap.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE_PATH),
        help="image file or directory of images (default: data/images/raw_board)",
    )
    ap.add_argument(
        "--points",
        type=float,
        nargs=8,
        metavar="N",
        help="x1 y1 ... x4 y4 for 20|1, 6|10, 3|19, 11|14 (skips clicking)",
    )
    ap.add_argument(
        "--manual",
        action="store_true",
        help="click calibration points instead of auto-detecting",
    )
    ap.add_argument(
        "--upright",
        type=float,
        default=0.0,
        help="angle (deg, clockwise) at which the 20 appears in the image; resolves the 36-deg ambiguity",
    )
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument(
        "--detect-darts",
        "--score",
        dest="detect_darts",
        action="store_true",
        help="detect darts and calculate turn score",
    )
    a = ap.parse_args(argv)

    source = Path(a.image)
    if source.is_file():
        images = [source]
        batch = False
    elif source.is_dir():
        images = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        batch = True
        if not images:
            sys.exit(f"no supported images found in {source}")
    else:
        sys.exit(f"image or directory does not exist: {source}")

    out = Path(a.out)
    failures = 0
    for image_path in images:
        image_out = out / image_path.stem if batch else out
        failures += not _process_image(image_path, image_out, a)
    if failures:
        sys.exit(f"{failures} of {len(images)} image(s) failed")
