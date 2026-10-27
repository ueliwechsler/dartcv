# Ground-Truth Benchmark Dataset and Annotation System

This document specifies the starter ground-truth dataset, annotation schema, interactive annotation interface, and quantitative evaluation harness for DartCV.

## 1. Dataset Overview

The starter benchmark dataset consists of 15 curated high-resolution images captured under realistic lighting and camera angles, stored under `data/images/`:
- **5 Raw Board Images (0 darts)**: `data/images/raw_board/IMG_2086.jpg`, `IMG_2092.jpg`, `IMG_2100.jpg`, `PXL_20261018_123812087.jpg`, and `PXL_20261018_123851704.jpg`. Used to evaluate false positive rates and baseline board alignment across diverse boards and orientations.
- **5 Single-Dart Images (1 dart each)**: `data/images/single_dart/PXL_20261018_152634348.jpg` (S4, 4 pts), `PXL_20261018_152711553.jpg` (DBULL, 50 pts), `PXL_20261018_152805333.jpg` (S16, 16 pts), `PXL_20261018_152924982.jpg` (T20, 60 pts), and `PXL_20261018_153231430.jpg` (S5, 5 pts). Used to benchmark tip localization precision and scoring on single-dart turns.
- **5 Multi-Dart Images (multiple darts each)**: `data/images/multi_dart/PXL_20261018_154231645.jpg`, `PXL_20261018_154304248.jpg`, `PXL_20261018_154400079.jpg`, `PXL_20261018_154523185.jpg`, and `PXL_20261018_154739543.jpg`. Used to evaluate occlusions, dart grouping, and multi-dart total turn scoring.

## 2. Coordinate Conventions

DartCV distinguishes two coordinate frames:
- **Image Pixel Space `(x, y)`**: Measured in original image pixels where `(0, 0)` is the top-left corner, `x` extends horizontally to the right across image width, and `y` extends vertically downwards across image height.
- **Canonical Board Model Space `(x_mm, y_mm)`**: Measured in millimeters on the physical board plane where the origin `(0, 0)` is at the exact center of the double bullseye, `x_mm` extends to the right, `y_mm` extends downwards (matching image convention), and sector angles are measured clockwise starting from the top vertical ray through the centre of sector 20 (0°).
- **Tip Ground-Truth Definition**: The dart tip coordinate is defined as the point where the steel needle tip penetrates into the sisal bed. In high-resolution perspective imagery, this point strictly coincides with the convergence of the needle shaft with its cast shadow on the board surface.

## 3. Annotation Schema

Annotations are versioned and stored in `data/annotations/ground_truth.json`:

```json
{
  "version": 1,
  "starter_15": [
    "data/images/raw_board/IMG_2086.jpg",
    "..."
  ],
  "images": {
    "data/images/single_dart/PXL_20261018_152634348.jpg": {
      "image_path": "data/images/single_dart/PXL_20261018_152634348.jpg",
      "calibration_points_px": null,
      "darts": [
        {
          "tip_px": [3456.0, 3314.0],
          "tail_px": [3836.0, 2527.0],
          "label": "S4",
          "sector": 4,
          "multiplier": 1,
          "score": 4
        }
      ],
      "total_score": 4,
      "verified": true
    }
  }
}
```

Field semantics:
- `calibration_points_px`: Optional 4-point list `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]` corresponding to the 4 outer double wire intersections `(20|1, 6|10, 3|19, 11|14)` if manual override is used; `null` indicates automatic ECC board pose.
- `wire_grid_points_px`: Optional 24-point list of physical spider wire intersections across 4 cardinal spokes (`20|1, 6|10, 3|19, 11|14`) across all concentric wire rings.
- `tip_px`: Exact pixel location `[x, y]` of the needle penetration point in original image coordinates.
- `tail_px`: Flight/tail point `[x, y]` along the dart barrel axis to define shaft orientation.
- `label`: Standard dart score notation (`S4`, `D16`, `T20`, `BULL`, `DBULL`, `MISS`).
- `verified`: Boolean flag indicating whether human inspection has confirmed the annotations.

## 4. Interactive Annotation Tool (`dartcv-annotate`)

The `dartcv-annotate` command provides a web-based annotation interface with pan, zoom, loupe inspection, and real-time score calculation:

```bash
# Start local web annotator (default port 8080)
.venv/bin/python -m dartcv.annotate

# Specify port or annotations file
.venv/bin/python -m dartcv.annotate --port 8080 --annotations data/annotations/ground_truth.json

# Export standalone HTML artifact
.venv/bin/python -m dartcv.annotate --export-html out/annotator.html
```

### Key UI Features:
- **Pan & Zoom Canvas**: Mouse wheel zooms smoothly up to 25x magnification; right-click drag, middle-click drag, or Shift+left-drag pans across 8K resolution images without downsampling artifacts.
- **4x Magnifier Loupe**: Floating loupe displays an uncompressed, pixelated 4x zoom around the cursor with fine crosshairs and live pixel coordinates `(X, Y)`.
- **Joint Projective Wire Grid Calibration (24-Pt)**: Calibrates exact spider wire intersections across 4 cardinal spokes (`20|1, 6|10, 3|19, 11|14`) across all concentric rings. Uses an 8-iteration Alternating Least Squares (ALS) solver to jointly optimize the 8-DoF planar projective homography `H` and 5 scalar ring radii `R_k` (with `R_do = 170.0 mm`), producing a physically valid planar projective ground truth where dragging any handle (outer bull, inner bull, treble, or double) immediately updates the full board perspective and moves all 20 straight radial segment lines together.
- **Projected Wireframe Overlay**: Projects the 6 concentric wire boundaries and 20 radial sector dividers directly onto the perspective view using the detected or calibrated homography `H`.
- **Click-to-Annotate Workflow**: Left-click 1 places the dart tip; Left-click 2 places the dart tail. The backend calculates sector, multiplier, score, and label instantaneously via `score_point_detailed(H, [x, y])`.
- **Keyboard Shortcuts**:
  - `Space` / `Enter`: Mark as verified, save to `ground_truth.json`, and advance to next image.
  - `N` / `P`: Next / Previous image without overwriting.
  - `Z`: Undo last placed point.
  - `O`: Toggle projected board wireframe overlay.
  - `R`: Reset pan and zoom to fit image.
  - `V`: Toggle verified status.

## 5. Quantitative Evaluation Harness (`dartcv-eval`)

The `dartcv-eval` CLI runs quantitative benchmarks against the ground-truth dataset:

```bash
# Run evaluation and generate reports and diagnostic composites in out/eval/
.venv/bin/python -m dartcv.evaluate

# Custom tolerance gate (default 15.0 mm) and output directory
.venv/bin/python -m dartcv.evaluate --max-error-mm 12.0 --out out/eval
```

### Metrics Computed:
- **Board Alignment Performance (Wire Grid 24-Pt)**:
  - Evaluated on verified `raw_board` images against human-verified spider intersections.
  - Mean, median, P95, corner, and center error in board millimeters (`mm`) and image pixels (`px`).
  - Current benchmark (`5 raw_board` images): `0.53 mm` (`5.08 px`) mean wire grid error, `0.48 mm` (`4.24 px`) median wire grid error, `0.36 mm` (`3.16 px`) center error, and `0.76 mm` (`7.73 px`) outer-double corner error.
- **Detection Performance**:
  - **Dart Count Accuracy**: Percentage of turns where detected dart count equals ground-truth dart count (`93.33%` overall across 15 images: `100.0%` on `raw_board`, `100.0%` on `single_dart`, `80.0%` on `multi_dart`).
  - **Precision, Recall, F1**: Based on minimum-weight bipartite matching in canonical board space within `max_error_mm` (`15.0 mm`) tolerance.
  - Breakdown by dataset category: `raw_board` (`100.0%` specificity, `0` false positives), `single_dart` (`100.0%` turn score accuracy, `80.0%` precision/recall within `15.0 mm`), and `multi_dart` (`80.0%` dart count accuracy).
- **Tip Localization Error**:
  - Median (`7.87 mm` / `78.52 px`), Mean (`8.29 mm` / `76.85 px`), and 90th percentile (`12.76 mm` / `113.09 px`) Euclidean error on matched darts in board millimeters (`mm`) and original image pixels (`px`).
- **Scoring Accuracy**:
  - Exact sector/multiplier label accuracy (`81.82%` on matched darts), point accuracy (`81.82%`), and turn total score accuracy (`73.33%` overall; `100.0%` on `raw_board` and `single_dart`).
- **Runtime Latency**:
  - Mean (`380.4 ms`), median (`375.4 ms`), and P95 (`425.6 ms`) end-to-end latency per image (`132.7 ms` mean board alignment + `247.7 ms` mean dart detection).
- **Visual Diagnostics**:
  - Generates side-by-side diagnostic composites (`out/eval/<stem>_eval.png`) displaying original perspective view with GT tips (magenta circles), predicted tips (green crosses), and matched error vectors alongside top-down rectified board analysis.

## 6. Dual-Tab Evaluation & Inspection Dashboard

The local server serves an integrated, dual-tab web application accessible at `http://localhost:8080/`:
- **[📊 Pipeline Dashboard]**: Complete evaluation overview, benchmark telemetry, and unseen image analysis.
- **[🎯 Annotator]**: High-resolution interactive ground-truth curation interface.

A standalone offline snapshot can also be exported:
```bash
.venv/bin/python -m dartcv.annotate --export-dashboard out/eval/dashboard.html
```

### Dashboard Sections:
- **Section A: Telemetry & KPI Cards**:
  - Dataset summary (total, verified, GT vs predicted darts).
  - Detection precision, recall, and F1 score with category-specific metrics.
  - Tip localization error (median, mean, P90) in mm and px.
  - 24-point wire grid alignment error for verified board calibration.
  - Stage latency percentile bars (Mean, Median, P95) for board alignment, dart detection, and total runtime.
  - Live "Re-run Evaluation" button (`POST /api/run_eval`) to recompute metrics on demand without server restart.
- **Section B: 15-Image Benchmark Evaluation Table**:
  - Interactive table displaying image preview, category badge, GT vs predicted dart counts, detected labels, tip error, and board alignment error.
  - Quick filter pills: `All`, `Raw Board (5)`, `Single Dart (5)`, `Multi Dart (5)`, `Verified (15)`, and `Needs Review (0)`.
  - Clicking any row opens a full-screen diagnostic composite modal with pan/zoom.
- **Section C: Unseen Dataset Inspector**:
  - Explores all 63 unannotated images in `data/images/` outside the starter-15 benchmark.
  - Starter 9-image visual gallery (3 raw board, 3 single dart, 3 multi dart) with one-click live CV pipeline execution (`GET /api/analyze_live`).
  - Dropdown selector to inspect any of the 63 unseen images.
  - Interactive side-by-side modal displaying the original image, detected board pose and wire grid overlay, dart bounding boxes, tip/tail locations, and live turn score breakdown (`GET /api/render_analysis`).
