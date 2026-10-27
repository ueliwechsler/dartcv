# Technical computer-vision pipeline

## Scope and coordinate contracts

The implemented pipeline estimates board geometry from a BGR photograph and detects/scores darts in play. The main entry points are [`detect_board()`](../src/dartcv/detect.py) for board alignment and [`analyze_image()`](../src/dartcv/detect.py) / [`detect_darts()`](../src/dartcv/darts.py) for full turn scoring. See the [architecture](architecture.md) for component boundaries, [ground-truth dataset](ground-truth-dataset.md) for benchmark evaluation, and the [README](../README.md#board-detection-and-diagnostic-images) for diagnostic filenames.

Model points use millimetres, origin at the bull, x right and y down; angles are
clockwise from sector 20 at the top. `H` maps model points to original-image pixels.
The [geometry module](../src/dartcv/geometry.py) owns nominal radii and sector order.

## Automatic stages, in execution order

### 1. Working image and colour segmentation

The longest side is reduced to at most `WORK_SIZE = 1000` pixels. The original
image remains available for fine measurements. HSV thresholding selects saturated
red and green pixels. A 3-by-3 opening removes small noise; a 7-by-7 elliptical
dilation joins nearby coloured regions. This grown mask is only a coarse seed,
not a measurement of the true paint edge.

### 2. Outer ellipse and initial mapping

The largest connected colour component supplies its farthest pixel in each of
180 angular bins. At least 40 boundary points are required. Iterative ellipse
fitting rejects outliers; implausibly small or elongated ellipses raise
`DetectionError`. The ellipse gives an affine mapping from the nominal
outer-double circle to the image. Its centre need not be the projected bull.

### 3. Independent bull detection and projective constraint

A crop of the original-resolution image is searched near the predicted centre.
A candidate compact red bull must have plausible size/shape and a green surround.
The initial mapping defines the search region and scale, not the measured centre.
If no candidate is reliable, `Detection.bull` is `None` and the seed is retained.

If found, `_anchor_bull()` applies a projective correction mapping the model
origin to the measured bull centre while preserving the outer-double conic.
It does not simply translate the board. The fitted red ellipse centre is still
an approximation to the projected bull, not annotated ground truth.

### 4. Orientation from the double-ring colour pattern

Red-minus-green samples at nominal radii 164, 166, and 168 mm are correlated with
the alternating sector pattern. Half-degree candidates are searched within a
36-degree period. The equivalent orientation closest to `upright_deg` is selected.
This does not independently identify the 20: an upright prior resolves the
36-degree ambiguity. The rotated seed becomes `H_initial` in original pixels.

### 5. Board-masked ECC refinement and bull re-anchoring

With `refine=True`, a nominal wire template and the rectified image are compared
using blurred Sobel-gradient magnitudes. Enhanced correlation coefficient (ECC)
alignment fits a homography at scales 0.125, 0.25, and 0.5 of the top-down image.
A board-interior mask excludes off-board lettering and background from the
objective. OpenCV errors or a final correlation below 0.2 retain the seed and
set `ecc_score` to `None`.

When a bull was measured, it is re-anchored after ECC. An excessive displacement
causing `_anchor_bull()` to reject the correction also restores the seed and
clears the score. ECC correlation describes this alignment objective, not a
probability of correct detection or final accuracy after subsequent refinement.

### 6. Source-resolution paint bands and shared geometry

[`refine_coloured_rings()`](../src/dartcv/rings.py) samples red/green radial strips
from the original image, without the coarse mask's dilation or grey-metal edges.
Spokes near sector wires are skipped. A usable run must have both paint edges
visible and width between 2 and 12 model millimetres. Each band needs at least
60 samples distributed over all four quadrants; missing spokes are omitted.

If both double and treble bands have support, a bounded five-parameter correction
is fitted to band centres with SciPy robust least squares (`soft_l1`). This
correction fixes the model origin, has no free rotation, and is accepted only
on optimizer success without worsening median absolute radial residual.
Sampling is repeated after an accepted correction. Otherwise the previous
homography remains available for independent edge fitting.

### 7. Independent paint boundaries and final result

Each inner/outer paint edge is robustly fitted as an ellipse in rectified
coordinates, then projected into the original image. Fits require enough inliers,
distributed support, plausible radius/centre, and a non-crossing 2-to-12 mm band.
Accepted edge pairs populate `Detection.rings`; rejected bands stay nominal in
overlays. Smooth ellipses bridge gaps and are not pixel-perfect segmentations.

`RingBoundary.median_error_px` is a radial fit residual in original-image pixels,
not a ground-truth accuracy estimate. Common geometry refinement can change the
outer conic from the earlier seed; the origin remains fixed during this stage.

### 8. Single-image self-referenced dart detection and tip localization

[`detect_darts()`](../src/dartcv/darts.py) locates and scores darts from a single
image and its estimated board homography `H`, without requiring a prior empty-board
reference frame:

1. **Top-down LAB self-referenced board model (`720x720`, `1.0 px/mm`)**: The source
   image is warped to an extended canonical frame (`360 mm` half-span covering the
   outer surround). Median CIE LAB values are estimated for each ring and sector
   colour class (black, cream, red, green, bull, and outer catch ring) after
   excluding metal spider wires. Cast shadows on the board bed are suppressed by
   identifying darkened pixels with low chromaticity shift (`dL < -20`, `dC < 14`).
2. **Distance-transform candidate proposal**: Morphological cleanup and Euclidean
   distance transforms on the foreground anomaly mask isolate thick flight/barrel
   blobs inside the scoring board (`r <= 170 mm`, `dist >= 3.6 mm`) and outer
   catch ring (`170 < r <= 220 mm`, `dist >= 8.5 mm`), supplemented by a dedicated
   bright-flight detector in the black surround (`170 < r <= 245 mm`, `L > 175`,
   `5.5 <= dist <= 26.0 mm`). Local maxima undergo `18.0 mm` non-maximum suppression
   to seed candidate flight/barrel origins.
3. **Coarse-to-fine inward ray search**: From each candidate origin, inward rays
   are evaluated across `360°` at `1.0°` coarse steps and `0.25°` fine steps on a
   wire- and shadow-masked `9x9` morphological top-hat map (`tophat9_angle`). Rays
   are scored by centre-minus-flank spine contrast (`C - 0.85 * F` at `±7 mm`
   flanks) weighted by active barrel core coverage, followed by `±2.5 mm` lateral
   ridge snapping (`tophat5`) to lock onto the barrel centerline.
4. **Axial barrel profile tracing and tip localization**: Along the snapped axis,
   a `1.0 mm` morphological profile (`thin9_axial`) traces the contiguous barrel
   plateau (`S_smooth >= 14.0`). When an outer-surround flight (`r > 175 mm`)
   drops into a dark or translucent shaft before `45 mm` of travel (`r_drop > 120 mm`),
   gated shaft plateau bridging checks for a downstream metallic barrel plateau
   (`S_smooth >= 22.0` over `6.0 mm` with `tophat5_no_wire >= 8.0`). For barrels
   terminating just outside the outer double wire (`167.0 <= r_barrel_end <= 176.0 mm`),
   wire-crossing needle extension traces `tophat5_no_wire` (`>= 6.5`) across the
   wire gap into the double bed. Rays passing within `3.5 mm` of the bullseye
   origin with bull anomaly support snap to the closest approach point.
5. **Post-ray deduplication and scoring**: Proposals whose tail centroids lie
   within `32.0 mm` (collapsing multi-fin flight peaks on the same dart) or that
   share an axial segment (`< 11.0 mm` midpoint and `< 16.0 mm` tip proximity)
   are suppressed. Surviving tips are mapped back to original image pixels via `H`
   and scored with [`score_point_detailed()`](../src/dartcv/geometry.py).

## Optional modes and result data

- `refine=False` skips ECC and colour-ring refinement, but still performs initial
  bull detection/anchoring and orientation estimation. Ring results are empty.
- `diagnostics=True` retains intermediate data in `DetectionDiagnostics`,
  including mask, boundary samples/inliers, attempted ECC inputs, and pre-bull
  and pre-ring baselines. Consult field comments for their coordinate frames.
- `Detection` also exposes final/initial homographies, the original-image seed
  ellipse, rotation, optional ECC score, bull measurement, and fitted boundaries.
- Manual or supplied-point CLI modes bypass automatic detection and compute a
  homography from four outer-double intersections at angles 9, 99, 189, 279 degrees
  (`20|1`, `6|10`, `3|19`, `11|14`). They produce supplied-point calibration
  diagnostics, not automatic-stage images.

## Rendering, rectification, and scoring

[`overlay`](../src/dartcv/overlay.py) uses measured paint contours when present,
including radial-line endpoints at the outer double. Other rings remain nominal.
The top-down transform is `model_to_topdown() @ inverse(H)`, with 4 pixels/mm
and a 25 mm margin outside the nominal outer double.

Automatic calibration markers are model-projected radial angles, not independently
measured wire intersections. [`diagnostics`](../src/dartcv/diagnostics.py) visualizes
these estimates and refinement changes; the README catalogs the generated images.

`score_from_image_point()` inversely maps an already supplied pixel location;
`score_from_model_point()` applies nominal radii and sector order.
`score_point_detailed()` uses the independently fitted `Detection.rings` boundaries
when supplied and falls back to nominal `RADII` otherwise.

## Validation, failure modes, and maintenance

The [homography tests](../tests/test_homography.py), [ring tests](../tests/test_rings.py), and [dart detection tests](../tests/test_darts.py) verify synthetic cases and fallback behavior. The [ground-truth evaluation](ground-truth-dataset.md) provides reproducible quantitative benchmarks on curated real imagery (`15` starter images across `raw_board`, `single_dart`, and `multi_dart`):
- **Board wire grid alignment (`5 raw_board` images, 24 spider intersections each)**: `0.53 mm` mean (`5.08 px`), `0.48 mm` median (`4.24 px`), `0.36 mm` mean bull centre error.
- **Clean-board specificity (`5 raw_board` images)**: `100.0%` (`0` false positives).
- **Single-dart turns (`5 single_dart` images)**: `100.0%` dart count accuracy (`5/5`), `100.0%` turn score accuracy (`5/5`), `80.0%` precision/recall within `15.0 mm`.
- **Overall dataset (`15` images, `20` GT darts)**: `93.33%` dart count accuracy (`14/15` images), `73.33%` exact turn total score match (`11/15` images), `81.82%` matched per-dart sector/multiplier and point accuracy, `7.87 mm` median matched tip error (`8.29 mm` mean), and `380.4 ms` mean end-to-end runtime per image (`132.7 ms` board alignment + `247.7 ms` dart detection).

Run `uv run --group dev pytest` for unit tests and `uv run dartcv-eval` for full dataset evaluation. Real-image visual checks should use one explicitly selected image unless a batch evaluation is requested.

Colour shifts, glare, multi-dart barrel/flight occlusion, dark or camouflaged shafts, steep perspective foreshortening where the barrel stops >15 mm short of the needle penetration point, and insufficient board size can undermine segmentation or localization. Optional fallbacks keep a result usable but do not certify it as correct. Dataset-level percentage accuracy is evaluated against [`data/annotations/ground_truth.json`](../data/annotations/ground_truth.json) using minimum-weight bipartite matching.

Update this document whenever stages/order, coordinate contracts, thresholds,
objectives, acceptance/fallback behavior, outputs, or scoring semantics change.
Keep source docstrings and the [architecture](architecture.md) consistent with it.