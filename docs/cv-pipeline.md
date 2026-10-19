# Technical computer-vision pipeline

## Scope and coordinate contracts

The implemented pipeline estimates board geometry from a BGR photograph. It does
not detect darts or recognize printed sector numbers. The main entry point is
[`detect_board()`](../src/dartcv/detect.py), which returns a `Detection` object.
See the [architecture](architecture.md) for component boundaries and the
[README](../README.md#board-detection-and-diagnostic-images) for diagnostic filenames.

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
`score_from_model_point()` applies nominal radii and sector order. Neither uses
the independent paint contours nor locates a dart tip.

## Validation, failure modes, and maintenance

The [homography tests](../tests/test_homography.py) and
[ring tests](../tests/test_rings.py) verify synthetic cases and fallback behavior.
Run `uv run --group dev pytest`. Real-image visual checks should use one explicitly
selected image unless a batch evaluation is requested.

Colour shifts, glare, occlusion, clutter, insufficient board size, and strong
perspective can undermine segmentation or measurements. Optional fallbacks keep
a result usable but do not certify it as correct. No dataset-level percentage
accuracy is established here; that requires annotated ground truth, a defined
metric, dataset/version, and reproducible evaluation.

Update this document whenever stages/order, coordinate contracts, thresholds,
objectives, acceptance/fallback behavior, outputs, or scoring semantics change.
Keep source docstrings and the [architecture](architecture.md) consistent with it.