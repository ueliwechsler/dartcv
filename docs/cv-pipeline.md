# Technical computer-vision pipeline

## Scope and coordinate contracts

The core pipeline estimates board geometry from a BGR photograph. Experimental
classical and optional pretrained backends attempt whole-dart boxes, not tips or
recognize printed sector numbers. The main board entry point is
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

## Recognition contracts and next detection stage

The [recognition module](../src/dartcv/recognition.py) extracts reusable contracts
from the [recognition notebook](../examples/dart_recognition.ipynb). It does not
implement inference. Backends have these call signatures:

- `DART_DETECTOR(image) -> Iterable[DartBox]`
- `NUMBER_READER(image, H) -> Iterable[NumberReading]`
- `TIP_LOCATOR(image, boxes) -> Iterable[DartTip]`

Coordinates refer to the original BGR image, not a resized input, crop, or
rectified view. Adapters must undo preprocessing before returning observations.
Boxes have unique IDs and ordered xyxy bounds inside the image. Number polygons
need at least three finite in-image vertices and nonempty observed text. Tips
must be finite in-image (x, y) points; associated tips must reference an existing
box and lie within its bounds plus 10% of its longest side. Every confidence must
be in [0, 1]; this range check does not establish probability calibration.

`run_stage()` materializes and validates the entire result. Missing backends
report `not configured`; inference or validation exceptions report `failed`
with no items. `ok` with no items means the configured backend returned no
observations. Validators check structural plausibility, not physical correctness.
`expected_sector()` only compares readings against the current orientation.
The [recognition overlay](../src/dartcv/recognition_overlay.py) draws observations
on a copy; optional scores must have exactly one entry per tip.

### Classical single-dart experiment

[`ClassicalDartDetector`](../src/dartcv/classical_darts.py) is a NumPy/OpenCV-only
backend for one visible flight plus an attached shaft/barrel. Its callable takes
the original BGR uint8 image; construction takes a finite invertible `H` mapping
board millimetres to source pixels. It returns zero or one `DartBox`, never tips.

1. Downsize the longest image side to at most `work_size=1200`. Preserve actual
  x/y resize factors independently, including rounding. Project a search disk
  with 1.4 times the nominal outer-double radius. Keep original perspective;
  do not assume the shaft points towards the bull or lies in the board plane.
2. Propose saturated local Lab chroma deviations (magnitude >12 from Gaussian
  background, sigma 9, HSV saturation >60, value >45), plus bright neutral
  deviations (Lab lightness residual >15, saturation <65, value >180).
  Open the neutral mask with a 5-pixel elliptical kernel before combining it,
  to suppress thin white wires. Close/open the combined mask with a 3-pixel
  kernel and intersect with the search region. Optional inclusive `flight_hsv`
  bounds replace these automatic proposals; hue 0-179 can wrap through red.
3. Keep compact contours with area between `(0.025*r)^2` and `(0.4*r)^2`, where
  `r` is the median projected outer-double radius. Require both bounding-box
  sides >=4 pixels, aspect <3 and contour/hull area ratio >0.45. Reject a contour
  if >=80% of its outline lies inside the outer bull plus 3 mm, or within the
  nominal treble/double bands plus 3 mm. Keep at most 80 largest candidates.
  This conservative paint prior can reject a real flight overlapping a band;
  nominal paint geometry and imperfect calibration are additional limitations.
4. Run Canny (50/120) and probabilistic Hough lines (1-pixel, 0.5-degree bins,
  20-vote threshold, minimum length `max(12, .08*r)`, gap `max(3, .08*r)`).
  Retain segments with midpoints in the search region. A shaft endpoint must
  be within .12 flight sizes of its contour and .9 sizes of its centroid;
  segment length must be .7-6 flight sizes. A flight size is its longest box side.
5. Pair edges within 8 degrees of parallel, with >60% axial overlap and separation
  between `max(1.5 pixels, .08 flight sizes)` and .25 flight sizes. Sample 24
  transverse profiles; require median brightness contrast >=10 between the
  middle and mean outer samples. The quality score combines overlap (weight
  .4), contrast capped at 40 (.3), flight solidity (.15), and shaft length capped
  at two flight sizes (.15). Keep the best pair per flight and rank flights.
6. Accept only above `min_score=.65` and at least `ambiguity_margin=.05` ahead
  of the runner-up. Empty, weak or competing evidence returns no box. The box
  includes the flight and paired segments, padded by `max(2 pixels, .04 flight
  sizes)`, restored to source pixels and clipped/validated. These are heuristic
  quality scores, not probabilities, and may be high for incorrect board marks.

`diagnostics` retains the working image, x/y scale, search mask, raw proposal
mask, filtered flight contours, line segments, ranked hypotheses and decision.
All diagnostic coordinates are working pixels; only output boxes are source
pixels. `diagnostic_views()` draws search, flight-mask, shaft-segment and
hypothesis views, with cyan flight contours, green accepted and orange rejected
hypotheses. It does not write files or alter input images.

Synthetic tests cover coloured/white flights, HSV hue wrap, empty/flight-only/
shaft-only negatives, bull/wire rejection, competing darts and resizing. On the
one selected real image `PXL_20261018_152634348.jpg` (2026-10-22), the revised
default detector abstained because six board-marking hypotheses competed; it
did not establish the actual white-flight/shaft pair. The classical stage took
0.250 seconds in one Apple Silicon CPU run, excluding image loading, board
calibration, plotting and imports. This is not a recognition-accuracy or latency
benchmark. Initial chroma-only proposals had falsely joined bull and wire;
that failure motivated neutral proposals and explicit paint rejection.

Dark flights, flights merged with paint, distant/thin barrels, broken shafts,
shadows and lettering remain failure modes. The single-image result is a known
limitation, not a success. No image-specific HSV tuning or forced detection is
used to conceal it. Known-colour support is an explicit assumption, not automatic
colour discovery. This detector is not a general multi-dart system.

### Pretrained whole-dart experiment

[`Owlv2DartDetector`](../src/dartcv/dart_detection.py) implements the dart backend
using `google/owlv2-base-patch16-ensemble`, an Apache-2.0 pretrained model.
Its default revision is pinned; no task-specific training is performed.
It matches image regions to text prompts rather than relying on a fixed set of
class names. This allows a dart query but does not establish dart recognition
quality. The default query is `a dart`, score threshold 0.1, NMS IoU threshold
0.5, and device `cpu`. Runtime dependencies are optional; see the README setup.

The callable accepts nonempty BGR uint8 images, converts to RGB without mutation,
and uses the supplied OWLv2 processor's native square padding, resizing, and
normalization. The image processor's postprocessing receives the original
`(height, width)` to undo its transformations. There is no top-down warp, hard
board mask, crop, or tiling. A planar board mapping cannot accurately rectify
protruding darts. Model construction is lazy; first inference loads weights once
per instance, with `local_files_only=True` available to prohibit network access.

Nonfinite coordinates, invalid prompt indices, and scores outside [0, 1] fail
the stage. Finite boxes are clipped to pixel bounds; collapsed/outside boxes and
weak scores are dropped. OpenCV `dnn.NMSBoxes` suppresses overlapping duplicate
predictions across prompts. Its IoU (intersection area divided by union area)
threshold is configurable because real overlapping darts can also be suppressed.
Remaining boxes receive IDs starting at 1 in descending score order, local to
the image, and record model/revision and prompt in provenance. No fixed dart
count is imposed. Load/inference failures propagate to `run_stage()` rather than
becoming successful empty results.

Inspect whole visible extent, not just the flight. Thin shafts may disappear
when downsampled; occlusion and unfamiliar dart designs may cause misses or
partial boxes. Lower thresholds can increase false positives on boards, wires,
and shadows. Text-match scores are not calibrated probabilities. Unit tests
verify integration and contracts, not recognition quality; a single-photo visual
check is not dataset-level evaluation. This backend produces no tips or scores.

Single-image feasibility check (2026-10-22): on
`data/images/single_dart/PXL_20261018_152634348.jpg`, pinned revision
`cfd3195ba4ea9592eec887ded089f4c08eff231d`, prompt `a dart`, threshold 0.1,
NMS IoU 0.5, and CPU, four boxes survived. Visual inspection showed the strongest
box (score 0.332) covered the real visible flight and shaft/barrel; three weaker
boxes (0.102-0.134) matched printed dart graphics in the board logo, not physical
darts. Inference including first load/cache lookup/download took 10.43 seconds
in that run (PyTorch 2.14.1, Transformers 4.57.6, Python 3.13, Apple Silicon).
This is a qualitative observation, not a precision/recall or latency benchmark.
It demonstrates why a successful stage and plausible score do not certify a box.

### Future training and tip localisation (not implemented)

1. Annotate each dart's full visible bounding box and its board entry-point
  keypoint on original images. Mark hidden or ambiguous tips as unavailable,
  not guessed. Include empty boards, multiple/overlapping darts, shadows, and
  varied views. Split by capture session/board before augmentation to avoid
  near-duplicate images leaking into evaluation.
2. Fine-tune an existing detector/keypoint framework, such as a YOLO pose model
  with one dart class and one tip keypoint, after checking its license and
  runtime compatibility. Generic pretrained object weights alone are not a
  dart-tip detector. Preserve original perspective: the board homography
  describes the board plane, not the protruding shaft or flight.
3. Add an optional inference adapter that handles resize/letterbox reversal,
  confidence filtering, duplicate suppression, and per-image IDs. Map boxes
  to `DartBox` and only reliable, visible entry points to `DartTip`. The notebook
  has separate box/tip slots; share one inference result between adapters rather
  than running a joint model twice. Keep missing tips missing, not box centres.
4. Evaluate box precision/recall, tip error in original pixels and board
  millimetres, and score agreement against annotated tips on held-out images.
  Report missed/hidden tips and errors near scoring boundaries separately.
  Evaluate scoring with known calibration first, then end-to-end calibration,
  so geometry and recognition failures can be distinguished.

No training annotations, dart-specific weights, or measured recognition accuracy
are bundled. The optional adapter fetches external pretrained weights explicitly
on first inference. OCR and automatic orientation correction can follow once tip
detection is evaluated.

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