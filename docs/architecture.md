# dartcv architecture

For the color-based multi-dart work, start with
[resuming development](#resuming-color-based-multi-dart-development), then read
the [agreed design](#agreed-color-based-multi-dart-design). These sections preserve
the decisions from the design interview; continuing does not require access to
the original conversation.

## Purpose and scope

dartcv estimates a planar dartboard's geometry in a photograph, draws the board
model, and rectifies the image. It also provides point-scoring helpers and
synthetic board rendering. It does not currently locate dart tips automatically
or implement a complete live scoring application.

## Components and responsibilities

| Component | Responsibility | Interfaces and dependencies |
| --- | --- | --- |
| [cli](../src/dartcv/cli.py) | Argument parsing, image loading, calibration mode selection, batch iteration, output writing | `main()`, `pick_points()`; calls detection, geometry, overlays, and diagnostics |
| [detect](../src/dartcv/detect.py) | Automatic homography estimation and optional diagnostic capture | `detect_board()` returns `Detection`; uses geometry, overlay transforms, and ring refinement |
| [rings](../src/dartcv/rings.py) | Full-resolution paint-band sampling, common geometry refinement, independent edge fits | `refine_coloured_rings()` returns a homography and `RingBoundary` mapping; uses SciPy optimization |
| [geometry](../src/dartcv/geometry.py) | Millimetre model, calibration, projection, contour-radius queries, point scoring | `RADII`, `SECTORS`, `compute_homography()`, `project()`, scoring helpers; NumPy/OpenCV |
| [edge detection](../src/dartcv/edge_detection.py) | General-purpose Canny edges from grayscale or BGR uint8 images | `canny_edges()`; grayscale conversion, Gaussian smoothing, and OpenCV Canny |
| [overlay](../src/dartcv/overlay.py) | Source-image overlays and top-down warping, optionally using fitted paint contours | `draw_overlay()`, `warp_topdown()`, `topdown_overlay()`; geometry and OpenCV |
| [diagnostics](../src/dartcv/diagnostics.py) | Visual explanations of detection and supplied calibration | `detection_overlays()`, `calibration_overlay()`; consumes detector results |
| [board](../src/dartcv/board.py) | Synthetic visualization in normalized coordinates | Rendering helpers used by the [example notebook](../examples/dartboard_visualization.ipynb); not the detector's metric model |
| [recognition](../src/dartcv/recognition.py) | Recognition data contracts, output validation, backend execution, expected-sector comparison | `DartBox`, `NumberReading`, `DartTip`, `StageResult`, `run_stage()`, validators, `expected_sector()`; no bundled inference backend |
| [recognition overlay](../src/dartcv/recognition_overlay.py) | Draw observed boxes, text, tips, and optional scores without mutating the source | `annotate()`; consumes recognition records, uses OpenCV |
| [dart detection](../src/dartcv/dart_detection.py) | Optional local pretrained, text-guided whole-dart box experiment | `Owlv2DartDetector(image)` returns `DartBox` records; lazy PyTorch/Transformers loading, processor coordinate restoration, OpenCV NMS |
| [classical darts](../src/dartcv/classical_darts.py) | Calibrated, single-dart flight/shaft hypothesis generation, conservative acceptance and diagnostics | `ClassicalDartDetector(H)(image)` returns zero/one `DartBox`; `diagnostic_views()`; NumPy/OpenCV only |
| [dartboard darts pipeline](../src/dartcv/dartboard_dart_pipeline.py) | Skeleton for the multi-dart board-recognition pipeline: stage registration, debug images, per-step timing, and modular replacement points | `DartboardDartRecognitionPipeline()` stores `PipelineResult.steps`, `board_mask`, `dart_mask`, and `timings_ms`; each stage keeps input/output images for inspection |
| [color-based dart pipeline](../src/dartcv/color_dart_pipeline.py) | Separate, incrementally implemented pipeline for board-aware multi-dart scoring | `ColorBasedDartRecognitionPipeline` uses typed replaceable stages; currently implements board localization and normalized-board warping, with later stages injected |
| [color-dart inspection](../src/dartcv/color_dart_inspection.py) | Explicit geometry overlays and board/orientation visualization | `draw_board_geometry()` returns a copied BGR image; `plot_board_preparation()` and `plot_orientation()` return Matplotlib figures without display or file I/O |
| [color-dart orientation](../src/dartcv/color_dart_orientation.py) | Board-style-specific printed-20 template matching on an optional wider rectification | `Printed20OrientationStage(template)`, `extract_printed20_template()`, `unwrap_number_ring()`; NumPy/OpenCV, no OCR, downloads or image I/O |
| [color-dart segmentation](../src/dartcv/color_dart_segmentation.py) | Geometry-derived field sampling, robust HSV-cone models, experimental whole-board alternatives, support-preserving cleanup | `RobustBoardColorSegmentationStage`, `build_field_masks()`, `fit_board_color_models()`, `fit_global_color_models()`, `preserve_dart_support()`; legacy hard cleanup remains available |
| [segmentation evaluation](../src/dartcv/color_dart_evaluation.py) | Same-input color/cleanup ablations and sparse manual-probe evaluation | Nine `segmentation_variants()`, `compare_segmentations()`, explicit reference-frame projection, eligible/excluded patch counts; not whole-mask accuracy |
| [color-board analysis](../src/dartcv/color_dart_analysis.py) | Run the implemented prefix without fake tip/scoring stages | `analyze_board_colors()` returns `ColorBoardAnalysis` with preparation, orientation/segmentation outcomes and stage timings; accepts injected stages |
| [notebook batch](../src/dartcv/notebook_batch.py) | Bounded, sequential final-output contact sheets | `image_batch_figures(paths, process)` yields pages of thumbnails with filenames/status/timing; expected failures are visible tiles |
| [board-inspection notebook](../examples/color_dart_board_inspection.ipynb) | Selected multi-dart photo inspection, detector-artifact views, optional resolution comparison | Calls `prepare_board()` without unfinished recognition stages; presents diagnostics and timings |
| [orientation-inspection notebook](../examples/color_dart_orientation_inspection.ipynb) | Explicit reference-template extraction, selected-photo matching and coverage/annotation inspection | Uses the small manual orientation manifest and optional number-ring normalizer; does not change the existing board-inspection notebook |
| [segmentation notebook](../examples/color_dart_segmentation_inspection.ipynb) | Field masks, model parameters/support, raw/cleaned anomalies and compact batches | Same analysis helper/stage instances as the overview; no automatic tips/scores |
| [combined overview](../examples/color_dart_pipeline_overview.ipynb) | All implemented color stages, plus final-output-only batch inspection | Preparation/orientation/segmentation figures and timings in one notebook; future stages remain explicitly absent |
| [segmentation comparison](../examples/color_dart_segmentation_comparison.ipynb) | Selected-image 3x3 ablation overlays, selectable final-output batches and sparse evaluation | Field/board-union/global fitting crossed with none/hard/support cleanup, identical geometry and frozen annotation frames |
| [recognition notebook](../examples/dart_recognition.ipynb) | Staged single-image orchestration, configuration, plotting, scoring, optional export | Defaults to classical boxes with session reuse keyed by settings/calibration; OWLv2 remains an optional library alternative; number and tip slots remain unset |

## Runtime and data flow

```mermaid
flowchart TD
    CLI[CLI: load BGR image] --> Mode{Calibration mode}
    Mode -->|automatic| Detect[Detector: ellipse, bull, rotation, ECC]
    Detect --> Rings[Colour-ring refinement]
    Rings --> Result[Detection: H, contours, optional diagnostics]
    Mode -->|manual or supplied points| Geometry[Four-point homography]
    Result --> Render[Source and top-down overlays]
    Geometry --> Render
    Result --> Debug[Pipeline diagnostics]
    Geometry --> Calib[Supplied-point diagnostic]
    Render --> Write[CLI: write PNG outputs]
    Debug --> Write
    Calib --> Write
```

The console entry point is `dartcv.cli:main`. An explicit file writes directly
to the selected output directory; a directory writes per-image subdirectories.
Manual calibration uses an OpenCV window; supplied points bypass the window.
Library detection returns data without writing files. See the
[README](../README.md) for CLI options and output names.

The color-based multi-dart pipeline is a separate library entry point, not an
extension of the earlier `DartboardDartRecognitionPipeline` skeleton. It accepts
one in-memory BGR image, detects and normalizes the board, then invokes injected
orientation, segmentation, entry-point, and scoring stages. Those injections
are required; explicit-template orientation and initial board-color segmentation
implementations are available, but entry-point localization and new scoring remain unimplemented.
The pipeline does not fabricate results for them.
`prepare_board()` runs only the shared localization/normalization prefix and
returns a `BoardPreparationResult`, so these stages can be inspected independently.
The complete pipeline calls the same function rather than maintaining a second
execution path. Expected failures retain completed steps and skip dependents.
`analyze_board_colors()` extends this inspectable prefix through the same
orientation/segmentation stage contracts, returning actual maps without requiring
dummy entry/scoring injections. Both new notebooks use it for detailed and batch
runs. It preserves an uncertain orientation estimate rather than silently promote
the result to confidence.

## Contracts and invariants

- Images are OpenCV BGR arrays. Image points are `(x, y)` in pixels.
- The geometry model uses millimetres, with origin at the bull, x right and y
  down. Angles increase clockwise from the centre of sector 20 at the top.
- `H` maps model millimetres to original-image pixels, not the reverse.
  Point scoring uses its inverse. Top-down warping uses
  `model_to_topdown() @ inverse(H)`.
- `Detection.H_initial` is the initial bull-constrained, rotated estimate before
  ECC and ring refinement. `Detection.H` is the final mapping. Diagnostic arrays
  can use different resolutions; their dataclass fields document their frames.
- `Detection.rings` contains fitted contours in original-image pixels. Overlay
  code transforms these for top-down rendering; absent contours remain nominal.
- `ColorBasedDartRecognitionPipeline` uses the board-model-to-source homography
  to create a configurable square view with the board diameter filling the
  canvas. Its run result keeps source-image board-entry coordinates; optional
  diagnostics contain copied stage artifacts, while timings are always returned.
  The scoring-board image excludes the printed-number ring. The optional
  `NormalizedBoardView.number_ring: NumberRingView | None` retains a separate
  wider image, model-to-view transform and source-coverage mask. Request it with
  `partial(normalize_board_view, number_ring_radius_mm=240.0)` as the injected
  normalizer. It uses the same output size but a wider physical extent, without
  changing any scoring-view image/mask/transform. The default does no extra warp.
  Normalization also returns scoring-view `source_coverage`, separate from the
  nominal board disk. Missing source pixels must not train color models or become
  dart evidence. Color modeling fails explicitly for injected legacy views without
  source coverage rather than assume their black padding is observed board.
  Inspection overlays show nominal rings/spokes, not detected wires or numbered
  sector identities. Timing summaries sum stage execution durations and exclude
  file loading, artifact snapshot copies, and visualization.
- Scoring uses nominal `RADII`, not the fitted paint contours. The normalized
  synthetic renderer is not a replacement for the millimetre geometry model.
- Recognition backends return original-image coordinates and confidence in [0, 1].
  `run_stage()` distinguishes missing, failed, and successful-empty results;
  exceptions or validation failures discard all items from that stage. Box IDs
  must be unique; associated tips must refer to a known nearby box. A box centre
  is never substituted for a tip, and expected sector labels never replace OCR.
- Supplied calibration points must be the outer-double intersections at
  `20|1`, `6|10`, `3|19`, `11|14`, in that order.

## Design decisions and trade-offs

- Coarse image segmentation provides a geometric seed, while original-resolution
  bull and paint measurements retain fine spatial information.
- A planar homography describes shared board geometry; independent paint edges
  allow real band widths to differ from nominal wire-based dimensions.
- Optional refinement stages retain usable estimates when measurements are
  unreliable. A successful return is not a guarantee of correct orientation or
  scoring accuracy. See the [pipeline](cv-pipeline.md) for acceptance and fallback rules.
- Detection, geometry, rendering, and file orchestration have separate modules;
  diagnostics explain results without defining the detector's objective.
- The color-based multi-dart pipeline deliberately owns separate scoring
  contracts and board constants; see [ADR 0001](adr/0001-separate-color-dart-pipeline.md).

### Agreed color-based multi-dart design

**Status recorded 2026-10-25:** milestones 1 through 4 are implemented. Everything
marked planned below is a design commitment, not an available recognition
algorithm. Develop and inspect one module at a time rather than deliver the
entire recognizer in one change.

The goal is robust automatic localization and point counting from a single
photograph of a standard 20-sector steel-tip board. Dart bodies may overlap or
extend outside the board; entry points are assumed not to overlap. The expected
dart count is a positive per-run setting, defaulting to three. Priorities are
understandable intermediate outputs, replaceable modules, and measured speed.
There is no committed real-time/video target or numerical latency budget yet.

The color-based idea is based on
[Dart detection with HSV colour classes](../examples/dartboard_dart_detection_algorithm.pdf):
model normal board colors and find anomalous pixels belonging to darts.
Red, green, beige, and black board fields form four color classes; shadows and
lighting variation need robust modeling. Metal wires/reflections can resemble
metal dart parts, so color outliers alone are insufficient. Do not copy the
PDF's bull-dependent board-localization approach: reuse `detect_board()` instead.

| Stage | Current state | Agreed responsibility / future implementation |
| --- | --- | --- |
| Board localization | Implemented | Replaceable `detect_board_stage()` adapter to `detect_board()`; board-model millimetres map to original-image pixels. It supplies geometry, not a physical wire segmentation. |
| Normalization | Implemented | `normalize_board_view()` retains transforms and a nominal board mask; configurable square output, default 1024 px across the scoring board. Optional wider number-ring view retains source coverage; original-image access remains available. |
| Numbered-sector orientation | Implemented initial template matcher; injected stage required | `Printed20OrientationStage(template)` identifies the printed `20` after perspective correction. Explicit board-style template, best alignment, heuristic match strength and trustworthy/uncertain flag. Weak/cropped/obscured evidence can fail or return an uncertain, potentially wrong best guess. |
| Board-color modeling and dart segmentation | Implemented initial model; injected stage required | `RobustBoardColorSegmentationStage` derives field interiors, fits four robust per-image color distributions, excludes expected wires, and produces a heuristic normalized-board dart-likelihood map. Not a validated dart classifier. |
| Board-entry-point localization | Planned; injected stage required | Separate non-overlapping entry points even if dart bodies form merged blobs. Return estimates in original-image pixels with confidence/provenance, not box centers. |
| Scoring | Planned; injected stage required | Independently map board-entry points to sector/ring/points; preserve best guesses and explicit uncertainty. Existing scoring helpers are not the new implementation. |

**Modularity and execution:** use typed injected callables/objects with explicit
stage outcomes, not an inheritance/plugin registry framework. Preserve the
earlier `DartboardDartRecognitionPipeline` rather than replace it. The new
pipeline owns independent types, stage helpers, board constants/sector ordering,
and future scoring math; sharing the existing detector is intentional.
The standard geometry constants are already present in `color_dart_pipeline.py`;
score mapping is not. Avoid silently coupling it back to `recognition.DartTip`
or the existing scorers.

**Orientation is not just perspective correction:** alternating ring colors
repeat every 36 degrees and cannot uniquely identify sector 20.
`detector_rotation_deg` is not verified numbered-sector alignment. An unreadable
`20` should not silently become a confident upright-board assumption. Preserve a
best-guess alignment/score with orientation uncertainty when evidence supports
a guess; missing localization evidence still permits an unknown result.

Milestone 3 keeps the existing `OrientationStage(board_view, *, diagnostics)`
contract. Its optional wider view covers a 240 mm radius by notebook default;
matching unwraps radii 194–236 mm into 85 radial rows and 1440 angular columns
(four samples per degree), clockwise from -180 degrees with the outer edge at
row zero. These radii fit the inspected board, not every board manufacturer's
number ring. Source coverage is propagated separately from image brightness;
partially missing template windows cannot supply matches.

The template is an explicit BGR patch extracted from a **manually identified**
reference `20`, with a default 14-degree width and radii 202–228 mm.
`extract_printed20_template()` checks coverage; the stage caches local-contrast
template features at scales 0.9/1.0/1.1. At runtime it unwraps the number ring,
subtracts local Gaussian brightness, performs normalized correlation, and ranks
up to five hypotheses separated by at least 18 degrees. Matching crosses the
360-degree seam; it is not restricted to the detector's initial upright guess.
The reference template is caller supplied: the library reads no images/files
and bundles no general-purpose font/OCR model.

Provisional defaults: correlation below 0.25 means no supported match; otherwise
retain the best guess. Set `confident=True` only at strength >=0.65 and a gap
>=0.10 from the next distinct candidate. Strength is **not a probability** and
these thresholds have not been calibrated across boards. Occluded labels can
select a different numeral: do not use an uncertain guess as verified alignment.
`correction_deg` is the negative of the matched clockwise label angle, wrapped
to [-180, 180), so it describes a **clockwise** correction in the rectified
model. OpenCV's affine-rotation helper uses the opposite sign. No image or
homography is silently rotated; downstream segmentation/scoring must apply this
estimate consistently when they are implemented.

With diagnostics enabled, orientation returns the wider input and its model
transform, number strip and coverage, template, angular response, and candidate
`[clockwise_angle_deg, correlation]` rows. `plot_orientation()` marks the best
label without claiming to detect physical wires. Internal timings separate
unwrapping from matching; normalizer execution includes the optional wider warp,
while template reference loading/preparation is caller-owned setup, not hidden
per-photo work.

**Required orientation fallback (planned, not yet implemented):** when a dart
fully or partially covers `20` and its match is unreliable, identify a visible
neighboring `5` or `1` and derive the 20-sector position from the standard sector
order. With angles clockwise from model top, `angle_20 = angle_5 + 18 degrees`
or `angle_20 = angle_1 - 18 degrees`, wrapped consistently. If both neighbors
are available, check that their derived positions agree; conflicting or weak
evidence must remain explicitly uncertain rather than force a confident result.
Expose which label supported the estimate in diagnostics and cover full/partial
20 occlusion in focused tests. The current matcher does not provide this fallback.

**Color-model sampling:** pool interior pixels only from geometry-identified
fields of the corresponding color, giving four per-image distributions, not one
model over all board pixels or separate distributions for every field. Apply
the expected field-color model in that field. Exclude wire margins and robustly
reject contaminated samples such as darts.

Milestone 4 implements `build_field_masks()` in model millimetres, using
`angle_from_20 = model_clockwise_angle + orientation.correction_deg`.
Sector 20 has black single fields and red colored rings; alternating sectors,
the red inner bull and green outer bull determine the four expected classes.
Sample interiors exclude a configurable 2.5 mm margin around nominal radial
and circular wires, plus missing source pixels. Models pool all interiors of
the same color; they are not fitted independently to each wedge.

The segmentation iteration separates paint alignment from numbered orientation:
when `orientation.confident=False`, the stage retains the detector's paint
alignment (zero additional correction) instead of rotating fields by an
unreliable numeral guess. This relies on the detector homography aligning the
alternating paint modulo 36 degrees; injected normalizers/localizers must
preserve that convention. Diagnostics explicitly record
`color_alignment_correction_deg` and `color_alignment_uses_detector`.
The original numbered-orientation uncertainty is NOT cleared, and scoring
must not infer a verified 20 from successful color modeling. Confident but
incorrect numeral matches and incorrect geometry can still spoil field masks.

`board_color_features()` maps OpenCV HSV to
`[s*v*cos(h), s*v*sin(h), v]` with s/v in [0,1] and h wrapped continuously.
Hue contributes little for dark/achromatic pixels. Each color uses at most
12000 deterministic, evenly selected interior samples, a coordinatewise median,
MAD scales multiplied by 1.4826, and three rounds of joint-distance trimming at
4.5 robust units. Scale floors are [0.025, 0.025, 0.05]. At least 64 retained
samples per color are required by default; missing support is an explicit
`InsufficientColorSamples` stage failure. Unexpected exceptions propagate.
These provisional choices assume dart contamination is a minority of each
pooled class; they do not solve severe illumination gradients or majority
occlusion. Centers, scales, and sampled/retained counts are diagnostic arrays.

For each observed pixel, compare only its expected field-color model. Raw
anomaly strength is `clip((robust_distance - 3) / 5, 0, 1)`, not a calibrated
dart probability. Expected color IDs are red=0, green=1, beige=2, black=3,
outside/unobserved=255. The returned `DartLikelihoodMap` remains float32 in [0,1].

That is the default `color_assignment="field"`. Two explicitly experimental
alternatives are available at the user's request:
`"board_union"` fits the same field-interior models but compares each pixel
against the closest of all four models, isolating the value of expected-field
matching. `"global"` fits four unlabeled modes to the entire observed scoring
board, including wires, without field-based training groups. It samples at most
16000 pixels (or enough for the configured four minimum supports), uses
deterministic K-means (three initializations, seed zero), then robustly trims
each cluster and compares pixels to the closest resulting mode. Cluster labels
are `mode_0` etc., not certified red/green/beige/black. Low support/fewer than
four distinct colors fails explicitly. It uses the existing scikit-learn
dependency, imported lazily; first-fit timing includes that import.
Whole-board matching can suppress dart colors resembling any paint class, and
global fitting can absorb a dart's appearance into its distribution. Do not
promote it to the default simply because its output has fewer foreground pixels.

**Metal suppression:** start with geometry-derived expected wire regions plus
edge/small-component cleanup; keep this replaceable inside segmentation.
Do not assume `detect_board()` returns physical wires, and do not treat metal
as a fifth ordinary board-color class. The PDF-inspired operations include HSV
features, robust outlier classification, edge-preserving filtering, morphology
and local regrowth, minimum-component filtering, hole filling, and optional
GrabCut refinement. These are candidate internal operations, not a requirement
to run every operation or expose each as a top-level pipeline stage.
Expose internal named artifacts/substep timings as those operations are added.

Milestone 4's original `suppress_wire_artifacts()` zeros a 1 mm band around nominal
wires, opens >=0.5 anomaly support with a 3x3 ellipse, and retains components
of at least 3 mm² (converted to pixels at the model origin). Inject an alternative
`cleanup` callable into the segmentation stage to replace this policy. It can
remove thin shafts and contact evidence, and residual reflections remain on the
inspected real photo. Preserve `raw_color_anomalies` for subsequent tip work;
do not treat cleaned components as separate darts or infer their contact points.
`plot_color_segmentation()` shows expected colors, sampling masks, expected
wires, raw/cleaned maps and the final overlay. Internal timings separately cover
mask/features, fitting, anomaly evaluation and cleanup. Uncertain orientation
is retained on the analysis result and prominently labeled in figures/batches.

The **revised default** is `preserve_dart_support()`: retain >=0.2 raw-anomaly
components of at least 3 mm² seeded only in sampling interiors (normally
>2.5 mm from nominal wires). Do not apply an opening. Restore original scores
in the excluded wire-margin area only within a single 3 mm dilation of retained
interiors. Restored wire pixels never seed another growth iteration, so an
entire connected grid cannot flood from one dart. This restores evidence, not
synthetic filled-in dart pixels; source coverage remains mandatory. Physical
area/reach use model-to-view scale at the bull. Thin unsupported components,
color-matching dart regions and wholly off-board parts can still be missed.
Nearby reflections may be restored as well. `keep_color_anomalies()` provides
the no-cleanup comparison; `suppress_wire_artifacts()` remains the hard-cleanup
comparison. All retain the existing injected cleanup callable signature.

**Board-entry and score semantics:**

- The target is the board-entry/contact point, not the visible metal tip end,
  dart body center, or bounding-box center. A hidden contact may be inferred from
  geometrically constrained shaft/barrel evidence; mark it inferred/uncertain.
  Insufficient evidence yields an unknown point, never a fabricated location.
- Return one slot per expected dart. Order localized points by source `(x, y)`,
  then unresolved slots. This is single-image ordering, not persistent tracking.
- A score estimate includes sector, scoring ring/multiplier, and total points.
  Boundary ambiguity retains a best guess plus uncertainty, rather than forcing
  certainty or discarding every guess. The board-space boundary margin is
  configurable (currently 2.5 mm) and must be calibrated on labeled photos.
- Allow multiple concurrent uncertainty reasons: inferred/unknown entry,
  orientation, score boundary, or unavailable score. A confidently localized
  entry outside the scoring surface is a `MISS` worth zero; an unknown entry
  has no score. Do not substitute zero for missing evidence.
- Aggregate all expected score estimates only when every expected slot has one;
  otherwise omit the total. Preserve total completeness and uncertainty flags
  independently: a complete best-guess total can still be uncertain.
- Initially focus on board-associated darts; do not add a general detector to
  find objects wholly outside the board. Such a dart may remain unresolved.
  Preserve original-image access so off-board evidence is not irretrievably
  cropped away.

**Observability and failures:** accept one non-empty BGR `uint8` NumPy image per
run; the caller handles loading and color conversion. Return structured timing
data on every run and a pure `format_timing_summary()` string formatter. Store
diagnostics per run rather than mutable latest-run state on the pipeline.
Diagnostic capture is opt-in, with named input/output array snapshots; no
automatic windows, file writes, or printing. Injected stages must not mutate
inputs. Expected algorithm failures produce explicit partial results and skip
dependent stages, preserving completed information; unexpected exceptions
propagate. Required unfinished stages must be injected, not represented by
success-shaped placeholders.

### Resuming color-based multi-dart development

1. Read this section, the design above, the [glossary](../GLOSSARY.md), and
   [ADR 0001](adr/0001-separate-color-dart-pipeline.md). Check `git status` before
   editing: substantial unrelated staged/unstaged work may be present. Do not
   revert, stage, or commit that work implicitly. **Commit progress after each
   completed stage**, including its related documentation, once verified.
   The initial checkpoint includes all then-staged work with explicit user
   approval; that approval does not extend to unrelated changes in later stages.
2. **Milestone 1, done:** separate contracts/orchestration, real board-localization
   adapter and normalization, injected future stages, partial failures,
   per-dart/aggregate result shape, timings, and optional diagnostic snapshots.
   Start at [color_dart_pipeline.py](../src/dartcv/color_dart_pipeline.py).
3. **Milestone 2, done:** board-only `prepare_board()` uses the same prefix as the
   full pipeline. [color_dart_inspection.py](../src/dartcv/color_dart_inspection.py)
   draws nominal geometry and returns a six-panel figure; it does not display
   or save automatically. Open
   [color_dart_board_inspection.ipynb](../examples/color_dart_board_inspection.ipynb)
   using the project environment (`uv sync --group dev`), select `IMAGE_INDEX`,
   and rerun from **Run preparation**. Inspect rings, bull, spokes, detector/ECC
   artifacts and timings. Resolution comparison is explicitly opt-in.
4. **Milestone 3, done:** optional wider view and source coverage, explicit
   reference-template extraction, replaceable `Printed20OrientationStage`,
   candidate/response diagnostics and substep timings. Open
   [color_dart_orientation_inspection.ipynb](../examples/color_dart_orientation_inspection.ipynb).
   Run setup/reference once, then change `IMAGE_INDEX` and rerun from selection.
   Index 1 has a visible `20`; index 0 is obscured and returns an uncertain,
   wrong best guess. The original board-inspection notebook is unchanged.
   Inject both its `NORMALIZER` and `ORIENTATION_STAGE` into the full pipeline
   when real downstream stages become available.
   **Outstanding orientation follow-up:** implement the `5`/`1` fallback above
   for a fully or partially dart-covered `20`.
5. **Milestone 4, done:** geometry-derived field-interior masks, same-color
   sampling, robust color models and an initial dart-likelihood map with
   replaceable cleanup. Open
   [color_dart_segmentation_inspection.ipynb](../examples/color_dart_segmentation_inspection.ipynb)
   for detailed mask/model/map inspection, or
   [color_dart_pipeline_overview.ipynb](../examples/color_dart_pipeline_overview.ipynb)
   for all implemented color stages in one place. Both run the explicit reference
   setup once, then `analyze_board_colors()` on a selected photo. No real tip or
   independent scoring stage is supplied yet.
   **Segmentation iteration, done:** reproduced shaft loss and noisy masks from
   unreliable numbered orientation; revised the default cleanup/color alignment
   policy and added whole-board alternatives. Open
   [color_dart_segmentation_comparison.ipynb](../examples/color_dart_segmentation_comparison.ipynb).
   Its selected-image matrix compares nine variants; `BATCH_VARIANT` selects the
   compact batch strategy; `RUN_PROBE_EVALUATION` reruns the six-photo sparse
   checks with initial regression/challenge results separated. The default
   selected image is a deliberately noisy challenge case. The overview and
   segmentation notebook have optional `RUN_ABLATION`
   cells without replacing their existing user-selected settings.
6. **Next milestone:** entry-point localization, then independent scoring incrementally.
   Their algorithm details remain open; apply the agreed uncertainty, miss,
   ordering, and aggregate contracts rather than inventing new semantics.

Every existing feature notebook, including the older experiments, now has a
separate **Compact batch** cell with `RUN_BATCH=False` by default. Choose
`BATCH_PATHS` (three photos by default; remove `[:3]` for the directory) and enable
it to inspect only final outputs. Pages contain at most six downsized thumbnails
with filename/status/elapsed processing time, so full-resolution intermediate
histories are not retained per image. To investigate, copy a bad filename into
that notebook's single-image selection. Color notebooks reuse the configured
stage objects; the legacy recipes expose explicit `BATCH_*` settings that must
match the detailed settings for a fair comparison. Batch times include callback
I/O and rendering preparation; they are not pure algorithm timings.

Special cases: the visualization notebook batches render sizes, not photos.
The `find_single_dart.ipynb` scratch notebook batches its supported board stage,
not its incomplete unrelated experimental cells. The raw-board-difference
batch uses a separately calibrated fixed reference and its final binary
candidate recipe. The scorer notebook batches the older prototype, with temporary
annotated images cleaned up, not the new color scorer. Classical-recognition
batching uses the default local backend, without optional model downloads or
transferring manual points between images. Expected read/detection failures
remain visible; unexpected programming errors are not swallowed.

The small [orientation manifest](../data/annotations/color_dart_orientation.json)
records one template reference, two visible-label checks with approximate
manually inspected source-image centers, and one occluded-label check.
The centers were visually selected independently of the matcher, with an
estimated 24 px uncertainty. This is a same-board development fixture, not an
accuracy benchmark; the reference is not counted as validation.
The [segmentation probe manifest](../data/annotations/color_dart_segmentation.json)
adds independently visually selected flight/shaft/background/wire points on four
initial regression photos and two additional noisy challenge photos (no wire
probes are labeled on those latter two). Its 768 px reference rectifications and
`number_match` outcomes are frozen
development inputs, NOT manually verified board homographies/orientation truth.
`evaluate_sparse_probes()` projects these patches into a current rectification and
excludes unobserved/out-of-scoring-view patches. It does not infer a full dart mask.
Ground-truth board geometry, full segmentation masks, dart entries and scores
remain unannotated. Extend annotations
as each later module needs them. Keep the imagery local/DVC-managed and do not publish photo metadata or
embed the full dataset into notebooks. Update this status and the relevant
contracts/limitations after each milestone so the next session can resume
without rediscovering the design.

## Dependencies, data, and artifacts

[Project metadata](../pyproject.toml) defines Python support, dependencies, the
console script, and the `uv_build` backend. NumPy/OpenCV support image geometry;
SciPy supports robust ring optimization; Matplotlib supports synthetic rendering.
DVC with Google Drive support manages the image dataset; its authentication
dependencies are not part of the detection algorithm.

The `dart-detection` extra adds PyTorch, Torchvision, Transformers, and Pillow.
The optional adapter is independent of board calibration and preserves original
perspective. Its first inference loads pinned external weights into a local
Hugging Face cache; imports and synthetic tests never fetch weights. Images stay
local. Core board commands do not require these dependencies.

The classical detector needs no additional dependencies. It consumes the board
homography as a search/paint prior, never as a warp of the dart. Its last-attempt
diagnostics store working-resolution arrays with an explicit source scale;
returned boxes use original pixels. It has no file-writing or scoring side effects.

Image contents are managed as one DVC dataset, referenced by
[data/images.dvc](../data/images.dvc). Output images are generated artifacts,
not input data. Credentials stay local and must not enter documentation or Git.
See the [README data workflow](../README.md#pictures-on-google-drive-uv--dvc).

The multi-dart input directory is **`data/images/multi_dart`**, not
`data/multi_dart`. At the 2026-10-25 inspection it contained 17 real JPEG photos,
each 6144 by 8160 pixels, with no annotation sidecars. Boards are visible in the
photos, with overlap/cropping and varying label visibility; do not infer exact
dart counts or ground-truth scores from this inventory. Full-resolution
diagnostic copies are memory-heavy, so the notebook retains only the selected
run and downsizes display previews, not processing inputs.

## Validation and limitations

- [Homography tests](../tests/test_homography.py) cover synthetic projection,
  scoring, detection, CLI outputs, diagnostics, and bull constraints.
- [Ring tests](../tests/test_rings.py) cover colour-band fitting, refinement,
  occlusion/fallback behavior, and overlay integration.
- [Edge-detection tests](../tests/test_edge_detection.py) cover grayscale/BGR
  inputs, output shape/type, input immutability, and parameter validation.
- [Recognition tests](../tests/test_recognition.py) cover backend states, output
  validation, tip associations, sector comparison, and non-mutating annotation.
- [Classical dart tests](../tests/test_classical_darts.py) cover synthetic flights,
  shafts, paint rejection, ambiguity, hue wrapping and coordinate restoration.
- [Color pipeline tests](../tests/test_color_dart_pipeline.py) cover board
  normalization, stage injection, partial failures, optional diagnostics, and
  per-run timing/result contracts, plus headless inspection rendering, BGR/RGB
  display conversion, and non-mutating geometry overlays. They do not establish
  dart-recognition accuracy.
- [Orientation tests](../tests/test_color_dart_orientation.py) cover unchanged
  scoring views, wider-view source coverage, signed rotation/seam handling,
  perspective, missing/cropped evidence, duplicate-label ambiguity, pipeline
  wiring and non-mutating headless visualization. Three annotated same-board
  photo checks run when the optional DVC-managed inputs are present; otherwise
  that test skips.
- [Segmentation tests](../tests/test_color_dart_segmentation.py) cover signed field
  assignments, bull/ring masks, hue continuity, robust minority-contamination
  rejection, missing-source exclusion, wire/component cleanup replacement,
  explicit sample failures and partial-prefix orchestration, thin shaft crossings,
  bounded grid restoration, unreliable number matches, and whole-board tradeoffs.
  Optional real-photo sparse checks use the fixed annotation frames.
- [Segmentation evaluation tests](../tests/test_color_dart_evaluation.py) cover
  reference-frame projection, missing-coverage exclusions, explicit failures and
  the nine-variant comparison surface.
- [Batch tests](../tests/test_notebook_batch.py) cover bounded lazy pages, BGR/RGB
  thumbnail conversion, visible failures and propagation of unexpected bugs.
- Run `uv run --group dev pytest`; use one explicitly selected real image for a
  visual check unless a batch evaluation is requested.
- Synthetic tests and diagnostic fit residuals do not establish real-dataset
  accuracy. Colour dependence, occlusion, orientation ambiguity, and nominal
  scoring boundaries remain important limitations.

### Recorded development checks (2026-10-25)

Milestone 1's real-photo smoke run returned board localization and normalized
views on all 17 multi-dart images. Mean board-detection time was approximately
0.92 seconds per full-resolution image on that development machine. This is a
single exploratory run, not a performance guarantee or verification that all
homographies are geometrically correct.

For milestone 2, the board-inspection notebook's code cells were executed on
`PXL_20261018_154231645.jpg`, including its optional 512/1024/1536 px comparison;
the six-panel output was rendered and inspected. The targeted pipeline and
homography command passed 40 tests:

```bash
uv run --group dev pytest -q tests/test_color_dart_pipeline.py tests/test_homography.py
```

For milestone 3, the orientation notebook was executed headlessly on
`PXL_20261018_154237729.jpg` using the explicit reference
`PXL_20261018_154400079.jpg`; its four-panel figure was rendered and inspected.
The visible label matched confidently at a clockwise correction of 0.25 degrees
(approximately 0.23 degrees from the manually inspected source center).
The other visible annotated photo also matched within the 2-degree tolerance;
the obscured-label photo returned an uncertain, wrong candidate, as documented.
The targeted command passed 55 tests:

```bash
uv run --group dev pytest -q tests/test_color_dart_orientation.py tests/test_color_dart_pipeline.py tests/test_homography.py
```

An exploratory notebook run measured approximately 1.1 ms to unwrap and 7.5 ms
to match at the default 1024 px normalization, excluding reference preparation,
board detection, I/O and rendering. These are development observations, not
latency guarantees or broad orientation accuracy measurements.

For milestone 4, both new detailed notebooks and all existing feature batch
cells were executed headlessly. Each photo batch used the first two available
inputs in its selected directory; the renderer batch used its three sizes.
One real segmentation figure on `PXL_20261018_154237729.jpg` was rendered and
inspected: darts are prominent, but wire residuals remain. On that photo,
segmentation took approximately 120 ms at 1024 px in exploratory runs; this
excludes board detection, reference setup, file I/O and display. No segmentation
ground-truth masks have been annotated, so this is not an accuracy benchmark.
The repository suite passed **122 tests, 1 skipped**:

```bash
uv run --group dev pytest -q
```

No accuracy claims for entry-point localization or new scoring follow from
these checks: those algorithms remain unimplemented.

### Segmentation iteration: recorded findings (2026-10-25)

The user prioritized retaining visible flights and thin shaft evidence, not
merely displaying a sparse mask. Four locally inspected photos span obscured
`20`, separated shafts, overlapping shafts, and overlapping white flights.
A cached deterministic repro showed the original cleanup lowering a clearly
visible shaft patch at [355,271] in the second photo from a raw mean of 0.497
to 0.093. The revised result is 0.485. Synthetic regressions separately reproduced
wire-crossing erasure and a wrong weak numeral rotating the field masks.

The table below reports **positive sparse patches / eligible patches**, using
a mean score threshold of 0.5. Higher is better for flights/shafts; lower for
board/wire backgrounds. All comparisons use identical fixed 768 px geometry.
One outside-scoring-view flight patch is excluded; these are development probes,
not full-image recall/precision, held-out accuracy or IoU.

| Variant | Flights retained | Shafts retained | Board false positives | Wire false positives |
| --- | --- | --- | --- | --- |
| Original milestone-4 baseline (before both fixes) | 11/19 | 30/41 | 2/40 | 3/32 |
| Field / no cleanup | 17/19 | 35/41 | 9/40 | 2/32 |
| Field / hard cleanup | 10/19 | 31/41 | 0/40 | 0/32 |
| **Field / support (new default)** | **17/19** | **35/41** | **0/40** | **0/32** |
| Any field-trained board color / no cleanup | 13/19 | 23/41 | 3/40 | 1/32 |
| Any field-trained board color / hard cleanup | 6/19 | 14/41 | 0/40 | 0/32 |
| Any field-trained board color / support | 13/19 | 22/41 | 0/40 | 0/32 |
| Whole-board modes / no cleanup | 4/19 | 18/41 | 2/40 | 0/32 |
| Whole-board modes / hard cleanup | 1/19 | 11/41 | 0/40 | 0/32 |
| Whole-board modes / support | 4/19 | 18/41 | 0/40 | 0/32 |

Conclusion for this subset: geometry-aware color assignment is useful; accepting
any board color loses real dart pixels, and fitting the whole-board distribution
loses still more. Blind grid removal also cuts shafts/flights. Interior-seeded,
finite evidence restoration is a better compromise here, not a universal winner.
Despite zero false positives on these sparse background probes, visual inspection
still shows residual ring/reflection noise; do not claim that only darts remain.
The unresolved next segmentation questions are color-matching dart regions,
residual near-dart wire contamination, and broader lighting/board styles.

Exploratory warm stage execution at 768 px was approximately 73–75 ms for
field/support versus 104–111 ms for global/support on these four cached inputs,
with BLAS/OpenMP threads limited to one. First global fitting cost about 298 ms
including its lazy import. These figures exclude decoding, board detection and
display, are not directly comparable with the older 1024 px measurements, and
are not performance guarantees.

The revised default was then run across **all 17 multi-dart photos**, and its
contact sheet was visually inspected. This exposed substantial background
stripes/reflections in additional photos: the four initial cases are not
representative enough to call segmentation solved. Two deliberately noisy
cases were added to the manifest, with separate reporting rather than folding
them into the already-fixed acceptance gates:

| Additional challenge variant | Flights retained | Shafts retained | Board false positives |
| --- | --- | --- | --- |
| Field / support | 11/12 | 10/18 | 12/20 |
| Any field-trained board color / support | 5/12 | 5/18 | 3/20 |
| Whole-board modes / support | 4/12 | 4/18 | 0/20 |

These challenging photos make the tradeoff explicit: a global model produces a
cleaner background but discards much more dart evidence. No tested strategy
fully separates only darts. Keep field/support as the evidence-preserving
default, not as a certified best method for every image.

Follow-up probes found ordinary adjacent paint assigned to the wrong field and
shadow/texture pixels outside narrow robust distributions. A small constant
paint-angle search reduced noise in one case but not the other; stricter
novel-color/thickness seeding reduced flight retention without resolving the
noisiest case. Neither prototype was promoted into the library. Further work
needs stronger geometric/photometric modeling or visible shaft/flight evidence,
not simply a more destructive morphology threshold. The planned entry-point
stage must retain access to raw evidence, not assume the cleaned map is complete.

The final comparison notebook (selected image, compact batch, and six-photo
evaluation) and the two existing notebooks' optional ablation cells were
executed headlessly. The full repository suite passed **131 tests, 1 skipped**.
The minimized real-photo shaft repro is covered by the optional sparse-photo
test; synthetic crossing/uncertain-orientation regressions run without DVC data.
Benchmark stage execution with diagnostics off and repeated representative
inputs before choosing a latency target. Compare geometric quality as well as
runtime; fast, successful execution is not sufficient evidence of correctness.

### Notebook compatibility follow-up (2026-10-25)

Saved `color_model_names` and `ColorAssignment` failures were reproduced by
retaining the previous segmentation module/result in a live notebook kernel;
the five color notebooks ran successfully with the current code in fresh
kernels. All notebooks now enable IPython autoreload and explain the required
**Restart Kernel / Run All** after API or dataclass changes. Autoreload is not
a migration of old results: the segmentation display cells explicitly reject
old diagnostic schemas with recomputation instructions instead of a `KeyError`.
The four number-ring notebooks resolve the repository source path before
importing `dartcv`, consistently from the root or `examples` directory.

Fresh-kernel execution also exposed an older scoring notebook's import of
`dart_scorer.polar(x, y)` where its examples require
`geometry.polar(radius_mm, angle_deg)`. Only the notebook import was corrected;
neither scorer's coordinate convention was changed.

`tests/test_notebooks.py` now exercises all 12 notebooks' default cells in fresh
IPython subprocesses, starting in an isolated `examples` directory with local
data/source links. Generated output stays in temporary directories, and missing
DVC photographs skip image runs. Optional batch/ablation/probe switches are not
enabled by this smoke check. The same file covers the scoring example assertions
and explicit stale-analysis rejection without requiring photographs.
The targeted suite passed **15 tests**.

### Source-photo dart annotation tool (2026-10-25)

`python -m dartcv.annotate` serves a small browser editor on loopback only,
using the standard-library HTTP server and bundled HTML/CSS/JavaScript (no
runtime dependencies added). `AnnotationStore` owns source decoding, identity
checks, validated per-photo JSON, atomic replacement, and optimistic revisions
to prevent overwriting another tab. The server serves only the selected image
directory and fixed UI assets; same-origin/Host checks and a session token
protect write requests. Use one server per annotation directory.

`dart_annotations.py` defines schema-version-1 labels, parsing, source-label
projection, visible-mask evaluation and error plotting. Source coordinates
were chosen over normalized coordinates so detector errors cannot change the
annotation ground truth and off-board evidence remains represented. Photographs
are OpenCV-decoded for both display and pipeline, with decoded dimensions and
file SHA-256 stored; annotation resume/evaluation rejects changed sources.

Optional per-dart manual `score` labels store sector number and single/double/
triple multiplier (default single). Missing/null scores remain unknown, preserving
existing schema-version-1 files. Miss is 0/single; bull is 25/single or 25/double.
Derived points are displayed, not duplicated on disk; score labels are independent
of tip/contact and excluded from visible-mask metrics. Editor history and saves
include scores. Controls have native hover descriptions and focusable `?`
context-help boxes, including explicit guidance that shaft/barrel polygons include
the visible metal tip while the exact point records its endpoint.

The default image root is `data/images`; a fixed allowlist of direct image
subfolders drives Folder -> Photograph -> Dart selection. A photo-containing
root is also selectable. Every folder has a separate `AnnotationStore`, with
folder-qualified API requests for listing, loading, pixels and saves; unsaved
changes are confirmed before switching. Existing multi-dart labels retain
their flat location, while other child datasets use output subdirectories to
avoid same-filename collisions. Supplying a single photo folder via `--images`
keeps the original flat-output behavior. Arbitrary browser-supplied paths are
not accepted.

Polygon validation permits crossing/self-touching edges and repeated vertices,
including bow-tie shapes whose signed contour area cancels to zero. It still
requires finite in-image coordinates and three non-collinear vertices. Browser
fill/hit testing uses even-odd semantics to match OpenCV mask rasterization;
separate region polygons remain unioned.
The editor marks self-intersections/touches with optional yellow circles and
a per-selected-dart polygon summary, without blocking saves. Collinear edge
overlaps mark their endpoints; intersections between distinct polygons are not
warnings.

Tip annotations additionally allow `approximate` with source coordinates for a
hidden/unclear but estimated endpoint. A separate placement button and dashed
circle distinguish it from `exact`; dragging preserves its status, and unknown
still means null coordinates. The circle is not an uncertainty bound. Visible-mask
evaluation ignores all tip labels; future localization metrics must distinguish
approximate labels from exact ground truth. Older exact/unknown files stay valid.

Each dart has visible polygons classified as flight, shaft/barrel, or uncertain
part; optional complete inferred silhouettes are separate. A physical tip is
exact or hidden/unknown; embedded/not-embedded/unknown contact determines
whether it is a board entry point. Ignore polygons identify genuinely
ambiguous foreground. Draft/completed is explicit, with per-dart visible-region
checks and a human assertion of exhaustive labeling at completion.

The browser implements polygon/vertex editing, zoom/pan, per-dart grouping,
tip placement, copy-to-inferred, ignore regions, undo/redo, explicit save,
save-and-next, unfinished navigation and dirty-state warnings. Save conflicts
offer download-before-reload recovery. No algorithm proposals or automatic
scores contaminate labels. No annotation save performs a git commit.

The comparison notebook's opt-in `RUN_LABEL_EVALUATION` projects completed
visible polygons into the existing scoring view and compares all nine
variants. It reports dart recall ("dart retained"), precision ("detected
pixels that are darts"), part recall, IoU and source-label coverage. It excludes
ignore regions, unobserved/out-of-board pixels, inferred silhouettes and draft
photos; empty denominators are unavailable. The overlay distinguishes
retained darts, misses, noise and excluded coverage. Exact tips are stored for
future localization evaluation, not scored against an unimplemented stage.
Existing sparse-probe manifests remain unchanged.

Tests in `test_dart_annotations.py` and `test_annotate.py` exercise round trips,
invalid labels, overlaps/projection/coverage, missing denominators, source
identity, notebook integration, editor state history, HTTP loading/saving,
conflicts and origin restrictions. A local headless Chrome check exercised
actual canvas drawing, vertex dragging, undo/redo, inferred copying, ignore
regions, tip/contact marking, completion, save/navigation and reload without
adding browser-test dependencies. Usage and labeling instructions are in
[`dart-annotation.md`](dart-annotation.md). The full suite passed **157 tests,
1 skipped**. The real CLI was also started and queried against all 17 local
multi-dart photos, including full-resolution source metadata, then stopped.

## Related documentation and maintenance

- [README](../README.md): commands, outputs, setup, and data workflows.
- [Annotation protocol](dart-annotation.md): local editor, visible/inferred
  labeling rules, JSON coordinates and evaluation semantics.
- [CV pipeline](cv-pipeline.md): algorithm stages, frames, acceptance rules,
  diagnostics, and technical limitations.
- [Glossary](../GLOSSARY.md): canonical board-entry/orientation/field terminology.
- [Separate color-pipeline ADR](adr/0001-separate-color-dart-pipeline.md):
  rationale for independent contracts and scoring geometry.
- [Architecture template](templates/architecture.md): required structure for this
  document; update both when that structure changes.

Update this document when module boundaries, dependencies, interfaces, data flow,
or contracts change. Keep detailed algorithm parameters in the pipeline document
and code rather than duplicating them here.