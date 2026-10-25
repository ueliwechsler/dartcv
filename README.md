# dartcv

WIP: Dart computer vision framework enabling a smart phone to count your darts score automatically.

## Developer documentation

- [Architecture](docs/architecture.md): components, data flow, and coordinate contracts.
- [Architecture template](docs/templates/architecture.md): required structure for the living architecture document.
- [Technical CV pipeline](docs/cv-pipeline.md): detection stages, refinement, fallbacks, and limitations.
- [Agent instructions](AGENTS.md): documentation maintenance and verification rules for coding agents.

## Board detection and diagnostic images

For a quick single-image check, run
`uv run dartcv data/images/raw_board/IMG_2092.jpg --out out/IMG_2092`.
An explicit image path writes directly into the supplied output directory.
Running `uv run dartcv --out out` instead processes the entire directory
`data/images/raw_board/`, writing results into `out/<image-stem>/`.

Alongside `overlay.png`, `topdown.png`, and `topdown_overlay.png`, automatic
detection writes these numbered pipeline visualizations:

| Image | What it shows |
| --- | --- |
| `01_colour_mask.png` | Red/green segmentation after morphological cleanup, highlighted over a dimmed source image. |
| `02_outer_ellipse.png` | Sampled outer boundary of the largest colour component: retained points in green, rejected points in red, fitted ellipse in yellow. |
| `03_initial_alignment.png` | Initial board model after fitting the ellipse, applying the measured bull constraint when available, and estimating rotation from the alternating red/green double-ring pattern. |
| `04_ecc_edges.png` | Actual image gradients and cyan wire-template gradients at the last attempted ECC scale, shown in the initial top-down frame before refinement. Off-board areas are masked out. The caption reports ECC acceptance and its correlation before the final bull constraint. |
| `05_calibration.png` | Colour-coded **20/1, 6/10, 3/19, and 11/14** wires, with crosses at their outer-double-ring intersections. Grey circles show the initial estimate and connecting lines show the change after refinement. |
| `06_bull_alignment.png` | Close-up comparing the ellipse-only seed with the final constrained homography. Cyan marks the independently measured red boundary; coloured rings show the model. Centre residuals are in original-image pixels relative to colour detection, not annotated ground truth. If detection is unreliable, a caption reports that no bull constraint was applied. |
| `07_ring_alignment.png` | Double/treble close-ups before and after colour-band refinement. Orange shows the previous nominal rings; green shows independently fitted inner and outer paint boundaries, excluding grey metal. Reported residuals measure agreement with sampled colour transitions, not annotated ground truth. |

**Automatic detection does not independently measure those four intersections.**
It estimates the homography from the colour-region ellipse, independently detected
bull, and rotation, then refines it using board-only image gradients (ECC) and
measured colour-band centres. The labelled radial-wire angles are still
**projections of the model**; their endpoints use the fitted coloured outer-double
boundary when available, rather than an independently detected wire intersection.
The colour pattern repeats every 36 degrees; use `--upright` to resolve orientation
when the 20 is not near the top of the image.

The bull is measured from an original-resolution crop: a compact red inner bull
must have a green surround and plausible size/shape. Its centre constrains a
projective correction that preserves the outer-double ellipse instead of simply
translating the overlay. The constraint is applied before rotation estimation and
again after ECC so the final model centre remains on the measured bull. When the
bull is occluded or cannot be identified confidently, detection continues without
this constraint and reports the fallback. The fitted small red ellipse centre is
an approximation to the projected bull centre, not a calibrated ground-truth point.

Double and treble bands are measured from full-resolution red/green radial strips,
without dilating the colour mask or relying on grey/metal gradients. Spokes near
sector wires are skipped. Both sides of a plausible-width coloured band must be
visible, with support distributed around the board. Robust refinement uses band
centres to correct common scale/perspective while holding the bull fixed. Each
paint boundary is then fitted separately, because a single homography cannot
account for every difference between actual coloured widths and nominal wire-based
radii. Grey metal therefore does not become part of the coloured scoring band.

The fitted boundaries are available through `Detection.rings` and used in both
source and top-down overlays, including where radial lines end. Occluded spokes
are omitted; unreliable bands retain nominal boundaries. Smooth ellipses bridge
wire gaps and small occlusions, so these are fitted boundaries rather than a
pixel-perfect segmentation. The existing `score_from_model_point` and
`score_from_image_point` helpers still use standard radii, not these independently
measured paint contours.

With `--manual` or `--points`, the homography is instead computed from the four
supplied intersections. These modes write the three standard images and
`05_calibration.png`, labelled as supplied points; automatic-stage images are
not generated. Use a fresh output directory when comparing different modes to
avoid confusing older images with a new run.

## Running notebooks after code changes

Select the repository's `.venv` Python kernel (`uv sync --group dev`). After
updating code, **Restart Kernel**, then **Run All**. Existing imports and analysis
objects can otherwise retain the previous API, causing errors such as missing
`ColorAssignment` or `color_model_names`. All notebooks enable autoreload for
subsequent edits, but analysis cells still need rerunning; structural changes to
classes/dataclasses require a restart.

`uv run --group dev pytest -q tests/test_notebooks.py` executes every notebook's
default cells in fresh IPython sessions, with outputs isolated in temporary
directories. Image notebooks are skipped if local DVC photographs are absent.
Optional batch/ablation switches remain off in this smoke check.

## Annotate darts locally

Run `uv run python -m dartcv.annotate` and open `http://127.0.0.1:8765`.
Choose an image folder, then a photograph, then a dart. The local browser
editor supports zoom/pan, per-dart editable polygons,
flight/shaft labels, optional inferred outlines, exact/unknown tips, ignore
regions, optional manual sector/ring scores (single by default), contextual
hover help, undo/redo, and save/resume. Images stay local and unchanged.

Follow [the annotation protocol](docs/dart-annotation.md). Save complete
source-coordinate labels under `data/annotations/darts/`, then enable
`RUN_LABEL_EVALUATION` in the segmentation comparison notebook for dart/part
retention, foreground precision, IoU, coverage, and an error overlay. Inferred
outlines and draft photos are not segmentation ground truth.

## Dart scoring notebook

[examples/dart_scorer_pipeline.ipynb](examples/dart_scorer_pipeline.ipynb) stages the
scoring logic from model-space checks to image-space homography scoring and a
final end-to-end board run. The notebook validates the standard radius/sector rules,
confirms the homography round-trip on synthetic calibration points, and shows how
`detect_board()` feeds the same score helpers used in the live pipeline.

## Multi-dart board inspection

Open [examples/color_dart_board_inspection.ipynb](examples/color_dart_board_inspection.ipynb)
in VS Code with the project's Python environment (`uv sync --group dev`).
Run the cells, choose `IMAGE_INDEX` from `data/images/multi_dart`, and rerun from
**Run preparation** to switch photos. The notebook shows source/normalized views,
nominal ring/spoke overlays, color and board masks, detector artifacts, and
per-stage timings; an opt-in cell compares normalization resolutions.

The notebook uses `prepare_board(image, diagnostics=True)`, also available as a
library function without constructing unfinished recognition stages. Expected
failures remain visible with partial diagnostics. Plotting returns a Matplotlib
figure without automatically showing or saving it.

This is board preparation only: overlays are nominal geometry, not detected
physical wires or confirmed numbered sectors. The normalized view excludes the
outer printed-number ring by default. Full-resolution diagnostic snapshots are
opt-in and memory-heavy.

For milestone 3, open
[examples/color_dart_orientation_inspection.ipynb](examples/color_dart_orientation_inspection.ipynb).
It requests an optional wider view with source-coverage mask, extracts an explicit
`20` template from a manually identified reference, and inspects the number strip,
candidate matches, confidence and internal timings on another photo. The original
board-inspection notebook and scoring-view transforms are unchanged.

`Printed20OrientationStage(template)` is an injectable, local OpenCV matcher,
not general OCR. Use `partial(normalize_board_view, number_ring_radius_mm=240.0)`
as the normalizer to supply its input. Its clockwise correction is relative to
the rectified model; uncertain guesses remain explicit and absent evidence fails.
The template, annular radii and thresholds are provisional and board-style
dependent. The tiny [annotation manifest](data/annotations/color_dart_orientation.json)
is for inspected same-board examples, not a dataset accuracy claim. Orientation
remains provisional; entry-point localization and new scoring are unimplemented.

## Color-pipeline overview and image batches

Start with [the combined pipeline overview](examples/color_dart_pipeline_overview.ipynb)
to see every implemented color-pipeline stage in one place. It covers board
preparation, printed-20 orientation, field sampling, four robust board-color
models, and an initial dart-likelihood map. For detailed color-stage investigation,
use [the segmentation notebook](examples/color_dart_segmentation_inspection.ipynb).
Entry-point localization and independent scoring are not implemented yet.

Every feature notebook now has a **Compact batch** section. Set `RUN_BATCH = True`
and edit `BATCH_PATHS` (three images by default; remove `[:3]` for the directory).
Only final outputs appear, in six-image pages with filenames, status and elapsed
time. Copy a bad filename into the single-image selection to inspect intermediates.
Run the color notebooks' explicit reference-template setup once before batching.
Legacy batch cells have explicit `BATCH_*` settings; match them to the detailed
settings when comparing results. Existing single-image cells and saved outputs
are preserved.

The visualization-only notebook batches render sizes instead of photographs.
The old `find_single_dart.ipynb` scratch notebook batches its supported
board-detection section, not its incomplete unrelated experiments. Batch scoring
in `dart_scorer_pipeline.ipynb` uses the older scorer prototype, not the new color
pipeline. Expected read/detection failures are shown rather than silently skipped.

`RobustBoardColorSegmentationStage` fits per-image field-interior distributions
in hue-wrapping HSV-cone coordinates. Its revised cleanup keeps thin interior
components and restores nearby wire-crossing evidence without flooding the grid
or opening away shafts. A weak numeral match retains the detector's paint
alignment; numbered orientation remains uncertain. Raw anomalies and model
support remain inspectable. Reflections and missed color-matching dart pixels
remain possible; no blob is automatically declared a dart or assigned a score.

Use [the segmentation comparison notebook](examples/color_dart_segmentation_comparison.ipynb)
to compare **field-specific**, **any board color**, and **whole-board clustering**
against no cleanup, hard grid removal, or support-preserving cleanup. It includes
selected-image overlays, compact batches with `BATCH_VARIANT`, and an opt-in
six-photo sparse annotation evaluation, split into four initial regressions and
two additional unresolved challenge cases. The overview and detailed segmentation
notebooks also have an optional `RUN_ABLATION` cell.

On the small development subset, whole-board clustering removed more dart
evidence because white flights/metal shafts resembled board colors. Field-specific
matching remains the default, with `color_assignment="board_union"` or `"global"`
available for experiments. These results do not establish full-mask accuracy;
residual wire noise is still visible. The comparison notebook starts on a noisy
challenge photo deliberately: no tested strategy cleanly isolates every dart.

## Dart recognition notebook

[examples/dart_recognition.ipynb](examples/dart_recognition.ipynb) walks one image
(default: a photo from `data/images/single_dart/`, set via `IMAGE_PATH`) through
board detection, rectification, dart/number/tip recognition slots, scoring, and a
combined view. Use the project's `dartcv` kernel.

Stage 4 defaults to the experimental classical single-dart detector in
[src/dartcv/classical_darts.py](src/dartcv/classical_darts.py). It combines local
colour/bright-flight contrast with attached parallel shaft edges, using the
calibrated board to limit the search and reject paint confounders. It requires
no optional runtime or model downloads. Configure `CLASSICAL_FLIGHT_HSV`,
`CLASSICAL_WORK_SIZE`, `CLASSICAL_MIN_SCORE`, and `CLASSICAL_AMBIGUITY_MARGIN`
in the contracts cell. `SHOW_CLASSICAL_DIAGNOSTICS` displays the search region,
flight mask, line segments and ranked explanations. Quality scores are heuristic,
not probabilities. It returns at most one box and abstains for weak or ambiguous
evidence; it does not certify that exactly one dart exists.

The current default photograph produces an abstention, not a correct box:
board markings compete with the flight/shaft evidence. This is a constrained
prototype, not established real-image detection accuracy. Dark flights, merged
paint components, thin shafts and occlusions remain important limitations.

The existing OWLv2 backend remains available for a pretrained comparison. Install
its optional runtime:

```sh
uv sync --extra dart-detection --group dev
```

The first inference downloads model weights (hundreds of MB) into the local
Hugging Face cache, not the repository or DVC dataset. The model runs locally;
photos are not uploaded. Model weights use Apache-2.0 licensing. The default
revision is pinned in [src/dartcv/dart_detection.py](src/dartcv/dart_detection.py).
Construction/import does not download; `local_files_only=True` on the adapter
forbids downloads and requires a populated cache. CPU is its tested default.

To compare it, assign `DART_DETECTOR = Owlv2DartDetector()` after importing that
class from `dartcv.dart_detection` and running the classical setup. Re-running
the setup restores the classical backend. The notebook caches one classical
detector while its settings and `H` match. A kernel restart clears it. Set
`ENABLE_DART_DETECTION = False` to skip detection. Core CLI behavior is unchanged.
OWLv2 is experimental: a flight-only box or a box around the board is not a
successful whole-dart detection. Its scores are not accuracy probabilities.

Number reading and tip localisation still have unset backend slots and report
`not configured`. Boxes alone never produce scores. Scores can be computed for
hand-picked `MANUAL_TIPS`, using the standard board radii.

Reusable contracts, validation, and backend execution live in
[src/dartcv/recognition.py](src/dartcv/recognition.py); drawing helpers live in
[src/dartcv/recognition_overlay.py](src/dartcv/recognition_overlay.py).
The notebook imports these helpers and keeps only configuration and presentation.
See the [recognition roadmap](docs/cv-pipeline.md#recognition-contracts-and-next-detection-stage)
for the proposed box/tip annotation, model integration, and evaluation sequence.

## Pictures on Google Drive (uv + DVC)

Pictures in `data/images/` are managed by DVC, not committed directly to Git.
Git tracks `data/images.dvc`, which identifies the version of the entire image
folder. DVC stores the image contents on Google Drive using the configured
default remote, `gdrive-remote`.

Run the following commands from the repository root.

### Initial setup and authentication

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and sync
the project environment (this also installs DVC with Google Drive support):

```sh
uv sync
```

Ask a project maintainer for access to the Google Drive remote folder and the
OAuth client credentials. Configure credentials locally; replace the
placeholders below with the supplied values:

```sh
uv run dvc remote modify --local gdrive-remote gdrive_client_id '<CLIENT_ID>'
uv run dvc remote modify --local gdrive-remote gdrive_client_secret '<CLIENT_SECRET>'
```

The `--local` flag saves these values in `.dvc/config.local`, which is ignored
by Git. Never commit credentials or authentication tokens, or paste them into
issues or chat. Commands containing actual secrets may also be saved in shell
history; handle that history as sensitive.

The first pull or push may open a browser for Google authorization. Use an
account with access to the remote folder and keep the terminal command running
until authentication finishes.

If Safari blocks the redirect to `http://localhost:8080` because HTTPS-Only mode
is enabled, open the full callback URL in another browser on the same Mac while
DVC is still running. Alternatively, temporarily disable HTTPS-Only mode and
reload, then re-enable it afterward. Do not change the callback URL to HTTPS.
If DVC has stopped, restart the operation to begin a fresh login. Treat callback
URLs as sensitive because they contain temporary authorization codes.

### Download the latest pictures

```sh
git pull
uv run dvc pull
```

`git pull` fetches the latest DVC metadata; `uv run dvc pull` downloads and
checks out the corresponding pictures into `data/images/`. Preserve any local
image edits first; do not force a pull over changes you want to keep.

### Upload added, edited, or deleted pictures

After changing files in `data/images/`:

```sh
# Inspect local changes relative to the recorded image version.
uv run dvc status

# Record the entire folder, including additions, edits, and deletions.
uv run dvc add data/images

# Upload the recorded image contents to Google Drive.
uv run dvc push

# Publish the metadata so others can download the same image version.
git add data/images.dvc data/.gitignore
git commit -m "Update image dataset"
git push
```

Use `uv run dvc add data/images`, not `uv run dvc add data/images/*`: the folder
is already tracked as a single dataset. A push alone does not record new image
changes; run `dvc add` first. Upload with DVC before pushing the Git commit so
the referenced data is available to collaborators. DVC retains older cached
versions; deleting a local picture does not purge its historical contents from
Google Drive.

### Check synchronization

```sh
# Compare local pictures with the version recorded in Git/DVC metadata.
uv run dvc status

# Compare the local DVC cache with Google Drive.
uv run dvc status -c
```

These checks serve different purposes: a synchronized remote cache does not
mean unrecorded local image changes have been uploaded. For those changes,
repeat the add, DVC push, Git commit, and Git push workflow above.
