# Annotating dart photographs

Start from the repository root:

```bash
uv run python -m dartcv.annotate
```

Open the printed `http://127.0.0.1:8765` address. The editor serves only local
files over loopback; no images or labels are uploaded. Stop with Ctrl-C.
No additional dependencies or model downloads are needed.

Choose **Folder**, then a photograph, then a dart within that photograph.
By default folders are discovered directly beneath `data/images`, starting
with `multi_dart`. Previous/Next and unfinished navigation stay within the
selected folder. Changing folders warns before discarding unsaved edits.
Only discovered folders under the configured image root can be selected.
`--images` also accepts a single folder containing photos.

Labels mirror dataset folders: `data/images/multi_dart/photo.jpg` has its labels
in `data/annotations/darts/multi_dart/photo.jpg.json`. The same rule applies to
`single_dart`, `edge_cases`, and `raw_board`, avoiding filename collisions.
When `--images` points directly to a photo folder, its labels go directly in
the exact directory supplied via `--annotations`.
If an image root itself contains photos as well as child folders, its own
photos appear as "Selected image folder" and all children use label subdirectories.
For evaluation of another dataset, set the notebook's `LABEL_DIR` accordingly.

To choose another local dataset or output directory:

```bash
uv run python -m dartcv.annotate --images data/images/multi_dart \
  --annotations data/annotations/darts/multi_dart --port 8765
```

## Labeling protocol

1. Select a photo and a dart. Three empty dart entries are provided as a
   convenience, not a count constraint: add or remove entries as needed. Keep a
   stable identity for each dart within that photo; IDs do not track darts
   between photographs.
2. Trace the **visible flight**, **visible shaft/barrel**, and, if necessary,
   **visible dart, uncertain part**. Click polygon vertices, then press Enter
   or Finish. A part can contain multiple disconnected polygons. For a shaft
   hidden behind another flight, trace its visible pieces separately; do not
   include the occluded gap as visible foreground.
3. Optionally draw a **complete inferred silhouette**, or copy a visible polygon
   and edit that copy. This is a separate hypothesis about the full dart shape;
   it never replaces visible labels or contributes to segmentation truth.
4. Click **Place exact tip**, then the physical dart point. Use **Place
   approximate tip** to record an estimated position when the endpoint is
   hidden or unclear, or **Hidden / unknown (no point)** to leave it unlocated.
   Approximate points have a dashed-circle marker and remain approximate when
   dragged; the circle is not a measured uncertainty radius. Set contact to
   **embedded**, **not embedded**, or **unknown**. A physical tip is a board
   entry point only when embedded. A dart outside the scoring area is still
   annotated; a dart entirely outside the photo is not.
   Include the visible metal tip in its shaft/barrel polygon as well: the
   polygon labels visible pixels; the exact point labels its endpoint.
   Optionally choose a **score number** and **single/double/triple** (single
   by default). The number is the sector, not multiplied points: 20 + triple
   means 60. Choose 25 + single for outer bull, 25 + double for inner bull,
   or Miss (0) for a miss. Leave Not labeled / unknown when unsure.
5. Mark an **ignore** polygon where dart/background identity is genuinely
   ambiguous, such as blur or an indistinguishable overlap boundary. Use this
   sparingly: uncertainty about *which dart part* is present still belongs in
   the visible `uncertain` category, not ignore. Do not label shadows or board
   reflections as dart.
6. Review all visible darts, including off-board parts. Set the photo to
   **Completed** only when every visible dart has visible regions and an exact,
   approximate or explicitly unknown tip. Pixels outside visible and ignore regions become
   background truth. **Save** or **Save & Next**; unfinished work stays Draft.
   A completed photo with zero darts is valid only if no dart is visible.

The application cannot verify that you found every dart, traced accurate
boundaries, or chose the right contact status. Completion is a human assertion;
the server checks label structure and requires visible regions for every listed
dart. No exact-three or on-board-only assumption is made.

### AI-assisted draft baseline

The 43-photo batch adds drafts for 26 `single_dart` photos
(`PXL_20261018_152711553.jpg` through `PXL_20261018_153344947.jpg`),
16 `multi_dart` photos (`PXL_20261018_154237729.jpg` through
`PXL_20261018_154823461.jpg`), and
`edge_cases/darts_overlapping_in_tripple_20.jpg`. Existing sidecars were preserved.
These 77 darts were labeled by visual inspection of local magnified crops, not
by the recognition pipeline or a trained detector. Polygons are coarse visible
flight and shaft/barrel drafts; no hidden silhouettes were inferred.

These 43 photos and the earlier AI experiment
(`single_dart/PXL_20261018_152650640.jpg`) use **AI draft** (`ai-draft`) status.
It means AI-preannotated and awaiting human review. Saving corrections retains
this status; explicitly choose Completed only after reviewing the whole photo.
AI drafts appear in the photo list and in Next unfinished, and are excluded
from evaluation just like ordinary drafts. This is a review status, not
permanent authorship metadata: completing the photo replaces it with Completed.

All 76 placed tips are approximate; one obscured tip remains unknown. Only eight
scores were tentatively labeled; other scores remain null. Review boundaries,
overlapping-dart ownership, tip/contact positions and any score before marking
these photos Completed or using them as training/evaluation truth. In particular,
draft tips near scoring wires do not establish which side of the wire was hit.

## Editing and saving

- Wheel or Zoom +/- to zoom; Fit Image resets the view. Hold Space and drag, or
  use the middle mouse button, to pan. Coordinates remain at original resolution.
- Hover a control for its native tooltip, or hover/focus the adjacent **?**
  badge for a contextual help box. Keyboard users can Tab to the badges.
- Select a polygon in the list, then drag vertices. Double-click an edge to add
  a vertex; right-click a vertex to remove it. For repeated corrections, select
  **Add vertex (A)** and click edges, or **Remove vertex (R)** and click corners.
  These modes stay active until changed; highlighted buttons and a mode hint
  show the current tool. **Esc** or **Select / cancel** returns to selection
  and dragging. A/R shortcuts do not activate while typing in a field.
  Removing a corner joins its neighbors; at least three vertices must remain.
  Right-click during drawing removes the most recent point.
  Delete selected polygon removes an entire region.
- Undo/redo applies to labels, not zoom/pan. Escape cancels an unfinished polygon.
  Finish or cancel a polygon before changing darts, saving, or navigating.
- An unsaved indicator and navigation/close warnings protect pending work.
  Saving is explicit, not automatic. Resume by reopening the same image folder
  and annotation directory. Download JSON preserves a separate copy but does
  not mark server changes saved.
- Saves validate finite in-image coordinates, at least three non-collinear vertices,
  completion rules and source identity, then replace JSON atomically.
  A revision conflict from another tab is shown explicitly; download your copy
  before reloading. Use one editor server per annotation directory.

## Ground-truth format

Files are named `<original-filename>.json`, including the original image
extension to avoid collisions. `schema_version: 1` contains:

| Field | Meaning |
| --- | --- |
| `image`, `width`, `height`, `sha256` | Original filename, decoded pixel dimensions and source-file fingerprint |
| `status` | `draft`, `ai-draft` (AI-preannotated, awaiting human review), or `completed` |
| `darts[].id` | Unique per-photo dart identity |
| `darts[].visible[]` | `part` (`flight`, `shaft_barrel`, `uncertain`) and polygon `points` |
| `darts[].silhouette[]` | Optional complete inferred polygons, distinct from observations |
| `darts[].tip_status`, `tip` | `exact` or `approximate` with `[x,y]`, or `unknown` with `null` |
| `darts[].contact` | `embedded`, `not_embedded`, `unknown` |
| `darts[].score` | Optional `null` or `{number, multiplier}`; number 0, 1-20, or 25; multiplier `single` (default), `double`, `triple` |
| `ignore[]` | Ambiguous polygons excluded from visible-mask evaluation |

Every polygon is an ordered array of source-photo `[x,y]` vertices, origin at
top left, x right, y down. There is no need to repeat the first vertex at the end.
Crossing edges, self-touching outlines and repeated vertices are accepted.
Overlapping loops within a polygon use even-odd filling in the GUI and masks;
inspect the filled area to confirm it represents the intended region. Polygons
have no holes; use multiple polygons for disconnected parts. Regions are
combined by union, not by winding direction. Ignore regions take precedence.
Yellow circles mark self-crossings, touches, repeated vertices and endpoints of
overlapping edges. The polygon list's warning summary identifies affected
regions for the selected dart and shared ignore regions. These are informational:
you can save without fixing them. Toggle **Mark crossings / touches** to hide
the markers. Separate polygons may overlap without a warning; this is normal
for different parts or inferred silhouettes.
The displayed full-resolution JPEG is decoded and re-encoded using OpenCV,
matching the pipeline's EXIF orientation handling rather than a browser-specific
image orientation. Source files are never modified. The SHA-256 is checked on
resume and when notebook evaluation loads labels, preventing reuse after the
source image changes.

These labels are independent of detector homographies and printed-number
orientation. The existing normalized sparse-probe manifest is retained as a
separate development fixture; it is not converted into complete outlines.
Store only small JSON labels in git, not source images; no automatic commits
are made when annotating.

Existing schema-version-1 files without `score` remain readable and are treated
as unlabeled scores. A score is optional even for completed photos, is entered
manually, and is never inferred from the tip or contact status. Only single is
valid for a miss; bull permits single/double, not triple. The GUI shows the
derived points without storing a redundant total. Score edits support undo/redo
and save/resume independently for each dart. Segmentation metrics ignore scores.

## Evaluating the color algorithm

In `examples/color_dart_segmentation_comparison.ipynb`, select the annotated
photo, run its analysis, then enable `RUN_LABEL_EVALUATION`. The final cell
compares all nine existing variants on the same labels and rectification.
`LABEL_OVERLAY_VARIANT` selects one error overlay; `LABEL_THRESHOLD` defaults
to 0.5.

| Output | Interpretation |
| --- | --- |
| Dart retained | Recall: retained labeled visible dart pixels / eligible labeled dart pixels |
| Detected pixels that are darts | Precision: correctly predicted dart pixels / all eligible predicted foreground |
| Flight / shaft-barrel retained | Recall within each visible part label |
| IoU | Correct foreground / union of labeled and predicted foreground |
| Label coverage | Fraction of visible source-labeled dart pixels inside this evaluation's observed scoring-board view |

All metrics use the nominal scoring-board mask and fully observed source
coverage. Off-board or cropped labels are not scored as misses; ignore regions
are excluded. Coverage is counted in source pixels, while overlap metrics use
normalized-view pixels. `n/a` means a denominator is empty. A cleaner mask can
have higher precision while losing darts, so inspect retention and the error
overlay together. Green is retained dart, red is missed dart, cyan is false
foreground, and gray is excluded coverage.

These are binary visible-region metrics, not per-dart instance accuracy,
entry-point error, scores, or a generalization claim. Inferred silhouettes,
contact status and tip labels are deliberately not used to score segmentation.
Future tip localization can consume exact labels, distinguishing embedded entry
points from nonembedded tips and excluding unknown points. Approximate points
must remain separate from exact ground truth rather than silently entering
exact-localization error metrics.
