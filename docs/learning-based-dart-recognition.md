# Learning-based dart recognition: proposal

Status: an alternative to revisit, not an implemented or validated pipeline.
This proposal does not replace the current color-based work.

## Goal

Recognize individual darts in a photograph, locate their board-contact points,
and calculate scores while preserving uncertainty. Learn dart appearance and
part relationships rather than relying only on differences from board colors.

The recent AI-assisted annotation was visual interpretation of a preview and
full-resolution crop, not output from the existing detector or a new Python
algorithm. Replaying its saved coordinates reproduces that annotation, not
automatic recognition. This proposal is a separate way to automate the task.

## Proposed pipeline

1. **Calibrate the board.** Reuse `detect_board()` and the numbered-sector
   orientation stage. Preserve their failure and uncertainty information.
   The existing printed-20 matcher requires a caller-supplied template and can
   be wrong when the label is obscured.
2. **Detect individual darts in the original photograph.** Fine-tune a pretrained
   detector, adding instance segmentation if needed to separate overlapping
   darts. Flight and shaft/barrel labels can provide additional part supervision.
   Do not rely on one connected foreground component per dart.
3. **Locate the contact point.** Use a keypoint predictor on a high-resolution
   crop around each detected dart. Restore crop/resize coordinates to original
   image pixels. Learn the relationship between flight, shaft/barrel, and contact
   end; do not assume the contact end is nearest the bull.
4. **Refine and check.** Optionally use local shaft edges to refine the axis and
   endpoint. Retain uncertain estimates or abstain when evidence is insufficient.
   A hidden contact cannot generally be recovered uniquely from one photograph;
   never manufacture it by intersecting the shaft with an arbitrary board line.
5. **Score geometrically.** Map each contact pixel through the inverse board
   homography, apply the numbered-orientation correction, then classify sector
   and nominal scoring ring. Scoring itself requires no learned model.
6. **Propagate uncertainty.** Evaluate whether plausible contact-position
   perturbations change the score. Also account for calibration and orientation
   uncertainty. Near boundaries, retain a best guess and possible alternatives.
   Unknown contacts remain unscored, not zero-point misses.

Use the original perspective for dart recognition: protruding darts are not
coplanar with the board, so board rectification distorts their appearance.
Only the contact point is mapped into board coordinates for scoring.
Model confidence scores are not automatically calibrated probabilities.

## Annotation requirements

Use the existing [annotation format and editor](dart-annotation.md).
For each visible dart, record:

- A box, or visible polygons from which a box can be derived.
- Flight and shaft/barrel polygons when training segmentation or part prediction.
- Exact, approximate, or unknown tip status and source-image coordinates when
  available.
- Embedded, not-embedded, or unknown contact status.
- Optional manually verified score for end-to-end evaluation.

A physical tip is a board-contact point only when the dart is embedded.
Approximate points must not be treated as exact targets, and unknown points
must not become fabricated coordinates. Visible masks exclude occluded regions;
inferred silhouettes remain separate hypotheses. Draft AI-generated annotations
need human review before being promoted to ground truth.

## Starting data budget

There is no universal minimum. The following are rough planning estimates for
fine-tuning a pretrained model, not measured sample requirements or accuracy
guarantees.

| Goal | Suggested starting photographs |
| --- | --- |
| Narrow proof of concept on one board and similar scenes | 50–100 |
| First useful detector and contact-point locator | 200–500 |
| Broad variation in boards, cameras, lighting, and overlap | Likely thousands |

A concrete first experiment could use 100 photographs: 60 training, 20 validation,
and 20 held-out testing. This is a feasibility study, not enough to certify
reliability. The small test set will give imprecise performance estimates.

Split by capture session or scene before augmentation. Near-identical consecutive
photos must not appear across training and testing. Include:

- Empty boards and distracting board markings.
- Different positions, sectors, dart appearances, and camera angles.
- Thin shafts, cropped flights, overlapping darts, and occluded contacts.
- Changes in lighting, shadows, focus, and image scale.

Augmentation supplements real diversity; it does not replace it. Multiple darts
in one image supply multiple instances, but not independent scenes. Existing
handfuls of labels support integration checks and exploration, not claims of
general recognition accuracy.

## Smallest useful experiment

1. Implement and verify the independent geometric scorer using manual entry
   points, including orientation, bull, miss, and boundary cases.
2. Curate the initial dataset and freeze the session-separated split.
3. Fine-tune one small pretrained detector/keypoint baseline. Preserve
   high-resolution contact detail; add segmentation only if it improves errors
   such as overlapping-dart separation.
4. Measure detection precision/recall, contact localization error in source
   pixels and board millimetres, and score correctness. Report unresolved cases,
   coverage, and wrong confident predictions separately. Evaluate approximate
   and hidden contacts separately from exact visible contacts.
5. Inspect failures, then decide whether additional labels, part segmentation,
   edge refinement, or a different model are justified.

Do not silently use the expected dart count to force detections or choose
predictions based on ground-truth proximity. Keep evaluation images out of
training and model selection.

## Risks and open decisions

- No model family, weights, license, latency target, or training budget has been
  selected or benchmarked.
- Contact localization may need higher resolution than whole-dart detection.
- Learned recognition does not fix bad board calibration or incorrect orientation.
- Occlusion and ambiguous contact remain physical limitations of a single view.
- A model that works on this board may fail on different boards or dart designs.
- Keep photographs local; any future external inference or training service
  requires an explicit data-handling decision.
