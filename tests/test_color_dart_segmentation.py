from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dartcv import (
    BoardGeometry, ColorBasedDartRecognitionPipeline, OrientationEstimate, RobustBoardColorSegmentationStage,
    StageOutcome, analyze_board_colors, board_color_features, build_field_masks,
    fit_board_color_models, normalize_board_view,
)
from dartcv.color_dart_inspection import plot_color_segmentation
from dartcv.color_dart_segmentation import InsufficientColorSamples, suppress_wire_artifacts


BOARD = BoardGeometry(np.array([[2., 0, 350], [0, 2., 350], [0, 0, 1]]), 0)
ORIENTATION = OrientationEstimate(0, .9, True)


def _board_view():
    image = np.zeros((701, 701, 3), np.uint8)
    view = normalize_board_view(image, BOARD, 701, diagnostics=False).value
    masks = build_field_masks(view, ORIENTATION)
    colors = np.array([[30, 30, 190], [45, 130, 35], [120, 170, 190], [25, 25, 25]], np.uint8)
    for index in range(4):
        view.image[masks.expected_color == index] = colors[index]
    return view


def _at(view, x, y):
    point = view.model_to_view @ [x, y, 1]
    return round(point[1]), round(point[0])


def test_masks_assign_rings_bulls_and_signed_orientation():
    view = _board_view()
    masks = build_field_masks(view, ORIENTATION)
    for x, y, color in [(0, -60, 3), (0, -103, 0), (0, 0, 0), (0, -10, 1),
                         (18.54, -57.06, 2), (31.83, -97.96, 1)]:
        assert masks.expected_color[_at(view, x, y)] == color
    rotated = build_field_masks(view, OrientationEstimate(-18, .9, True))
    assert rotated.expected_color[_at(view, 0, -60)] == 2
    assert rotated.expected_color[_at(view, 0, -103)] == 1
    assert not masks.sample_interior[_at(view, 0, -99)]
    assert masks.wire_region[_at(view, 0, -99)]
    assert not masks.wire_region[_at(view, 2, -2)]


def test_models_reject_sparse_dart_contamination_and_detect_different_colors():
    view = _board_view()
    # Bright blue flight in a black single field, away from expected wires.
    y, x = _at(view, 0, -65)
    view.image[y-12:y+12, x-12:x+12] = (240, 30, 20)
    before = view.image.copy()
    result = RobustBoardColorSegmentationStage()(view, BOARD, ORIENTATION, diagnostics=True)
    assert result.value is not None
    assert result.value.probabilities[y, x] > .9
    assert result.value.probabilities[_at(view, 0, -45)] == 0
    np.testing.assert_array_equal(view.image, before)
    models = fit_board_color_models(board_color_features(view.image), build_field_masks(view, ORIENTATION))
    assert models[3].retained_count < models[3].sample_count
    np.testing.assert_allclose(models[3].center, [0, 0, 25/255], atol=.01)
    FigureCanvasAgg(plot_color_segmentation(result)).draw()


def test_a_normal_board_color_is_still_anomalous_in_the_wrong_field():
    view = _board_view()
    y, x = _at(view, 0, -65)
    # This exact green belongs elsewhere on the board, not in this black field.
    view.image[y-10:y+10, x-10:x+10] = (45, 130, 35)
    outcome = RobustBoardColorSegmentationStage()(view, BOARD, ORIENTATION, diagnostics=False)
    assert outcome.value.probabilities[y, x] > .9


def test_hue_wrap_and_dark_pixels_have_continuous_features():
    hsv = np.array([[[0, 220, 180], [179, 220, 180], [90, 255, 0], [0, 0, 0]]], np.uint8)
    features = board_color_features(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))[0]
    assert np.linalg.norm(features[0] - features[1]) < .04
    np.testing.assert_array_equal(features[2], features[3])


def test_missing_source_pixels_never_train_models_or_become_darts():
    view = _board_view()
    y, x = _at(view, 0, -65)
    view.source_coverage[y-15:y+15, x-15:x+15] = 0
    view.image[y-15:y+15, x-15:x+15] = (240, 30, 20)
    result = RobustBoardColorSegmentationStage()(view, BOARD, ORIENTATION, diagnostics=True)
    assert result.value.probabilities[y, x] == 0
    assert result.artifacts["field_sample_interiors"][y, x] == 0
    unsupported = replace(view, source_coverage=np.zeros_like(view.board_mask))
    failure = RobustBoardColorSegmentationStage()(unsupported, BOARD, ORIENTATION, diagnostics=True)
    assert failure.value is None and "Insufficient" in failure.failure
    FigureCanvasAgg(plot_color_segmentation(failure)).draw()
    missing = RobustBoardColorSegmentationStage()(replace(view, source_coverage=None),
                                                 BOARD, ORIENTATION, diagnostics=False)
    assert "source coverage" in missing.failure


def test_cleanup_removes_expected_wire_and_small_components_but_is_replaceable():
    view = _board_view()
    masks = build_field_masks(view, ORIENTATION)
    raw = np.zeros(view.image.shape[:2], np.float32)
    raw[masks.wire_region] = 1
    y, x = _at(view, 0, -65)
    raw[y:y+12, x:x+12] = 1
    raw[y+30, x] = 1
    cleaned = suppress_wire_artifacts(raw, masks, 8)
    assert not cleaned[masks.wire_region].any()
    assert cleaned[y+5, x+5] == 1
    assert cleaned[y+30, x] == 0
    calls = []

    def replacement(probabilities, field_masks, area):
        calls.append(area)
        return probabilities

    result = RobustBoardColorSegmentationStage(cleanup=replacement)(
        view, BOARD, ORIENTATION, diagnostics=False
    )
    assert calls and result.value is not None and result.artifacts == {}


def test_analysis_keeps_expected_failures_and_skips_dependents():
    def no_board(image, *, diagnostics):
        return StageOutcome.failed("No board")

    def unused(*args, **kwargs):
        pytest.fail("Dependent stage should not execute")

    result = analyze_board_colors(np.zeros((10, 10, 3), np.uint8),
                                 orientation=unused, segmentation=unused,
                                 board_localization=no_board)
    assert result.orientation is None and result.segmentation is None
    assert [step.status.value for step in result.steps] == ["failed", "skipped", "skipped", "skipped"]
    assert result.preparation.failure.message == "No board"


def test_analysis_runs_real_segmentation_and_preserves_orientation_uncertainty():
    view = _board_view()

    def localize(image, *, diagnostics):
        return StageOutcome.succeeded(BOARD)

    def normalize(image, board, size, *, diagnostics):
        return StageOutcome.succeeded(view)

    def orient(board_view, *, diagnostics):
        return StageOutcome.succeeded(OrientationEstimate(0, .4, False))

    result = analyze_board_colors(view.image, orientation=orient,
                                 segmentation=RobustBoardColorSegmentationStage(),
                                 board_localization=localize, board_normalization=normalize,
                                 diagnostics=True)
    assert result.segmentation.value is not None
    assert not result.orientation.value.confident
    assert not result.segmentation.artifacts["orientation_confident"][0]
    assert all(step.elapsed_ms >= 0 for step in result.steps)
    assert all(step.status.value == "succeeded" for step in result.steps)
    np.testing.assert_array_equal(result.steps[2].input_artifacts["normalized_board"], view.image)


def test_real_segmentation_stage_wires_into_the_full_pipeline():
    view = _board_view()

    def localize(image, *, diagnostics):
        return StageOutcome.succeeded(BOARD)

    def normalize(image, board, size, *, diagnostics):
        return StageOutcome.succeeded(view)

    def orient(board_view, *, diagnostics):
        return StageOutcome.succeeded(ORIENTATION)

    def tips(image, board_view, board, orientation, likelihood, count, *, diagnostics):
        assert likelihood.probabilities.shape == view.image.shape[:2]
        return StageOutcome.succeeded(())

    pipeline = ColorBasedDartRecognitionPipeline(
        board_localization=localize, board_normalization=normalize,
        orientation=orient, segmentation=RobustBoardColorSegmentationStage(),
        tip_localization=tips, scoring=lambda *a, **kw: StageOutcome.succeeded((None, None, None)),
    )
    result = pipeline.run(view.image, diagnostics=True)
    assert result.failure is None
    assert not result.total_complete
    assert any("dart_likelihood" in step.output_artifacts for step in result.steps)


def test_insufficient_model_support_and_configuration_errors_are_explicit():
    view = _board_view()
    masks = build_field_masks(view, ORIENTATION)
    with pytest.raises(InsufficientColorSamples, match="red"):
        fit_board_color_models(board_color_features(view.image),
                              replace(masks, sample_interior=np.zeros_like(masks.sample_interior)))
    for options in ({"wire_margin_mm": 3}, {"minimum_samples": 0},
                    {"minimum_component_area_mm2": -1}):
        with pytest.raises(ValueError):
            RobustBoardColorSegmentationStage(**options)


def test_default_cleanup_preserves_a_thin_shaft_across_expected_wires():
    view = _board_view()
    left = _at(view, -25, -60)
    right = _at(view, 25, -60)
    cv2.line(view.image, left[::-1], right[::-1], (240, 30, 20), 2)
    outcome = RobustBoardColorSegmentationStage()(view, BOARD, ORIENTATION, diagnostics=True)
    y = left[0]
    shaft_scores = outcome.value.probabilities[y, left[1]+2:right[1]-2]
    assert np.mean(shaft_scores >= .5) >= .9
    crossing = _at(view, 60 * np.tan(np.deg2rad(9)), -60)
    assert outcome.value.probabilities[crossing] > .9


def test_uncertain_number_match_does_not_rotate_detector_color_fields():
    view = _board_view()
    unreliable = OrientationEstimate(-102, .4, False)
    outcome = RobustBoardColorSegmentationStage()(view, BOARD, unreliable, diagnostics=True)
    assert outcome.value is not None
    assert np.mean(outcome.value.probabilities >= .5) < .005
    assert not outcome.artifacts["orientation_confident"][0]
    assert outcome.artifacts["color_alignment_uses_detector"][0]
    assert outcome.artifacts["color_alignment_correction_deg"][0] == 0


@pytest.mark.parametrize("assignment", ["board_union", "global"])
def test_whole_board_modes_can_hide_dart_pixels_matching_another_paint_color(assignment):
    view = _board_view()
    y, x = _at(view, 0, -65)
    view.image[y-10:y+10, x-10:x+10] = (45, 130, 35)
    outcome = RobustBoardColorSegmentationStage(color_assignment=assignment)(
        view, BOARD, ORIENTATION, diagnostics=True
    )
    assert outcome.value is not None
    assert outcome.value.probabilities[y, x] < .1
    assert outcome.artifacts["color_assignment"][0] == assignment


def test_support_cleanup_cannot_flood_a_connected_wire_network():
    from dartcv import preserve_dart_support

    view = _board_view()
    masks = build_field_masks(view, ORIENTATION)
    raw = np.zeros(view.image.shape[:2], np.float32)
    raw[masks.wire_region] = 1
    y, x = _at(view, 0, -60)
    raw[y-5:y+5, x-5:x+5] = 1
    raw[y+30, x] = 1
    before = raw.copy()
    cleaned = preserve_dart_support(raw, masks, 10)
    assert cleaned[y, x] == 1
    assert cleaned[y+30, x] == 0
    assert cleaned[_at(view, 0, 99)] == 0
    assert np.count_nonzero(cleaned[masks.wire_region]) < np.count_nonzero(raw[masks.wire_region]) / 10
    np.testing.assert_array_equal(raw, before)


def test_global_fit_requires_observed_color_modes_instead_of_silent_fallback():
    view = _board_view()
    view.image[:] = 0
    outcome = RobustBoardColorSegmentationStage(color_assignment="global")(
        view, BOARD, ORIENTATION, diagnostics=False
    )
    assert outcome.value is None
    assert "four distinct" in outcome.failure
    with pytest.raises(ValueError, match="assignment"):
        RobustBoardColorSegmentationStage(color_assignment="unknown")


def test_sparse_real_photo_probes_preserve_shafts_without_background_grid_noise():
    from dartcv.color_dart_evaluation import evaluate_sparse_probes

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/annotations/color_dart_segmentation.json").read_text())
    image_dir = root / "data/images/multi_dart"
    regressions = [item for item in manifest["images"] if item["role"] == "regression"]
    if not all((image_dir / item["image"]).exists() for item in regressions):
        pytest.skip("Optional DVC-managed segmentation photographs are unavailable.")
    totals = {category: [0, 0] for category in ("flight", "shaft", "board", "wire")}
    for annotation in regressions:
        board = BoardGeometry(np.array(annotation["model_to_source"]), 0)
        view = normalize_board_view(
            cv2.imread(str(image_dir / annotation["image"])), board, 768, diagnostics=False
        ).value
        orientation = OrientationEstimate(**annotation["number_match"])
        outcome = RobustBoardColorSegmentationStage()(view, board, orientation, diagnostics=False)
        assert outcome.value is not None
        metrics = evaluate_sparse_probes(
            view, board, outcome.value.probabilities, board.model_to_source,
            {category: annotation[category] for category in totals},
        )
        for category, metric in metrics.items():
            totals[category][0] += metric.positive
            totals[category][1] += metric.eligible
        if annotation["image"] == "PXL_20261018_154237729.jpg":
            # Minimized real repro: old opening/wire cleanup returned only 0.09.
            assert outcome.value.probabilities[270:273, 354:357].mean() >= .4
    for category in ("flight", "shaft"):
        assert totals[category][0] / totals[category][1] >= .8, (category, totals)
    for category in ("board", "wire"):
        assert totals[category][0] / totals[category][1] <= .05, (category, totals)
