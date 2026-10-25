from functools import partial
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dartcv.color_dart_inspection import plot_orientation
from dartcv.color_dart_orientation import (
    Printed20OrientationStage,
    extract_printed20_template,
    unwrap_number_ring,
)
from dartcv.color_dart_pipeline import (
    BoardGeometry,
    ColorBasedDartRecognitionPipeline,
    DartLikelihoodMap,
    StageOutcome,
    normalize_board_view,
    prepare_board,
)


BASE = np.array([[2.0, 0, 480], [0, 2.0, 480], [0, 0, 1.0]])


def _image(*, angle=0.0, duplicate=False, blank=False):
    image = np.zeros((961, 961, 3), np.uint8)
    if not blank:
        cv2.putText(image, "20", (453, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (240, 240, 240), 2, cv2.LINE_AA)
        if duplicate:
            image = cv2.add(image, cv2.rotate(image, cv2.ROTATE_180))
    transform = cv2.getRotationMatrix2D((480, 480), -angle, 1.0)
    return cv2.warpAffine(image, transform, (961, 961))


def _view(image, *, model_to_source=BASE):
    outcome = normalize_board_view(
        image, BoardGeometry(model_to_source, 0.0), 961,
        diagnostics=False, number_ring_radius_mm=240.0,
    )
    assert outcome.value is not None
    return outcome.value


@pytest.fixture
def stage():
    template = extract_printed20_template(_view(_image()), label_angle_deg=0.0)
    return Printed20OrientationStage(template)


def test_optional_number_ring_preserves_scoring_view_and_tracks_source_coverage():
    image = _image()
    plain = normalize_board_view(image, BoardGeometry(BASE, 0), 401, diagnostics=False).value
    wide = normalize_board_view(
        image, BoardGeometry(BASE, 0), 401, diagnostics=False,
        number_ring_radius_mm=300,
    ).value
    assert plain.number_ring is None
    np.testing.assert_array_equal(plain.image, wide.image)
    np.testing.assert_array_equal(plain.source_to_view, wide.source_to_view)
    assert wide.number_ring.source_coverage[0, 200] == 0
    assert wide.number_ring.source_coverage[200, 200] == 255
    point = wide.number_ring.model_to_view @ np.array([0, 0, 1])
    np.testing.assert_allclose(point, [200, 200, 1])


@pytest.mark.parametrize("angle", [0.0, 36.0, -72.0, 179.5, -179.5, 180.0])
def test_matching_correction_is_clockwise_and_wraps_at_seam(stage, angle):
    result = stage(_view(_image(angle=angle)), diagnostics=True)
    assert result.value is not None
    assert result.value.confident
    error = (result.value.correction_deg + angle + 180) % 360 - 180
    assert abs(error) <= 0.75
    assert set(result.substep_timings_ms) == {"unwrap_number_ring", "match_printed20"}
    assert result.artifacts["orientation_candidates"].shape[1] == 2


def test_matching_survives_perspective_rectification(stage):
    projective = np.array([[0.82, 0.04, 40], [-0.02, 0.8, 80],
                           [0.00012, -0.00009, 1.0]])
    source = cv2.warpPerspective(_image(angle=72), projective, (1000, 1000))
    result = stage(_view(source, model_to_source=projective @ BASE), diagnostics=False)
    assert result.value is not None and result.value.confident
    assert result.value.correction_deg == pytest.approx(-72, abs=0.75)
    assert result.artifacts == {}


def test_duplicate_labels_return_uncertain_best_guess(stage):
    result = stage(_view(_image(duplicate=True)), diagnostics=False)
    assert result.value is not None
    assert not result.value.confident


def test_no_contrast_or_cropped_label_has_no_supported_match(stage):
    for image in (_image(blank=True), _image()[200:]):
        # Cropping changes source coordinates, not the underlying board model.
        transform = BASE.copy()
        if image.shape[0] < 961:
            transform[1, 2] -= 200
        result = stage(_view(image, model_to_source=transform), diagnostics=True)
        assert result.value is None
        assert "No supported" in result.failure


def test_missing_wider_view_is_explicit_failure(stage):
    view = normalize_board_view(_image(), BoardGeometry(BASE, 0), 401,
                                diagnostics=False).value
    assert "wider number-ring" in stage(view, diagnostics=False).failure
    with pytest.raises(ValueError, match="wider number-ring"):
        unwrap_number_ring(view)


def test_template_rejects_missing_coverage_and_blank_content():
    cropped = _image()[200:]
    transform = BASE.copy()
    transform[1, 2] -= 200
    with pytest.raises(ValueError, match="missing source"):
        extract_printed20_template(_view(cropped, model_to_source=transform),
                                   label_angle_deg=0)
    with pytest.raises(ValueError, match="contrast"):
        Printed20OrientationStage(np.zeros((40, 50, 3), np.uint8))
    for radius in (170, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="radius"):
            normalize_board_view(_image(), BoardGeometry(BASE, 0), 100,
                                 diagnostics=False, number_ring_radius_mm=radius)


def test_plotting_is_non_mutating_and_requires_diagnostics(stage):
    view = _view(_image(angle=36))
    before = view.number_ring.image.copy()
    result = stage(view, diagnostics=True)
    figure = plot_orientation(result)
    FigureCanvasAgg(figure).draw()
    np.testing.assert_array_equal(view.number_ring.image, before)
    with pytest.raises(ValueError, match="diagnostics=True"):
        plot_orientation(stage(view, diagnostics=False))


def test_real_orientation_stage_wires_into_pipeline(stage):
    def localize(_image, *, diagnostics):
        return StageOutcome.succeeded(BoardGeometry(BASE, 0))

    def segment(view, _board, orientation, *, diagnostics):
        assert orientation.correction_deg == pytest.approx(-36, abs=0.75)
        return StageOutcome.succeeded(
            DartLikelihoodMap(np.zeros(view.image.shape[:2], np.float32))
        )

    normalizer = partial(normalize_board_view, number_ring_radius_mm=240)
    image = _image(angle=36)
    preparation = prepare_board(image, board_localization=localize,
                                board_normalization=normalizer, normalized_size_px=961)
    assert preparation.board_view.number_ring is not None
    pipeline = ColorBasedDartRecognitionPipeline(
        board_localization=localize, board_normalization=normalizer,
        orientation=stage, segmentation=segment,
        tip_localization=lambda *args, **kwargs: StageOutcome.succeeded(()),
        scoring=lambda *args, **kwargs: StageOutcome.succeeded((None, None, None)),
        normalized_size_px=961,
    )
    result = pipeline.run(image, diagnostics=True)
    assert result.orientation.confident


def test_manually_inspected_same_board_photos():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/annotations/color_dart_orientation.json").read_text())
    directory = root / "data/images/multi_dart"
    names = [manifest["reference"]["image"], *(entry["image"] for entry in manifest["checks"])]
    if not all((directory / name).exists() for name in names):
        pytest.skip("Optional DVC-managed orientation photographs are unavailable.")
    normalizer = partial(normalize_board_view, number_ring_radius_mm=240)

    def preparation(name):
        image = cv2.imread(str(directory / name))
        assert image is not None
        result = prepare_board(image, board_normalization=normalizer)
        assert result.failure is None
        return result

    def angle(board, source_xy):
        point = np.linalg.inv(board.model_to_source) @ np.array([*source_xy, 1.0])
        point = point[:2] / point[2]
        return float(np.rad2deg(np.arctan2(point[0], -point[1])))

    reference = preparation(manifest["reference"]["image"])
    template = extract_printed20_template(
        reference.board_view,
        label_angle_deg=angle(reference.board, manifest["reference"]["label_center_source_px"]),
    )
    matcher = Printed20OrientationStage(template)
    for annotation in manifest["checks"]:
        prepared = preparation(annotation["image"])
        result = matcher(prepared.board_view, diagnostics=False)
        confident = result.value is not None and result.value.confident
        assert confident == annotation["expected_confident"], annotation["image"]
        if annotation["label_center_source_px"] is not None:
            assert result.value is not None
            observed_angle = angle(prepared.board, annotation["label_center_source_px"])
            error = (result.value.correction_deg + observed_angle + 180) % 360 - 180
            assert abs(error) <= annotation["angle_tolerance_deg"], annotation["image"]
