import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

import dartcv.color_dart_pipeline as pipeline_module
from dartcv.color_dart_inspection import draw_board_geometry, plot_board_preparation
from dartcv.color_dart_pipeline import (
    BoardEntryEstimate,
    BoardGeometry,
    ColorBasedDartRecognitionPipeline,
    ColorPipelineStatus,
    DartLikelihoodMap,
    EntryProvenance,
    OrientationEstimate,
    ScoringRing,
    ScoreEstimate,
    StageOutcome,
    StageStatus,
    UncertaintyReason,
    detect_board_stage,
    format_timing_summary,
    normalize_board_view,
    prepare_board,
)
from dartcv.detect import DetectionError


def _test_image() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _board() -> BoardGeometry:
    return BoardGeometry(
        model_to_source=np.array(
            [[0.1, 0.0, 50.0], [0.0, 0.1, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        detector_rotation_deg=4.0,
    )


def _pipeline(
    *,
    board_localization=None,
    board_normalization=normalize_board_view,
    orientation=None,
    segmentation=None,
    tip_localization=None,
    scoring=None,
    diagnostics_size=64,
):
    def localize(_image, *, diagnostics):
        del diagnostics
        return StageOutcome.succeeded(_board())

    def find_orientation(_view, *, diagnostics):
        del diagnostics
        return StageOutcome.succeeded(
            OrientationEstimate(correction_deg=0.0, confidence=0.95, confident=True)
        )

    def segment(view, _board, _orientation, *, diagnostics):
        del diagnostics
        return StageOutcome.succeeded(
            DartLikelihoodMap(np.zeros(view.image.shape[:2], dtype=np.float32))
        )

    def locate_tips(
        _image, _view, _board, _orientation, _likelihood, expected_count, *, diagnostics
    ):
        del diagnostics
        return StageOutcome.succeeded(
            tuple(
                BoardEntryEstimate(
                    xy=(float(index * 10), float(index * 5)),
                    confidence=0.9,
                    provenance=EntryProvenance.OBSERVED,
                )
                for index in range(min(expected_count, 2))
            )
        )

    def score(_board, _orientation, entries, _margin, *, diagnostics):
        del diagnostics
        return StageOutcome.succeeded(
            tuple(
                None
                if entry.xy is None
                else ScoreEstimate(sector=20, ring=ScoringRing.SINGLE, points=20)
                for entry in entries
            )
        )

    return ColorBasedDartRecognitionPipeline(
        board_localization=board_localization or localize,
        board_normalization=board_normalization,
        orientation=orientation or find_orientation,
        segmentation=segmentation or segment,
        tip_localization=tip_localization or locate_tips,
        scoring=scoring or score,
        normalized_size_px=diagnostics_size,
    )


def test_normalize_board_view_maps_board_center_to_canvas_center():
    result = normalize_board_view(_test_image(), _board(), 64, diagnostics=False)

    assert result.value is not None
    view = result.value
    assert view.image.shape == (64, 64, 3)
    assert view.board_mask.shape == (64, 64)
    center = view.source_to_view @ np.array([50.0, 50.0, 1.0])
    assert center[0] / center[2] == pytest.approx(31.5)
    assert center[1] / center[2] == pytest.approx(31.5)
    assert view.board_mask[32, 32] == 255
    assert view.board_mask[0, 0] == 0


def test_pipeline_orders_entries_and_keeps_missing_slots_explicit():
    result = _pipeline().run(_test_image(), expected_dart_count=3)

    assert result.status is ColorPipelineStatus.COMPLETED
    assert len(result.darts) == 3
    assert [dart.entry.xy for dart in result.darts] == [
        (0.0, 0.0),
        (10.0, 5.0),
        None,
    ]
    assert result.darts[2].score is None
    assert UncertaintyReason.ENTRY_UNLOCALIZED in result.darts[2].uncertainty_reasons
    assert result.total_points is None
    assert not result.total_complete
    assert result.total_uncertain
    assert len(result.steps) == 6
    assert all(step.elapsed_ms >= 0.0 for step in result.steps)
    assert "Total:" in format_timing_summary(result)


def test_pipeline_aggregates_when_each_expected_dart_has_a_score():
    def locate_two(
        _image, _view, _board, _orientation, _likelihood, _count, *, diagnostics
    ):
        del diagnostics
        return StageOutcome.succeeded(
            (
                BoardEntryEstimate((30.0, 20.0), 0.9, EntryProvenance.OBSERVED),
                BoardEntryEstimate((10.0, 40.0), 0.9, EntryProvenance.OBSERVED),
            )
        )

    result = _pipeline(tip_localization=locate_two).run(
        _test_image(), expected_dart_count=2
    )

    assert result.total_points == 40
    assert result.total_complete
    assert not result.total_uncertain
    assert all(dart.score is not None for dart in result.darts)


def test_pipeline_preserves_inferred_best_guess_and_uncertainty_reasons():
    def find_orientation(_view, *, diagnostics):
        del diagnostics
        return StageOutcome.succeeded(
            OrientationEstimate(correction_deg=36.0, confidence=0.4, confident=False)
        )

    def locate_tip(
        _image, _view, _board, _orientation, _likelihood, _count, *, diagnostics
    ):
        del diagnostics
        return StageOutcome.succeeded(
            (
                BoardEntryEstimate(
                    xy=(20.0, 30.0),
                    confidence=0.6,
                    provenance=EntryProvenance.INFERRED,
                ),
            )
        )

    def score(_board, _orientation, _entries, _margin, *, diagnostics):
        del diagnostics
        return StageOutcome.succeeded(
            (
                ScoreEstimate(
                    sector=20,
                    ring=ScoringRing.SINGLE,
                    points=20,
                    uncertainty_reasons=(UncertaintyReason.SCORE_BOUNDARY,),
                ),
            )
        )

    result = _pipeline(
        orientation=find_orientation,
        tip_localization=locate_tip,
        scoring=score,
    ).run(_test_image(), expected_dart_count=1)

    dart = result.darts[0]
    assert dart.score is not None
    assert dart.score.points == 20
    assert result.total_points == 20
    assert result.total_complete
    assert result.total_uncertain
    assert set(dart.uncertainty_reasons) == {
        UncertaintyReason.ENTRY_INFERRED,
        UncertaintyReason.SCORE_BOUNDARY,
        UncertaintyReason.ORIENTATION_UNCERTAIN,
    }


def test_scoring_failure_keeps_completed_entry_estimates():
    def scoring_failure(_board, _orientation, _entries, _margin, *, diagnostics):
        del diagnostics
        return StageOutcome.failed("scoring unavailable")

    result = _pipeline(scoring=scoring_failure).run(
        _test_image(), expected_dart_count=2
    )

    assert result.status is ColorPipelineStatus.PARTIAL
    assert result.failure is not None
    assert result.failure.stage == "scoring"
    assert result.darts[0].entry.xy == (0.0, 0.0)
    assert result.darts[1].entry.xy == (10.0, 5.0)
    assert all(dart.score is None for dart in result.darts)


def test_expected_board_failure_returns_partial_result_and_skips_dependents():
    def board_missing(_image, *, diagnostics):
        del diagnostics
        return StageOutcome.failed("board not found")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("dependent stage should have been skipped")

    result = _pipeline(
        board_localization=board_missing,
        orientation=must_not_run,
        segmentation=must_not_run,
        tip_localization=must_not_run,
        scoring=must_not_run,
    ).run(_test_image())

    assert result.status is ColorPipelineStatus.PARTIAL
    assert result.failure is not None
    assert result.failure.message == "board not found"
    assert len(result.darts) == 3
    assert all(dart.score is None for dart in result.darts)
    assert [step.status for step in result.steps] == [
        StageStatus.FAILED,
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
    ]


def test_diagnostics_are_opt_in_independent_snapshots():
    def localize(image, *, diagnostics):
        return StageOutcome.succeeded(
            _board(),
            artifacts={"received_image": image} if diagnostics else {},
        )

    result = _pipeline(board_localization=localize).run(
        _test_image(), expected_dart_count=1, diagnostics=True
    )

    captured = result.steps[0].output_artifacts["received_image"]
    assert not np.shares_memory(captured, result.steps[0].input_artifacts["image"])
    assert not np.shares_memory(captured, _test_image())

    without_diagnostics = _pipeline().run(_test_image(), expected_dart_count=1)
    assert all(
        not step.input_artifacts and not step.output_artifacts
        for step in without_diagnostics.steps
    )


def test_detector_adapter_turns_only_detection_failure_into_expected_failure(monkeypatch):
    def raise_detection_error(*_args, **_kwargs):
        raise DetectionError("no board")

    monkeypatch.setattr(pipeline_module, "detect_board", raise_detection_error)
    outcome = detect_board_stage(_test_image(), diagnostics=False)

    assert outcome.failure == "no board"
    assert outcome.value is None

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pipeline_module, "detect_board", raise_runtime_error)
    with pytest.raises(RuntimeError, match="unexpected"):
        detect_board_stage(_test_image(), diagnostics=False)


@pytest.mark.parametrize(
    ("image", "expected_count"),
    [
        (np.zeros((20, 20), dtype=np.uint8), 3),
        (np.zeros((20, 20, 4), dtype=np.uint8), 3),
        (np.zeros((20, 20, 3), dtype=np.float32), 3),
        (_test_image(), 0),
    ],
)
def test_pipeline_rejects_invalid_inputs(image, expected_count):
    with pytest.raises(ValueError):
        _pipeline().run(image, expected_dart_count=expected_count)


def test_constructor_requires_unimplemented_stages():
    with pytest.raises(TypeError):
        ColorBasedDartRecognitionPipeline()


def test_dart_likelihood_rejects_invalid_probability_map():
    with pytest.raises(ValueError):
        DartLikelihoodMap(np.full((2, 2), 1.1, dtype=np.float32))


def test_board_only_preparation_returns_view_timings_and_stable_snapshots():
    def localize(image, *, diagnostics):
        return StageOutcome.succeeded(
            _board(), artifacts={"source": image} if diagnostics else {}
        )

    image = _test_image()
    result = prepare_board(
        image, board_localization=localize, normalized_size_px=64, diagnostics=True
    )
    assert result.failure is None
    assert result.board_view is not None
    assert result.board_view.image.shape == (64, 64, 3)
    assert len(result.steps) == 2
    assert "board_normalization:" in format_timing_summary(result)
    image[:] = 255
    assert np.all(result.steps[0].input_artifacts["image"] == 0)
    assert np.all(result.steps[0].output_artifacts["source"] == 0)


def test_board_only_detection_failure_keeps_diagnostics_and_skips_normalization():
    def missing(image, *, diagnostics):
        return StageOutcome.failed(
            "no board",
            artifacts={"colour_mask": image[:, :, 0]} if diagnostics else {},
        )

    def must_not_run(*args, **kwargs):
        raise AssertionError("Normalization must be skipped.")

    result = prepare_board(
        _test_image(),
        board_localization=missing,
        board_normalization=must_not_run,
        diagnostics=True,
    )
    assert result.board is None
    assert result.board_view is None
    assert result.failure is not None
    assert result.failure.stage == "board_localization"
    assert result.steps[1].status is StageStatus.SKIPPED
    figure = plot_board_preparation(result, max_display_px=80)
    FigureCanvasAgg(figure).draw()
    assert any("no board" in text.get_text() for text in figure.texts)
    figure.clear()


def test_normalization_failure_is_consistent_in_board_only_and_full_runs():
    def localize(image, *, diagnostics):
        return StageOutcome.succeeded(_board())

    def unavailable(image, board, output_size_px, *, diagnostics):
        return StageOutcome.failed("unusable board view")

    result = prepare_board(
        _test_image(),
        board_localization=localize,
        board_normalization=unavailable,
    )
    assert result.board is not None
    assert result.board_view is None
    assert result.failure is not None
    assert result.failure.stage == "board_normalization"
    pipeline = _pipeline(board_normalization=unavailable)
    full = pipeline.run(_test_image())
    assert full.failure == result.failure
    assert [step.status for step in full.steps] == [
        StageStatus.SUCCEEDED,
        StageStatus.FAILED,
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
    ]


def test_inspection_renders_rgb_without_changing_source_or_geometry():
    def localize(image, *, diagnostics):
        return StageOutcome.succeeded(_board())

    image = _test_image()
    image[:, :, 0] = 255
    source = image.copy()
    board = _board()
    transform = board.model_to_source.copy()
    overlay = draw_board_geometry(image, board.model_to_source)
    assert not np.array_equal(overlay, image)
    np.testing.assert_array_equal(image, source)
    np.testing.assert_array_equal(board.model_to_source, transform)
    result = prepare_board(
        image, board_localization=localize, normalized_size_px=64, diagnostics=True
    )
    figure = plot_board_preparation(result, max_display_px=80)
    FigureCanvasAgg(figure).draw()
    assert len(figure.axes) == 6
    assert tuple(figure.axes[0].images[0].get_array()[0, 0]) == (0, 0, 255)
    assert figure.axes[0].images[0].get_array().shape[:2] == (80, 80)
    figure.clear()
    np.testing.assert_array_equal(image, source)


def test_inspection_requires_explicit_diagnostic_capture():
    def localize(image, *, diagnostics):
        return StageOutcome.succeeded(_board())

    result = prepare_board(
        _test_image(), board_localization=localize, normalized_size_px=64
    )
    assert all(not step.input_artifacts for step in result.steps)
    with pytest.raises(ValueError, match="diagnostics=True"):
        plot_board_preparation(result)


@pytest.mark.parametrize("size", [True, 0, 1.5])
def test_board_only_rejects_invalid_output_size_before_detection(size):
    def must_not_run(*args, **kwargs):
        raise AssertionError("Validation must happen before detection.")

    with pytest.raises(ValueError, match="integer"):
        prepare_board(_test_image(), normalized_size_px=size, board_localization=must_not_run)
