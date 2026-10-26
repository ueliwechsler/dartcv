from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dartcv import BoardGeometry, StageOutcome, normalize_board_view
from dartcv.color_dart_evaluation import (
    evaluate_sparse_probes, plot_segmentation_comparison, segmentation_variants,
)


def test_probe_projection_respects_reference_frame_and_missing_source_coverage():
    board = BoardGeometry(np.array([[1., 0, 50], [0, 1., 50], [0, 0, 1]]), 0)
    view = normalize_board_view(np.zeros((101, 101, 3), np.uint8), board, 101,
                                diagnostics=False).value
    probabilities = np.zeros((101, 101), np.float32)
    probabilities[48:53, 48:53] = 1
    metrics = evaluate_sparse_probes(
        view, board, probabilities, board.model_to_source,
        {"shaft": [[383.5, 383.5]], "board": [[0, 0]]},
    )
    assert metrics["shaft"].positive_rate == 1
    assert metrics["board"].eligible == 0 and metrics["board"].excluded == 1
    assert metrics["board"].positive_rate is None
    coverage = view.source_coverage.copy()
    coverage[50, 50] = 0
    metrics = evaluate_sparse_probes(
        replace(view, source_coverage=coverage), board, probabilities, board.model_to_source,
        {"shaft": [[383.5, 383.5]]},
    )
    assert metrics["shaft"].excluded == 1
    with pytest.raises(ValueError, match="category"):
        evaluate_sparse_probes(view, board, probabilities, board.model_to_source,
                               {"unknown": [[0, 0]]})


def test_comparison_is_complete_and_failures_remain_inspectable():
    variants = segmentation_variants()
    assert len(variants) == 9
    assert set(variants) == {
        f"{assignment}/{cleanup}"
        for assignment in ("field", "board_union", "global")
        for cleanup in ("none", "hard", "support")
    }
    board = BoardGeometry(np.eye(3), 0)
    view = normalize_board_view(np.zeros((101, 101, 3), np.uint8), board, 101,
                               diagnostics=False).value
    figure = plot_segmentation_comparison(view, {"field/hard": StageOutcome.failed("No samples")})
    FigureCanvasAgg(figure).draw()
    assert "No samples" in figure.axes[0].texts[0].get_text()
