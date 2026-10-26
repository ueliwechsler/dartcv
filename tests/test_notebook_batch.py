from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dartcv.detect import DetectionError
from dartcv.notebook_batch import BatchOutput, image_batch_figures


def test_batch_pages_show_final_rgb_and_failures_without_hiding_bugs():
    calls = []
    source = np.full((100, 200, 3), (255, 0, 0), np.uint8)

    def process(path):
        calls.append(path)
        if path.name == "bad":
            raise DetectionError("No board")
        return BatchOutput(source, "done")

    pages = image_batch_figures(map(Path, ("first", "bad", "last")), process,
                               page_size=2, preview_size_px=20)
    first = next(pages)
    assert len(calls) == 2
    pixels = first.axes[0].images[0].get_array()
    assert pixels.shape == (10, 20, 3)
    np.testing.assert_array_equal(pixels[0, 0], [0, 0, 255])
    assert "bad" in first.axes[1].get_title() and "FAILED" in first.axes[1].get_title()
    FigureCanvasAgg(first).draw()
    last = next(pages)
    assert "last" in last.axes[0].get_title()
    with pytest.raises(StopIteration):
        next(pages)

    def broken(path):
        raise RuntimeError("Unexpected")

    with pytest.raises(RuntimeError, match="Unexpected"):
        next(image_batch_figures([Path("first")], broken))


def test_empty_batch_and_invalid_options_are_explicit():
    with pytest.raises(ValueError, match="at least one"):
        next(image_batch_figures([], lambda path: None))
    with pytest.raises(ValueError, match="positive"):
        next(image_batch_figures([Path("first")], lambda path: None, page_size=0))
