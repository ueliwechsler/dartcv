"""Small output-only image batches for interactive feature notebooks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from matplotlib.figure import Figure

from .color_dart_pipeline import Image, _validate_image
from .detect import DetectionError


@dataclass(frozen=True, slots=True)
class BatchOutput:
    """One final BGR/grayscale output and its explicit status, not intermediates."""

    image: Image | None
    status: str


def read_batch_image(path: Path) -> Image:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def image_batch_figures(
    paths: Iterable[Path],
    process: Callable[[Path], BatchOutput],
    *,
    page_size: int = 6,
    columns: int = 3,
    preview_size_px: int = 600,
) -> Iterator[Figure]:
    """Process sequentially, yielding bounded pages with only final thumbnails.

    Each tile names its source file, elapsed processing time and final status.
    Expected detection/read failures are visible tiles; unexpected bugs propagate.
    No display, writes, debug printing, or retained full-resolution intermediates.
    """
    for name, value in (
        ("page size", page_size), ("columns", columns), ("preview size", preview_size_px)
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Batch {name} must be a positive integer.")
    iterator = iter(paths)
    processed = 0
    while True:
        page = []
        for _ in range(page_size):
            path = next(iterator, None)
            if path is None:
                break
            page.append(Path(path))
        if not page:
            if not processed:
                raise ValueError("Select at least one batch image.")
            return
        ncols = min(columns, len(page))
        figure = Figure(figsize=(5 * ncols, 5 * ((len(page) + ncols - 1) // ncols)),
                        layout="constrained")
        axes = figure.subplots((len(page) + ncols - 1) // ncols, ncols, squeeze=False)
        for axis in axes.flat:
            axis.set_axis_off()
        for axis, path in zip(axes.flat, page):
            start = perf_counter()
            try:
                output = process(path)
            except (DetectionError, FileNotFoundError) as exc:
                output = BatchOutput(None, f"FAILED: {exc}")
            milliseconds = (perf_counter() - start) * 1000
            if output.image is not None:
                image = output.image
                if image.ndim == 3:
                    _validate_image(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.ndim != 2 or image.dtype != np.uint8:
                    raise ValueError("Batch outputs must be uint8 BGR or grayscale images.")
                height, width = image.shape[:2]
                scale = min(1.0, preview_size_px / max(height, width))
                preview = cv2.resize(
                    image, (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                axis.imshow(preview, cmap="gray", vmin=0, vmax=255)
                del image
            else:
                axis.text(.5, .5, output.status, ha="center", va="center", wrap=True, color="red")
            axis.set_title(f"{path.name}\n{output.status}\n{milliseconds:.0f} ms", fontsize=9)
            del output
            processed += 1
        yield figure
