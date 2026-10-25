"""Standalone edge-detection helpers for grayscale and BGR images."""

from __future__ import annotations

from numbers import Integral

import cv2
import numpy as np
from numpy.typing import NDArray


def canny_edges(
    image: NDArray[np.uint8],
    low_threshold: float = 50,
    high_threshold: float = 150,
    blur_kernel: tuple[int, int] = (5, 5),
    blur_sigma: float = 0,
) -> NDArray[np.uint8]:
    """Return a Canny edge map for a grayscale or BGR ``uint8`` image.

    BGR input is converted to grayscale before a Gaussian blur and Canny edge
    detection. The input array is not modified. This helper is independent of
    the dartboard-detection pipeline.

    Args:
        image: Non-empty image with shape ``(height, width)`` for grayscale or
            ``(height, width, 3)`` for OpenCV BGR color.
        low_threshold: Lower Canny hysteresis threshold. Weak edges are kept
            only when connected to an edge stronger than ``high_threshold``.
        high_threshold: Upper Canny hysteresis threshold. Must be greater than
            or equal to ``low_threshold``.
        blur_kernel: Gaussian kernel size as ``(width, height)``. Both values
            must be positive odd integers. Larger kernels smooth over a wider
            neighborhood and can remove fine edges along with noise.
        blur_sigma: Gaussian sigma in the x direction. Zero asks OpenCV to
            derive sigma from ``blur_kernel``; a larger positive value applies
            stronger smoothing. OpenCV derives the y sigma from the x sigma.

    Returns:
        A two-dimensional ``uint8`` array with value 255 at detected edges and
        0 elsewhere.

    Raises:
        TypeError: If ``image`` is not a NumPy array or is not ``uint8``.
        ValueError: If the image is empty or has an unsupported shape, the
            thresholds are invalid, the blur kernel is invalid, or sigma is
            negative/non-finite.

    Example:
        >>> edges = canny_edges(bgr_image, low_threshold=40, high_threshold=120)
        >>> edges.shape == bgr_image.shape[:2]
        True
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    if image.size == 0:
        raise ValueError("image must not be empty")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be grayscale or a three-channel BGR image")

    try:
        thresholds = np.asarray((low_threshold, high_threshold), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Canny thresholds must be finite numbers") from exc
    if not np.isfinite(thresholds).all():
        raise ValueError("Canny thresholds must be finite numbers")
    low, high = thresholds
    if low < 0 or high < low:
        raise ValueError("thresholds must satisfy 0 <= low_threshold <= high_threshold")

    try:
        kernel_values = tuple(blur_kernel)
    except TypeError as exc:
        raise ValueError("blur_kernel must contain two positive odd integers") from exc
    if len(kernel_values) != 2 or any(
        not isinstance(size, Integral) or size <= 0 or size % 2 == 0
        for size in kernel_values
    ):
        raise ValueError("blur_kernel must contain two positive odd integers")
    kernel = tuple(int(size) for size in kernel_values)

    try:
        sigma = float(blur_sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError("blur_sigma must be a finite nonnegative number") from exc
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("blur_sigma must be a finite nonnegative number")

    blurred = cv2.GaussianBlur(gray, kernel, sigma)
    return cv2.Canny(blurred, float(low), float(high))
