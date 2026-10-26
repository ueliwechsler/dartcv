import cv2
import numpy as np
import pytest

from dartcv import canny_edges


def test_canny_edges_returns_uint8_map_without_mutating_input():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[16:48, 16:48] = 255
    original = image.copy()

    edges = canny_edges(
        image,
        low_threshold=40,
        high_threshold=120,
        blur_kernel=(3, 3),
        blur_sigma=0.5,
    )

    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
    assert np.count_nonzero(edges) > 0
    np.testing.assert_array_equal(image, original)


def test_canny_edges_converts_bgr_input_to_grayscale():
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[16:48, 16:48] = 255
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    np.testing.assert_array_equal(canny_edges(bgr), canny_edges(gray))


def test_canny_edges_rejects_invalid_input_and_parameters():
    with pytest.raises(TypeError, match="dtype uint8"):
        canny_edges(np.zeros((16, 16), dtype=np.float32))

    with pytest.raises(ValueError, match="grayscale or a three-channel BGR"):
        canny_edges(np.zeros((16, 16, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="thresholds must satisfy"):
        canny_edges(np.zeros((16, 16), dtype=np.uint8), 120, 40)

    with pytest.raises(ValueError, match="positive odd integers"):
        canny_edges(np.zeros((16, 16), dtype=np.uint8), blur_kernel=(4, 5))
