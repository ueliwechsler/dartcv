import cv2
import numpy as np
import pytest

from dartcv.classical_darts import ClassicalDartDetector
from dartcv.recognition import validate_boxes


def synthetic_dart(include_flight=True, include_shaft=True, colour=(230, 65, 25)):
    image = np.full((480, 640, 3), 170, np.uint8)
    cv2.circle(image, (320, 240), 180, (100, 100, 100), 2)
    cv2.circle(image, (320, 240), 105, (220, 220, 220), 2)
    for angle in range(0, 360, 18):
        direction = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
        end = np.round(np.array([320, 240]) + direction * 180).astype(int)
        cv2.line(image, (320, 240), tuple(end), (210, 210, 210), 1)
    if include_shaft:
        cv2.line(image, (380, 155), (330, 290), (25, 25, 25), 6)
    if include_flight:
        flight = np.array([[380, 120], [401, 145], [380, 170], [359, 145]])
        cv2.fillConvexPoly(image, flight, colour)
    return image, np.array([[1, 0, 320], [0, 1, 240], [0, 0, 1]], float)


@pytest.mark.parametrize("known_colour", [False, True])
def test_classical_detects_whole_synthetic_dart(known_colour):
    image, homography = synthetic_dart()
    options = {"flight_hsv": ((95, 80, 50), (125, 255, 255))} if known_colour else {}
    detector = ClassicalDartDetector(homography, **options)
    image.flags.writeable = False
    boxes = detector(image)
    assert len(boxes) == 1, detector.diagnostics.message
    validate_boxes(image, boxes)
    left, top, right, bottom = boxes[0].xyxy
    assert left < 335 and right > 395
    assert top < 125 and bottom > 280
    assert detector.diagnostics.accepted
    assert set(detector.diagnostic_views()) == {
        "search",
        "flight_mask",
        "shaft_segments",
        "hypotheses",
    }


@pytest.mark.parametrize("flight,shaft", [(False, False), (True, False), (False, True)])
def test_classical_does_not_force_one_dart(flight, shaft):
    image, homography = synthetic_dart(flight, shaft)
    detector = ClassicalDartDetector(homography)
    assert detector(image) == ()
    assert not detector.diagnostics.accepted


def test_classical_resizing_preserves_source_frame():
    image, homography = synthetic_dart()
    enlarged = cv2.resize(image, None, fx=2, fy=2)
    detector = ClassicalDartDetector(np.diag([2, 2, 1]) @ homography, work_size=640)
    boxes = detector(enlarged)
    assert len(boxes) == 1
    assert boxes[0].xyxy[3] > 560
    np.testing.assert_allclose(detector.diagnostics.scale_xy, [0.5, 0.5])


def test_classical_detects_white_flight():
    image, homography = synthetic_dart(colour=(255, 255, 255))
    detector = ClassicalDartDetector(homography)
    boxes = detector(image)
    assert len(boxes) == 1, detector.diagnostics.message
    assert boxes[0].xyxy[1] < 125 and boxes[0].xyxy[3] > 280


def test_classical_rejects_bull_with_attached_wire():
    image, homography = synthetic_dart(False, False)
    cv2.circle(image, (320, 240), 16, (40, 150, 40), -1)
    cv2.circle(image, (320, 240), 6, (30, 30, 210), -1)
    cv2.line(image, (335, 240), (400, 240), (15, 15, 15), 5)
    detector = ClassicalDartDetector(homography)
    assert detector(image) == ()


def test_classical_hue_wrapping_and_off_image_board():
    image, homography = synthetic_dart(colour=(0, 0, 255))
    detector = ClassicalDartDetector(
        homography, flight_hsv=((170, 80, 50), (10, 255, 255))
    )
    assert len(detector(image)) == 1
    homography[:2, 2] += 10000
    detector = ClassicalDartDetector(homography)
    assert detector(image) == ()
    assert not detector.diagnostics.search_mask.any()


def test_classical_abstains_for_competing_darts():
    image, homography = synthetic_dart()
    cv2.line(image, (260, 155), (310, 290), (25, 25, 25), 6)
    cv2.fillConvexPoly(
        image, np.array([[260, 120], [281, 145], [260, 170], [239, 145]]), (230, 65, 25)
    )
    detector = ClassicalDartDetector(homography, ambiguity_margin=1.0)
    assert detector(image) == ()
    assert "ambiguous" in detector.diagnostics.message


@pytest.mark.parametrize(
    "options",
    [
        {"work_size": 0},
        {"min_score": np.nan},
        {"flight_hsv": ((180, 0, 0), (190, 255, 255))},
    ],
)
def test_classical_rejects_bad_configuration(options):
    with pytest.raises(ValueError):
        ClassicalDartDetector(np.eye(3), **options)


def test_classical_rejects_invalid_input_and_homography():
    with pytest.raises(ValueError):
        ClassicalDartDetector(np.zeros((3, 3)))
    detector = ClassicalDartDetector(np.eye(3))
    assert detector.diagnostic_views() == {}
    with pytest.raises(ValueError, match="BGR uint8"):
        detector(np.zeros((20, 20), np.uint8))
