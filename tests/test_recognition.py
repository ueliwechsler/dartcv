import builtins
import json
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dartcv.dart_detection import Owlv2DartDetector
from dartcv.geometry import SECTORS, polar
from dartcv.recognition import (
    DartBox,
    DartTip,
    NumberReading,
    expected_sector,
    run_stage,
    validate_boxes,
    validate_numbers,
    validate_tips,
)
from dartcv.recognition_overlay import annotate


@pytest.fixture
def image():
    return np.zeros((100, 120, 3), np.uint8)


def test_stage_distinguishes_missing_empty_and_failed(image):
    validate = lambda items: validate_boxes(image, items)
    assert run_stage("darts", None, validate, image).status == "not configured"
    empty = run_stage("darts", lambda _: [], validate, image)
    assert empty.status == "ok" and empty.items == ()

    def broken(_):
        yield DartBox(1, (10, 10, 40, 50), 0.9, "test")
        raise RuntimeError("inference failed")

    failed = run_stage("darts", broken, validate, image)
    assert failed.status == "failed" and failed.items == ()
    assert "RuntimeError: inference failed" in failed.message


def test_stage_validates_and_preserves_observations(image):
    box = DartBox(1, (10, 10, 40, 50), 0.9, "test")
    validate = lambda items: validate_boxes(image, items)
    result = run_stage("darts", lambda _: [box], validate, image)
    assert result.status == "ok" and result.items == (box,)
    assert run_stage("darts", lambda _: [box, box], validate, image).status == "failed"


@pytest.mark.parametrize(
    "xyxy", [(10, 10, 10, 20), (-1, 0, 20, 20), (0, 0, 120, 99), (0, 0, np.nan, 20)]
)
def test_invalid_boxes(image, xyxy):
    with pytest.raises(ValueError):
        validate_boxes(image, [DartBox(1, xyxy, 0.9, "test")])


@pytest.mark.parametrize("confidence", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_confidence(image, confidence):
    with pytest.raises(ValueError):
        validate_boxes(image, [DartBox(1, (0, 0, 20, 20), confidence, "test")])


def test_number_readings_preserve_observed_text(image):
    polygon = np.array([[10, 10], [30, 10], [20, 30]])
    reading = NumberReading(polygon, "5", 0.8, "test")
    validate_numbers(image, [reading])
    assert reading.text == "5"
    for invalid in (polygon[:2], polygon + 200, np.array([1, 2])):
        with pytest.raises(ValueError):
            validate_numbers(image, [NumberReading(invalid, "5", 0.8, "test")])
    with pytest.raises(ValueError, match="empty number"):
        validate_numbers(image, [NumberReading(polygon, " ", 0.8, "test")])


def test_tip_associations_and_padding(image):
    boxes = [DartBox(1, (20, 20, 50, 60), 0.9, "test")]
    validate_tips(image, [DartTip(1, (18, 20), 0.9, "test")], boxes)
    validate_tips(image, [DartTip(None, (0, 0), 1.0, "manual")], ())
    for tip in (
        DartTip(2, (30, 30), 0.9, "test"),
        DartTip(1, (90, 90), 0.9, "test"),
        DartTip(None, (np.nan, 20), 0.9, "test"),
        DartTip(None, (120, 20), 0.9, "test"),
        DartTip(None, (20, 20, 30, 30), 0.9, "test"),
    ):
        with pytest.raises(ValueError):
            validate_tips(image, [tip], boxes)


def test_expected_sectors():
    for index, sector in enumerate(SECTORS):
        assert expected_sector(np.eye(3), polar(200, 18 * index)) == sector


def test_annotations_do_not_mutate_input(image):
    image.flags.writeable = False
    boxes = [DartBox(1, (20, 20, 50, 60), 0.9, "test")]
    numbers = [
        NumberReading(np.array([[60, 20], [80, 20], [70, 40]]), "5", 0.8, "test")
    ]
    tips = [DartTip(1, (30, 40), 0.9, "test")]
    result = annotate(image, boxes, numbers, tips, [20])
    assert result.shape == image.shape and result.any()
    assert not image.any() and not np.shares_memory(image, result)
    assert np.array_equal(annotate(image), image)
    with pytest.raises(ValueError, match="one entry per tip"):
        annotate(image, tips=tips, scores=[])


def test_owlv2_filters_and_maps_boxes_without_mutating_input(image, monkeypatch):
    detector = Owlv2DartDetector(prompts=("a dart", "a throwing dart"))
    image[:] = (10, 20, 30)
    image.flags.writeable = False

    def predict(rgb):
        assert rgb.shape == (100, 120, 3)
        assert tuple(rgb[0, 0]) == (30, 20, 10)
        return dict(
            boxes=[
                [-2, 10, 40, 70],
                [0, 10, 40, 70],
                [80, 70, 125, 105],
                [20, 20, 30, 30],
                [200, 0, 220, 10],
            ],
            scores=[0.9, 0.8, 0.7, 0.05, 0.8],
            labels=[0, 1, 0, 0, 0],
        )

    monkeypatch.setattr(detector, "_predict", predict)
    result = detector(image)
    assert [box.id for box in result] == [1, 2]
    assert result[0].xyxy == (0, 10, 40, 70)
    assert result[1].xyxy == (80, 70, 119, 99)
    assert "a dart" in result[0].provenance
    assert tuple(image[0, 0]) == (10, 20, 30)
    assert detector(image) == result


def test_owlv2_empty_and_failed_predictions(image, monkeypatch):
    detector = Owlv2DartDetector()
    monkeypatch.setattr(
        detector,
        "_predict",
        lambda _: dict(boxes=np.empty((0, 4)), scores=[], labels=[]),
    )
    assert detector(image) == ()
    monkeypatch.setattr(
        detector,
        "_predict",
        lambda _: dict(boxes=[[0, 0, np.nan, 5]], scores=[0.9], labels=[0]),
    )
    result = run_stage(
        "darts", detector, lambda items: validate_boxes(image, items), image
    )
    assert result.status == "failed" and result.items == ()


@pytest.mark.parametrize(
    "options",
    [dict(prompts="a dart"), dict(prompts=()), dict(threshold=np.nan), dict(nms_iou=2)],
)
def test_owlv2_invalid_configuration(options):
    with pytest.raises(ValueError):
        Owlv2DartDetector(**options)


def test_owlv2_rejects_invalid_image_before_loading():
    detector = Owlv2DartDetector()
    with pytest.raises(ValueError, match="BGR uint8"):
        detector(np.zeros((10, 10), np.uint8))
    assert detector._model is None


def test_owlv2_loads_once_and_passes_original_target_size(image, monkeypatch):
    calls = []

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def __init__(self):
            self.image_processor = self

        @classmethod
        def from_pretrained(cls, model_id, **options):
            calls.append(("processor", model_id, options))
            return cls()

        def __call__(self, **inputs):
            assert inputs["images"].shape == (100, 120, 3)
            assert inputs["text"] == [["a dart"]]
            return Inputs(pixel_values="prepared")

        def post_process_object_detection(self, **options):
            assert options["target_sizes"].value.tolist() == [[100, 120]]
            assert options["threshold"] == 0.1
            return [
                {
                    "boxes": Tensor([[10, 20, 80, 90]]),
                    "scores": Tensor([0.9]),
                    "labels": Tensor([0]),
                }
            ]

    class Model:
        config = SimpleNamespace(_commit_hash="resolved-test-revision")

        @classmethod
        def from_pretrained(cls, model_id, **options):
            calls.append(("model", model_id, options))
            assert options["use_safetensors"] is True
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            assert inputs == {"pixel_values": "prepared"}
            return "raw predictions"

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=nullcontext, tensor=lambda value, **_: Tensor(value)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Owlv2Processor=Processor, Owlv2ForObjectDetection=Model),
    )
    detector = Owlv2DartDetector(local_files_only=True)
    assert not calls
    assert detector(image)[0].xyxy == (10, 20, 80, 90)
    assert "resolved-test-revision" in detector(image)[0].provenance
    assert len(calls) == 2
    assert all(call[2]["local_files_only"] for call in calls)


def test_owlv2_missing_runtime_has_actionable_error(image, monkeypatch):
    original_import = builtins.__import__

    def import_without_runtime(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_runtime)
    detector = Owlv2DartDetector()
    result = run_stage(
        "darts", detector, lambda items: validate_boxes(image, items), image
    )
    assert result.status == "failed" and result.items == ()
    assert "uv sync --extra dart-detection" in result.message
    assert detector._model is None


def test_owlv2_processor_reverses_square_padding_without_weights():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    processor = transformers.Owlv2ImageProcessor()
    outputs = SimpleNamespace(
        logits=torch.tensor([[[3.0]]]),
        pred_boxes=torch.tensor([[[0.25, 0.25, 0.1, 0.2]]]),
    )
    result = processor.post_process_object_detection(
        outputs, threshold=0.1, target_sizes=torch.tensor([[100, 120]])
    )[0]
    np.testing.assert_allclose(result["boxes"].numpy(), [[24, 18, 36, 42]], atol=1e-5)


@pytest.mark.parametrize(
    "result",
    [
        {"boxes": [[0, 0, 20, 20]], "scores": [1.1], "labels": [0]},
        {"boxes": [[0, 0, 20, 20]], "scores": [0.9], "labels": [1]},
        {"boxes": [[0, 0, 20, 20]], "scores": [0.9], "labels": [0.5]},
        {"boxes": [[0, 0, 20]], "scores": [0.9], "labels": [0]},
    ],
)
def test_owlv2_rejects_malformed_model_output(image, monkeypatch, result):
    detector = Owlv2DartDetector()
    monkeypatch.setattr(detector, "_predict", lambda _: result)
    with pytest.raises(ValueError, match="invalid OWLv2"):
        detector(image)


def test_owlv2_load_failure_is_not_a_successful_empty_result(image, monkeypatch):
    detector = Owlv2DartDetector(local_files_only=True)

    def load():
        raise OSError("weights are not cached")

    monkeypatch.setattr(detector, "_load", load)
    result = run_stage("darts", detector, lambda _: None, image)
    assert result.status == "failed" and result.items == ()
    assert "weights are not cached" in result.message


def test_notebook_reuses_session_detector_without_loading_weights(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/dart_recognition.ipynb"
    notebook = json.loads(path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "ENABLE_DART_DETECTION =" in "".join(cell["source"])
    )

    def run_setup(code, namespace):
        script = tmp_path / "notebook_setup.py"
        script.write_text(code)
        return runpy.run_path(str(script), init_globals=namespace)

    namespace = run_setup(source, {"H": np.eye(3)})
    detector = namespace["DART_DETECTOR"]
    previous_diagnostics = object()
    detector.diagnostics = previous_diagnostics
    namespace = run_setup(source, namespace)
    assert namespace["DART_DETECTOR"] is detector
    assert namespace["DART_DETECTOR"].diagnostics is previous_diagnostics
    namespace = run_setup(
        source.replace("ENABLE_DART_DETECTION = True", "ENABLE_DART_DETECTION = False"),
        namespace,
    )
    assert namespace["DART_DETECTOR"] is None
    namespace = run_setup(source, namespace)
    assert namespace["DART_DETECTOR"] is detector
    namespace = run_setup(
        source.replace("CLASSICAL_MIN_SCORE = 0.65", "CLASSICAL_MIN_SCORE = 0.75"),
        namespace,
    )
    assert namespace["DART_DETECTOR"] is not detector
    assert namespace["DART_DETECTOR"].min_score == 0.75
    assert namespace["DART_DETECTOR"].diagnostics is None
    detector = namespace["DART_DETECTOR"]
    namespace["H"] = np.diag([2.0, 2.0, 1.0])
    namespace = run_setup(source, namespace)
    assert namespace["DART_DETECTOR"] is not detector
    np.testing.assert_array_equal(namespace["DART_DETECTOR"].H, namespace["H"])
