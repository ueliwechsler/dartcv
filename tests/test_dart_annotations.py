from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dartcv.color_dart_pipeline import DartLikelihoodMap, NormalizedBoardView, StageOutcome
from dartcv.dart_annotations import (
    annotation_path, evaluate_visible_masks, load_annotation, parse_annotation,
    plot_visible_mask_evaluation,
)


def labels():
    return {
        "schema_version": 1, "image": "photo.jpg", "width": 100, "height": 100,
        "sha256": "a" * 64, "status": "completed",
        "darts": [{
            "id": "dart-1",
            "visible": [
                {"part": "flight", "points": [[10, 10], [19, 10], [19, 19], [10, 19]]},
                {"part": "shaft_barrel", "points": [[20, 12], [29, 12], [29, 14], [20, 14]]},
            ],
            "silhouette": [[[10, 10], [40, 10], [40, 19], [10, 19]]],
            "tip_status": "exact", "tip": [29, 13], "contact": "embedded",
        }],
        "ignore": [[[50, 50], [59, 50], [59, 59], [50, 59]]],
    }


def view(size=100, transform=None):
    transform = np.eye(3) if transform is None else transform
    mask = np.full((size, size), 255, np.uint8)
    return NormalizedBoardView(
        np.zeros((size, size, 3), np.uint8), mask, transform,
        transform, np.linalg.inv(transform), source_coverage=mask.copy(),
    )


def test_round_trip_and_invalid_completed_or_unknown_labels():
    data = labels()
    annotation = parse_annotation(data)
    assert parse_annotation(json.loads(json.dumps(annotation.to_dict()))) == annotation
    assert annotation_path("labels", "photo.jpg").name == "photo.jpg.json"
    assert annotation_path("labels", "photo.png").name == "photo.png.json"
    for change, message in [
        (lambda d: d["darts"][0].update(visible=[]), "visible region"),
        (lambda d: d["darts"][0].update(tip_status="unknown"), "null coordinates"),
        (lambda d: d["darts"][0].update(tip=[100, 20]), "outside"),
        (lambda d: d.update(image="../photo.jpg"), "filename"),
        (lambda d: d["darts"].append(deepcopy(d["darts"][0])), "unique"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d["darts"][0]["visible"][0].update(points=[[0, 0], [10, 10], [20, 20]]), "non-collinear"),
    ]:
        invalid = deepcopy(data)
        change(invalid)
        with pytest.raises(ValueError, match=message):
            parse_annotation(invalid)
    data["status"] = "draft"
    data["darts"][0].update(visible=[], tip_status="unknown", tip=None)
    assert parse_annotation(data).darts[0].tip is None


@pytest.mark.parametrize("points", [
    [[10, 10], [30, 30], [10, 30], [30, 10]],
    [[10, 10], [20, 10], [20, 20], [10, 10], [5, 20], [5, 10]],
    [[10, 10], [10, 10], [30, 10], [30, 30], [10, 30], [10, 10]],
])
def test_crossing_touching_and_repeated_vertices_save_and_evaluate(points):
    data = labels()
    data["darts"][0]["visible"] = [{"part": "flight", "points": points}]
    annotation = parse_annotation(data)
    assert parse_annotation(annotation.to_dict()) == annotation
    result = evaluate_visible_masks(view(), np.ones((100, 100), np.float32), annotation)
    assert result.metrics.true_positive > 0
    assert result.metrics.recall == 1.0


def test_visible_evaluation_ignores_inferred_shapes_and_ambiguous_pixels():
    annotation = parse_annotation(labels())
    probabilities = np.zeros((100, 100), np.float32)
    probabilities[10:20, 10:20] = 1  # Flight retained; shaft missed.
    probabilities[50:60, 50:60] = 1  # Ignore region, not a false positive.
    probabilities[70:75, 70:75] = 1  # True board noise.
    result = evaluate_visible_masks(view(), probabilities, annotation)
    metrics = result.metrics
    assert metrics.true_positive == 100
    assert metrics.false_negative == 30
    assert metrics.false_positive == 25
    assert metrics.recall == pytest.approx(100 / 130)
    assert metrics.precision == 0.8
    assert metrics.iou == pytest.approx(100 / 155)
    assert metrics.part_recall == {"flight": 1.0, "shaft_barrel": 0.0, "uncertain": None}
    assert metrics.label_coverage == 1.0
    assert len(plot_visible_mask_evaluation(result).axes) == 1


def test_projection_coverage_and_no_applicable_labels_are_explicit():
    annotation = parse_annotation(labels())
    transform = np.array([[1., 0, -20], [0, 1, 0], [0, 0, 1]])
    normalized = view(30, transform)
    normalized.source_coverage[:13] = 0
    probabilities = np.ones((30, 30), np.float32)
    metrics = evaluate_visible_masks(normalized, probabilities, annotation).metrics
    assert metrics.true_positive == 20  # Only two rows of the shaft are observed.
    assert metrics.part_recall["flight"] is None
    assert metrics.part_recall["shaft_barrel"] == 1.0
    assert metrics.label_coverage == pytest.approx(20 / 130)
    normalized.board_mask[:] = 0
    metrics = evaluate_visible_masks(normalized, probabilities, annotation).metrics
    assert metrics.evaluated_pixels == 0
    assert metrics.recall is metrics.precision is metrics.iou is None
    assert metrics.label_coverage == 0
    data = labels()
    data["status"] = "draft"
    with pytest.raises(ValueError, match="Complete"):
        evaluate_visible_masks(view(), np.zeros((100, 100), np.float32), parse_annotation(data))


def test_loading_labels_checks_source_identity(tmp_path):
    data = labels()
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"changed image bytes")
    path = tmp_path / "photo.jpg.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source photograph"):
        load_annotation(path, source_image=source)


def test_optional_manual_score_defaults_and_old_labels_remain_readable():
    data = labels()
    assert parse_annotation(data).darts[0].score is None
    data["darts"][0]["score"] = {"number": 20}
    annotation = parse_annotation(data)
    assert annotation.darts[0].score.multiplier == "single"
    assert annotation.darts[0].score.points == 20
    assert annotation.to_dict()["darts"][0]["score"] == {"number": 20, "multiplier": "single"}
    for number, multiplier, points in [(20, "double", 40), (20, "triple", 60), (25, "single", 25), (25, "double", 50), (0, "single", 0)]:
        data["darts"][0]["score"] = {"number": number, "multiplier": multiplier}
        annotation = parse_annotation(data)
        assert annotation.darts[0].score.points == points
        assert parse_annotation(annotation.to_dict()) == annotation
    for score in (
        {"number": 21}, {"number": True}, {"number": 20, "multiplier": "quadruple"},
        {"number": 25, "multiplier": "triple"}, {"number": 0, "multiplier": "double"},
        {"number": 20, "points": 60},
    ):
        data["darts"][0]["score"] = score
        with pytest.raises(ValueError, match="Score|miss"):
            parse_annotation(data)


def test_approximate_tip_round_trip_and_required_coordinates():
    data = labels()
    data["darts"][0]["tip_status"] = "approximate"
    annotation = parse_annotation(data)
    assert annotation.darts[0].tip == (29, 13)
    assert annotation.darts[0].tip_status == "approximate"
    assert parse_annotation(annotation.to_dict()) == annotation
    data["darts"][0]["tip"] = None
    with pytest.raises(ValueError, match="xy pairs"):
        parse_annotation(data)


@pytest.mark.parametrize("status", ["completed", "draft", "missing"])
def test_comparison_notebook_consumes_saved_labels(tmp_path, capsys, status):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "examples/color_dart_segmentation_comparison.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "visible-label-evaluation")
    source_image = tmp_path / "photo.jpg"
    source_image.write_bytes(b"source identity for integration")
    data = labels()
    data["sha256"] = sha256(source_image.read_bytes()).hexdigest()
    output = tmp_path / "data/annotations/darts"
    output.mkdir(parents=True)
    if status != "missing":
        data["status"] = status
        annotation_path(output, source_image.name).write_text(json.dumps(data))
    figures = []
    namespace = {
        "ROOT": tmp_path, "image_path": source_image,
        "preparation": SimpleNamespace(board_view=view()),
        "variants": {"field/support": StageOutcome.succeeded(
            DartLikelihoodMap(np.zeros((100, 100), np.float32))
        )},
        "display": figures.append,
    }
    exec("".join(cell["source"]).replace("RUN_LABEL_EVALUATION = False", "RUN_LABEL_EVALUATION = True"), namespace)
    text = capsys.readouterr().out
    if status == "completed":
        assert "field/support" in text and "n/a" in text and "100.0%" in text
        assert len(figures) == 1
    else:
        assert not figures
        expected = "draft labels are excluded" if status == "draft" else "No visible-region labels"
        assert expected in text


def test_editor_history_and_polygon_layers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed only for the browser editor's pure-state tests.")
    editor = Path(__file__).resolve().parents[1] / "src/dartcv/annotation_ui/editor.js"
    result = subprocess.run([node, "-e", """
const assert = require('node:assert/strict');
const E = require(process.argv[1]);
const initial = JSON.parse(process.argv[2]);
const history = new E.History(initial);
const next = E.clone(history.document);
const points = [[1,1], [9,1], [5,8]];
E.addPolygon(next, 0, 'silhouette', points);
E.addPolygon(next, 0, 'ignore', points);
E.setTip(next.darts[0], null);
E.setScore(next.darts[0], 20);
assert.deepEqual(next.darts[0].score, {number:20, multiplier:'single'});
E.setScore(next.darts[0], 20, 'triple');
history.commit(next);
assert.equal(history.document.darts[0].visible.length, 2);
assert.equal(history.document.darts[0].silhouette.length, 2);
assert.equal(history.document.darts[0].tip_status, 'unknown');
assert.deepEqual(history.document.darts[0].score, {number:20, multiplier:'triple'});
history.undo();
assert.deepEqual(history.document, initial);
history.redo();
assert.equal(history.document.darts[0].tip, null);
assert.deepEqual(history.document.darts[0].score, {number:20, multiplier:'triple'});
const scoreEdit = E.clone(history.document);
E.setScore(scoreEdit.darts[0], null);
assert.equal(scoreEdit.darts[0].score, null);
assert.throws(() => E.setScore(scoreEdit.darts[0],25,'triple'));
const selected = E.regions(history.document, 0)[0];
const changed = E.clone(history.document);
E.polygonPoints(changed, 0, selected)[0] = [11,11];
assert.deepEqual(history.document.darts[0].visible[0].points[0], [10,10]);
E.removePolygon(changed, 0, selected);
assert.equal(changed.darts[0].visible.length, 1);
assert.equal(changed.darts[0].silhouette.length, 2);
assert.equal(E.completeErrors(initial).length, 0);
assert.equal(E.nearestEdge(points, [5,1]).index, 0);
assert.deepEqual(E.intersections([[0,0],[10,10],[0,10],[10,0]]), [[5,5]]);
assert.deepEqual(E.intersections([[0,0],[10,0],[10,10],[0,10]]), []);
assert.ok(E.intersections([[0,0],[10,0],[10,10],[0,0],[-10,10],[-10,0]]).some(p => p[0] === 0 && p[1] === 0));
assert.ok(E.intersections([[0,0],[10,0],[5,0],[5,10]]).some(p => p[0] === 5 && p[1] === 0));
assert.ok(E.intersections([[0,0],[0,0],[10,0],[10,10]]).some(p => p[0] === 0 && p[1] === 0));
const approximate = new E.History(initial);
const placed = E.clone(approximate.document);
E.setTip(placed.darts[0], [25,12], 'approximate');
approximate.commit(placed);
assert.equal(approximate.document.darts[0].tip_status, 'approximate');
const moved = E.clone(approximate.document);
E.setTip(moved.darts[0], [26,13], moved.darts[0].tip_status);
approximate.commit(moved);
assert.equal(approximate.document.darts[0].tip_status, 'approximate');
approximate.undo();
assert.deepEqual(approximate.document.darts[0].tip, [25,12]);
approximate.undo();
assert.equal(approximate.document.darts[0].tip_status, 'exact');
approximate.redo();
assert.equal(approximate.document.darts[0].tip_status, 'approximate');
E.setTip(placed.darts[0], null);
assert.equal(placed.darts[0].tip_status, 'unknown');
""", str(editor), json.dumps(labels())], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
