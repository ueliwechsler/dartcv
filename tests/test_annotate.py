from copy import deepcopy
from hashlib import sha256
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from dartcv.annotate import AnnotationFolders, AnnotationStore, SaveConflict, make_server


@pytest.fixture
def store(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "one.jpg"), np.full((30, 40, 3), 100, np.uint8))
    cv2.imwrite(str(images / "two.jpg"), np.full((30, 40, 3), 120, np.uint8))
    return AnnotationStore(images, tmp_path / "labels")


def completed(store):
    result = store.load("one.jpg")
    annotation = result["annotation"]
    annotation["status"] = "completed"
    annotation["darts"] = annotation["darts"][:1]
    annotation["darts"][0]["visible"] = [{
        "part": "shaft_barrel", "points": [[1, 1], [10, 1], [10, 4], [1, 4]],
    }]
    return annotation


def test_atomic_resume_and_concurrent_save_conflicts(store):
    annotation = completed(store)
    annotation["darts"][0]["score"] = {"number": 20, "multiplier": "triple"}
    original = (store.image_dir / "one.jpg").read_bytes()
    saved = store.save("one.jpg", annotation, None)
    assert store.load("one.jpg") == {"annotation": json.loads(json.dumps(annotation)), "revision": saved["revision"]}
    assert store.listing()[0] == {"name": "one.jpg", "status": "completed"}
    assert store.load("one.jpg")["annotation"]["darts"][0]["score"] == {"number": 20, "multiplier": "triple"}
    with pytest.raises(SaveConflict, match="another tab"):
        store.save("one.jpg", annotation, None)
    edited = deepcopy(annotation)
    edited["status"] = "draft"
    store.save("one.jpg", edited, saved["revision"])
    assert (store.image_dir / "one.jpg").read_bytes() == original
    assert not list(store.output_dir.glob("*.tmp"))
    wrong = deepcopy(annotation)
    wrong["sha256"] = "0" * 64
    with pytest.raises(SaveConflict, match="source photograph"):
        store.save("one.jpg", wrong, None)
    with pytest.raises(ValueError, match="Unknown"):
        store.load("../one.jpg")


def test_changed_or_corrupt_inputs_never_load_as_clean_drafts(store):
    annotation = completed(store)
    store.save("one.jpg", annotation, None)
    cv2.imwrite(str(store.image_dir / "one.jpg"), np.zeros((30, 40, 3), np.uint8))
    with pytest.raises(SaveConflict, match="changed"):
        store.load("one.jpg")
    path = store.output_dir / "one.jpg.json"
    path.write_text("{bad JSON")
    with pytest.raises(ValueError):
        store.load("one.jpg")


def test_folder_selection_keeps_same_named_images_and_labels_separate(tmp_path):
    root = tmp_path / "images"
    for folder, value in (("multi_dart", 100), ("single_dart", 150)):
        directory = root / folder
        directory.mkdir(parents=True)
        cv2.imwrite(str(directory / "one.jpg"), np.full((30, 40, 3), value, np.uint8))
    folders = AnnotationFolders(root, tmp_path / "labels")
    assert folders.default == "multi_dart"
    multi = folders.select("multi_dart")
    single = folders.select("single_dart")
    annotation = completed(multi)
    multi.save("one.jpg", annotation, None)
    assert single.load("one.jpg")["revision"] is None
    assert single.load("one.jpg")["annotation"]["sha256"] != annotation["sha256"]
    single.save("one.jpg", completed(single), None)
    assert (tmp_path / "labels/one.jpg.json").is_file()
    assert (tmp_path / "labels/single_dart/one.jpg.json").is_file()
    with pytest.raises(ValueError, match="Unknown image folder"):
        folders.select("../elsewhere")
    server = make_server(folders, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/folders", timeout=5) as response:
            assert json.load(response) == {"folders": ["multi_dart", "single_dart"], "default": "multi_dart"}
        for folder in folders.stores:
            with urlopen(base + f"/api/annotation?folder={folder}&name=one.jpg", timeout=5) as response:
                assert json.load(response)["annotation"]["sha256"] == folders.select(folder).photo("one.jpg")["sha256"]
        with pytest.raises(HTTPError) as invalid:
            urlopen(base + "/api/images?folder=../elsewhere", timeout=5)
        assert invalid.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_http_load_save_and_origin_guards(store):
    server = make_server(store, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, data=None, headers=None):
        body = None if data is None else json.dumps(data).encode()
        with urlopen(Request(base + path, body, headers or {}), timeout=5) as response:
            return response.read(), response.headers

    try:
        body, headers = request("/")
        assert b"Draw polygon" in body and "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        for asset in ("/editor.js", "/app.js", "/style.css"):
            assert request(asset)[0]
        photo = json.loads(request("/api/annotation?name=one.jpg")[0])
        assert photo["annotation"]["width"] == 40
        assert photo["annotation"]["sha256"] == sha256((store.image_dir / "one.jpg").read_bytes()).hexdigest()
        image = cv2.imdecode(np.frombuffer(request("/api/pixels?name=one.jpg")[0], np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (30, 40, 3)
        token = json.loads(request("/api/session")[0])["token"]
        payload = {"annotation": completed(store), "revision": None}
        with pytest.raises(HTTPError) as missing:
            request("/api/annotation?name=one.jpg", payload, {"Content-Type": "application/json"})
        assert missing.value.code == 403
        saved = json.loads(request("/api/annotation?name=one.jpg", payload, {
            "Content-Type": "application/json", "X-Annotation-Token": token,
        })[0])
        assert saved["status"] == "completed"
        with pytest.raises(HTTPError) as conflict:
            request("/api/annotation?name=one.jpg", payload, {
                "Content-Type": "application/json", "X-Annotation-Token": token,
            })
        assert conflict.value.code == 409
        with pytest.raises(HTTPError) as cross_origin:
            request("/api/session", headers={"Origin": "https://untrusted.invalid"})
        assert cross_origin.value.code == 403
        with pytest.raises(HTTPError) as bad_host:
            request("/api/session", headers={"Host": "not-local.invalid"})
        assert bad_host.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
