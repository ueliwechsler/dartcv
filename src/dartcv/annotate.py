"""Local-only dart annotation editor: ``python -m dartcv.annotate``."""

from __future__ import annotations

import argparse
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .dart_annotations import annotation_path, parse_annotation

ASSET_DIR = Path(__file__).with_name("annotation_ui")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class SaveConflict(ValueError):
    pass


class AnnotationStore:
    def __init__(self, image_dir: Path, output_dir: Path):
        self.image_dir = image_dir.resolve(strict=True)
        self.output_dir = output_dir.resolve()
        if not self.image_dir.is_dir():
            raise ValueError("Image input must be a directory.")
        self.images = {
            path.name: path for path in sorted(self.image_dir.iterdir())
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in SUFFIXES
        }
        if not self.images:
            raise ValueError(f"No supported photographs in {self.image_dir}. Restore the local DVC images first.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self._cached = None

    def photo(self, name):
        if name not in self.images:
            raise ValueError("Unknown photograph.")
        path = self.images[name]
        stat = path.stat()
        cache_key = (name, stat.st_mtime_ns, stat.st_size)
        with self.lock:
            if self._cached is not None and self._cached[0] == cache_key:
                return self._cached[1]
            raw = path.read_bytes()
            # Default OpenCV decoding matches the pipeline, including EXIF orientation.
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {name}.")
            height, width = image.shape[:2]
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError(f"Cannot display {name}.")
            result = {
                "image": name, "width": width, "height": height,
                "sha256": sha256(raw).hexdigest(), "pixels": encoded.tobytes(),
            }
            self._cached = (cache_key, result)
            return result

    def _read(self, name):
        if name not in self.images:
            raise ValueError("Unknown photograph.")
        path = annotation_path(self.output_dir, name)
        if path.is_symlink():
            raise ValueError("Annotation files must not be symlinks.")
        if not path.exists():
            return None, None
        raw = path.read_bytes()
        annotation = parse_annotation(json.loads(raw))
        if annotation.image != name:
            raise ValueError(f"Annotation filename/image mismatch for {name}.")
        return annotation, sha256(raw).hexdigest()

    def listing(self):
        with self.lock:
            result = []
            for name in self.images:
                annotation, _ = self._read(name)
                result.append({"name": name, "status": annotation.status if annotation else "new"})
            return result

    def load(self, name):
        photo = self.photo(name)
        with self.lock:
            annotation, revision = self._read(name)
        if annotation is None:
            data = {
                "schema_version": 1,
                **{key: photo[key] for key in ("image", "width", "height", "sha256")},
                "status": "draft",
                "darts": [
                    {
                        "id": f"dart-{i}", "visible": [], "silhouette": [],
                        "tip_status": "unknown", "tip": None, "contact": "unknown", "score": None,
                    }
                    for i in range(1, 4)
                ],
                "ignore": [],
            }
        else:
            data = annotation.to_dict()
            if any(data[key] != photo[key] for key in ("width", "height", "sha256")):
                raise SaveConflict(f"{name} has changed since labeling. Preserve the labels and resolve the image mismatch before continuing.")
        return {"annotation": data, "revision": revision}

    def save(self, name, data, revision):
        annotation = parse_annotation(data)
        photo = self.photo(name)
        if any(getattr(annotation, key) != photo[key] for key in ("image", "width", "height", "sha256")):
            raise SaveConflict("Annotation does not match the current source photograph.")
        if revision is not None and not isinstance(revision, str):
            raise ValueError("Revision must be a string or null.")
        with self.lock:
            _, current_revision = self._read(name)
            if current_revision != revision:
                raise SaveConflict("Labels changed in another tab. Save a copy of your work with Download JSON, then reload.")
            content = (json.dumps(annotation.to_dict(), indent=2) + "\n").encode()
            temporary = None
            try:
                with NamedTemporaryFile(dir=self.output_dir, prefix=".annotation-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, annotation_path(self.output_dir, name))
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return {"revision": sha256(content).hexdigest(), "status": annotation.status}


def make_server(store: AnnotationStore, port: int = 8765) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, content_type="application/json"):
            if content_type == "application/json":
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _allowed(self):
            address = self.server.server_address
            hosts = {f"127.0.0.1:{address[1]}", f"localhost:{address[1]}"}
            if self.headers.get("Host") not in hosts or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._send(HTTPStatus.FORBIDDEN, {"error": "Only same-origin localhost requests are allowed."})
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://{host}" for host in hosts}:
                self._send(HTTPStatus.FORBIDDEN, {"error": "Origin is not this local editor."})
                return False
            return True

        def _dispatch(self, post=False):
            if not self._allowed():
                return
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            name = params.get("name", [""])[0]
            try:
                if post:
                    if parsed.path != "/api/annotation":
                        self._send(404, {"error": "Unknown endpoint."})
                        return
                    if not secrets.compare_digest(self.headers.get("X-Annotation-Token", ""), token):
                        self._send(403, {"error": "Missing or invalid session token. Reload the editor."})
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4 * 1024 * 1024:
                        self._send(413, {"error": "Annotation must be between 1 byte and 4 MiB."})
                        return
                    if self.headers.get_content_type() != "application/json":
                        raise ValueError("Save requests require application/json.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or set(payload) != {"annotation", "revision"}:
                        raise ValueError("Save requires annotation and revision.")
                    self._send(200, store.save(name, payload["annotation"], payload["revision"]))
                elif parsed.path == "/api/session":
                    self._send(200, {"token": token})
                elif parsed.path == "/api/images":
                    self._send(200, store.listing())
                elif parsed.path == "/api/annotation":
                    self._send(200, store.load(name))
                elif parsed.path == "/api/pixels":
                    self._send(200, store.photo(name)["pixels"], "image/jpeg")
                elif parsed.path in ("/", "/app.js", "/editor.js", "/style.css"):
                    filename, content_type = {
                        "/": ("index.html", "text/html; charset=utf-8"),
                        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                        "/editor.js": ("editor.js", "text/javascript; charset=utf-8"),
                        "/style.css": ("style.css", "text/css; charset=utf-8"),
                    }[parsed.path]
                    self._send(200, (ASSET_DIR / filename).read_bytes(), content_type)
                else:
                    self._send(404, {"error": "Unknown endpoint."})
            except SaveConflict as exc:
                self._send(409, {"error": str(exc)})
            except (ValueError, UnicodeDecodeError) as exc:
                self._send(400, {"error": str(exc)})
            except OSError as exc:
                self.log_error("Annotation I/O failed: %s", exc)
                self._send(500, {"error": f"Local file operation failed: {exc}"})

        def do_GET(self):
            self._dispatch()

        def do_POST(self):
            self._dispatch(post=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/images/multi_dart"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/darts"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        store = AnnotationStore(args.images, args.annotations)
        server = make_server(store, args.port)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot start annotation editor: {exc}\n")
    print(f"Open http://127.0.0.1:{server.server_port} (local only; Ctrl-C to stop).", flush=True)
    print(f"Photographs: {store.image_dir}\nLabels: {store.output_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
