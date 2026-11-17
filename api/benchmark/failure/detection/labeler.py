"""A small local web page for labeling face boxes on the sampled frames.

Stdlib only (``http.server``), one self-contained ``labeler.html``. Routes:

- ``GET /``            the labeling page
- ``GET /api/state``   frames from proposals.json merged with labels.json
- ``GET /img/<file>``  a frame; only plain names listed in proposals.json
- ``POST /api/label``  ``{file, boxes, reviewed}``; saved to labels.json

Boxes are in source-image pixels. labels.json is written atomically (temp
file + replace) under a lock, so a crash or a second request can't leave it
half-written.
"""

import json
import math
import os
import tempfile
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import FORMAT_VERSION, FRAMES_DIR, LABELS_FILE, load_proposals

HTML_PATH = Path(__file__).with_name("labeler.html")
MAX_BODY = 1 << 20

_LABELS_LOCK = threading.Lock()


# ---------------------------------------------------------------- labels I/O

def load_labels(out_dir: Path) -> dict:
    path = Path(out_dir) / LABELS_FILE
    if not path.exists():
        return {"version": FORMAT_VERSION, "frames": {}}
    data = json.loads(path.read_text())
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported labels version {data.get('version')}")
    data.setdefault("frames", {})
    return data


def save_labels(out_dir: Path, data: dict) -> None:
    """Write labels.json atomically: a temp file in the same dir, then replace."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".labels.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_dir / LABELS_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def update_label(out_dir: Path, file: str, boxes: list[dict], reviewed: bool) -> dict:
    """Read-modify-write one frame's entry under the lock; return the entry."""
    entry = {
        "reviewed": bool(reviewed),
        "boxes": boxes,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with _LABELS_LOCK:
        data = load_labels(out_dir)
        data["frames"][file] = entry
        save_labels(out_dir, data)
    return entry


# ---------------------------------------------------------------- validation

class BadRequest(ValueError):
    pass


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BadRequest(f"{what} must be a finite number")
    return float(value)


def validate_label(payload, frames_by_file: dict) -> tuple[str, list[dict], bool]:
    """Check a POST /api/label body; boxes come back clamped to the image, in ints."""
    if not isinstance(payload, dict):
        raise BadRequest("body must be a JSON object")
    file = payload.get("file")
    if not isinstance(file, str) or file not in frames_by_file:
        raise BadRequest("unknown file")
    reviewed = payload.get("reviewed")
    if not isinstance(reviewed, bool):
        raise BadRequest("reviewed must be true or false")
    boxes = payload.get("boxes")
    if not isinstance(boxes, list):
        raise BadRequest("boxes must be a list")
    frame = frames_by_file[file]
    width, height = frame["width"], frame["height"]
    clean = []
    for i, b in enumerate(boxes):
        if not isinstance(b, dict):
            raise BadRequest(f"box {i} must be an object")
        x, y, w, h = (_number(b.get(k), f"box {i} {k}") for k in ("x", "y", "w", "h"))
        if w <= 0 or h <= 0:
            raise BadRequest(f"box {i} must have positive size")
        x1, y1 = max(0, min(round(x), width)), max(0, min(round(y), height))
        x2, y2 = max(0, min(round(x + w), width)), max(0, min(round(y + h), height))
        if x2 <= x1 or y2 <= y1:
            raise BadRequest(f"box {i} lies outside the image")
        clean.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return file, clean, reviewed


# ---------------------------------------------------------------- server

def build_state(out_dir: Path, frames: list[dict]) -> dict:
    labels = load_labels(out_dir)["frames"]
    merged = []
    for frame in frames:
        label = labels.get(frame["file"]) or {}
        merged.append({
            **frame,
            "reviewed": bool(label.get("reviewed")),
            "boxes": label.get("boxes") or [],
            "updated_at": label.get("updated_at"),
        })
    reviewed = sum(1 for f in merged if f["reviewed"])
    return {"frames": merged, "reviewed": reviewed, "total": len(merged)}


def _handler_class(out_dir: Path, frames: list[dict]):
    frames_by_file = {f["file"]: f for f in frames}
    frames_dir = (out_dir / FRAMES_DIR).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "DetectionLabeler/1"

        def log_message(self, fmt, *args):  # quiet; the page polls nothing
            pass

        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, data):
            self._send(status, json.dumps(data).encode(), "application/json")

        def _error(self, status, message):
            self._json(status, {"error": message})

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                with _LABELS_LOCK:
                    state = build_state(out_dir, frames)
                self._json(HTTPStatus.OK, state)
            elif path.startswith("/img/"):
                self._image(path[len("/img/"):])
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def _image(self, raw_name: str):
            name = unquote(raw_name)
            # Only plain names that proposals.json lists; nothing with a path.
            if (not name or "/" in name or "\\" in name or "\0" in name
                    or name in (".", "..") or name not in frames_by_file):
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            path = (frames_dir / name).resolve()
            if path.parent != frames_dir or not path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._send(HTTPStatus.OK, path.read_bytes(), "image/jpeg")

        def do_POST(self):
            if urlsplit(self.path).path != "/api/label":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY:
                self._error(HTTPStatus.BAD_REQUEST, "missing or oversized body")
                return
            try:
                payload = json.loads(self.rfile.read(length))
                file, boxes, reviewed = validate_label(payload, frames_by_file)
            except (ValueError, UnicodeDecodeError) as e:  # BadRequest and JSON errors
                self._error(HTTPStatus.BAD_REQUEST, str(e))
                return
            entry = update_label(out_dir, file, boxes, reviewed)
            self._json(HTTPStatus.OK, {"file": file, **entry})

    return Handler


def make_server(out_dir: Path, host: str = "0.0.0.0", port: int = 8139) -> ThreadingHTTPServer:
    """Build (but don't start) the labeler server; port 0 picks a free port."""
    out_dir = Path(out_dir)
    frames = load_proposals(out_dir)["frames"]
    load_labels(out_dir)  # fail now, not on the first request, if it's unreadable
    server = ThreadingHTTPServer((host, port), _handler_class(out_dir, frames))
    server.daemon_threads = True
    return server


def serve(out_dir: Path, host: str = "0.0.0.0", port: int = 8139) -> None:
    """Serve the labeler until interrupted."""
    server = make_server(out_dir, host, port)
    bound_host, bound_port = server.server_address[:2]
    shown = "localhost" if bound_host in ("0.0.0.0", "::") else bound_host
    print(f"Labeling {out_dir}", flush=True)
    print(f"Open http://{shown}:{bound_port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
