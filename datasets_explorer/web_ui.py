"""Lightweight live-status web UI for dataset_search runs.

Pure stdlib — runs an in-process ThreadingHTTPServer that:
  - serves a single-page dashboard on  /
  - streams events via Server-Sent Events on  /events

The agent (and CLI) push events through a global Broker; every browser tab
connected to /events receives them in real time.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import queue
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

logger = logging.getLogger("datasets_explorer.web_ui")


class Broker:
    """Pub/sub fan-out for SSE clients. Each subscriber gets its own queue."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: list[dict] = []  # replayed to new subscribers

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2048)
        with self._lock:
            for ev in self._history[-500:]:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, ev: dict) -> None:
        with self._lock:
            self._history.append(ev)
            if len(self._history) > 5000:
                del self._history[:1000]
            for q in self._subs:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass


_broker = Broker()
_server_thread: Optional[threading.Thread] = None
_server: Optional[ThreadingHTTPServer] = None
_server_port: Optional[int] = None

# Model-switch coordination. The web UI writes here; the agent loop polls
# `consume_model_switch()` at the top of each iteration.
_pending_model: Optional[str] = None
_pending_lock = threading.Lock()


def publish(ev: dict) -> None:
    """Public API — agents and CLI call this to emit an event."""
    _broker.publish(ev)


def request_model_switch(name: str) -> None:
    """Called by the HTTP handler when the user picks a new model."""
    global _pending_model
    with _pending_lock:
        _pending_model = (name or "").strip() or None


def consume_model_switch() -> Optional[str]:
    """Atomically read + clear the pending model switch. Returns the new
    model name if one was requested, else None."""
    global _pending_model
    with _pending_lock:
        m, _pending_model = _pending_model, None
        return m


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        # Suppress access-log spam to stderr; we have our own logger.
        def log_message(self, format, *args):  # noqa: A002
            return

        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._serve_html()
            elif self.path == "/events":
                self._serve_sse()
            elif self.path == "/models":
                self._serve_models()
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if self.path == "/switch-model":
                self._serve_switch_model()
            else:
                self.send_error(404)

        def _serve_models(self):
            try:
                import ollama
                from .config import OLLAMA_HOST
                client = ollama.Client(host=OLLAMA_HOST)
                resp = client.list()
                names = []
                for m in (resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", [])) or []:
                    n = m.get("name") if isinstance(m, dict) else getattr(m, "model", None) or getattr(m, "name", None)
                    if n:
                        names.append(n)
            except Exception as e:
                names = []
                logger.warning(f"/models error: {e}")
            body = json.dumps({"models": sorted(set(names))}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_switch_model(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                name = (payload.get("model") or "").strip()
            except Exception:
                name = ""
            if not name:
                self.send_error(400, "missing 'model' in body")
                return
            request_model_switch(name)
            _broker.publish({"type": "model_switch_requested", "model": name})
            body = json.dumps({"ok": True, "model": name}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_html(self):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = _broker.subscribe()
            try:
                last_ping = time.time()
                while True:
                    try:
                        ev = q.get(timeout=1.0)
                    except queue.Empty:
                        ev = None
                    if ev is not None:
                        try:
                            self.wfile.write(b"data: ")
                            self.wfile.write(json.dumps(ev, default=str).encode("utf-8"))
                            self.wfile.write(b"\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    now = time.time()
                    if now - last_ping > 15:
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        last_ping = now
            finally:
                _broker.unsubscribe(q)

    return Handler


def start(port: int = 7860, host: str = "127.0.0.1") -> Optional[int]:
    """Start the web UI server in a background thread. Idempotent."""
    global _server, _server_thread, _server_port
    if _server_thread is not None and _server_thread.is_alive():
        return _server_port
    # Try a few ports if the requested one is busy.
    for candidate in [port, port + 1, port + 2, 0]:
        try:
            srv = ThreadingHTTPServer((host, candidate), _make_handler())
            break
        except OSError:
            continue
    else:
        logger.warning("Could not bind a port for the web UI")
        return None
    _server = srv
    _server_port = srv.server_address[1]

    def _run():
        try:
            srv.serve_forever(poll_interval=0.5)
        except Exception as e:
            logger.warning(f"web_ui server stopped: {e}")

    _server_thread = threading.Thread(target=_run, daemon=True, name="web_ui")
    _server_thread.start()
    return _server_port


def stop() -> None:
    global _server
    if _server is not None:
        try:
            _server.shutdown()
        except Exception:
            pass
        _server = None


# --------------------------------------------------------------------------
# Static dashboard. Loaded from external template file at import time.
# --------------------------------------------------------------------------
_TEMPLATE_DIR = Path(__file__).resolve().parent
_INDEX_PATH = _TEMPLATE_DIR / "web_ui_template.html"
if _INDEX_PATH.exists():
    INDEX_HTML = _INDEX_PATH.read_text(encoding="utf-8")
else:
    INDEX_HTML = ""

