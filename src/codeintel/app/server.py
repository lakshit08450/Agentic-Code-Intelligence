"""Tiny local HTTP server (standard library only) for the integration API.

    .venv\\Scripts\\python -m codeintel.app.server --port 8765            # real models (warm-up in background)
    .venv\\Scripts\\python -m codeintel.app.server --port 8765 --stub     # canned results, instant

GET  /health  -> readiness, sandbox status, versions (200 always)
POST /search  -> JSON body {"query", "version"?, "k"?, "verify"?, "pipeline"?, "stub"?, "require_stage2"?}
Requests are served by threads but searches are serialized (one at a time); each request waits at most
--timeout seconds (504 TIMEOUT). Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from codeintel.app.schema import error_response
from codeintel.app.service import SearchService, http_status

MAX_BODY = 1 << 20


def make_handler(svc: SearchService, timeout_s: float):
    pool = ThreadPoolExecutor(max_workers=4)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802 - CORS preflight for a browser UI
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(200, svc.health())
            else:
                self._send(404, {"error": {"code": "NOT_FOUND", "message": self.path}})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/search":
                self._send(404, {"error": {"code": "NOT_FOUND", "message": self.path}})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > MAX_BODY:
                    raise ValueError("body missing or too large")
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(req, dict):
                    raise ValueError("body must be a JSON object")
            except (ValueError, json.JSONDecodeError) as e:
                r = error_response("BAD_REQUEST", f"invalid JSON body: {e}")
                self._send(400, r)
                return
            kwargs = {k: req[k] for k in ("version", "k", "verify", "pipeline", "stub", "require_stage2") if k in req}
            fut = pool.submit(svc.search, req.get("query", ""), **kwargs)
            try:
                resp = fut.result(timeout=timeout_s)
            except FutTimeout:
                resp = error_response("TIMEOUT", f"search exceeded {timeout_s:.0f} s", query=req.get("query", ""))
            self._send(http_status(resp), resp)

        def log_message(self, fmt, *args):  # quieter console
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stub", action="store_true", help="canned results, no models")
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout in seconds")
    a = ap.parse_args()
    svc = SearchService(stub=a.stub, **({"demo_dir": a.demo_dir} if a.demo_dir else {}))
    threading.Thread(target=lambda: _load(svc), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(svc, a.timeout))
    print(f"codeintel server on http://127.0.0.1:{a.port} (stub={a.stub}); loading models in background...", flush=True)
    server.serve_forever()


def _load(svc: SearchService) -> None:
    try:
        svc.load()
        print(f"ready (load {svc.load_time_s}s, sandbox={svc.sandbox})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"load failed: {e}", flush=True)


if __name__ == "__main__":
    main()
