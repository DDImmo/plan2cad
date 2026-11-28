"""Petite interface web (stdlib uniquement) : étapes de travail, plans par calques, vue 3D."""
from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

STATIC = Path(__file__).parent / "static"
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/octet-stream", ".dxf")
mimetypes.add_type("text/plain", ".obj")
mimetypes.add_type("application/octet-stream", ".stl")


def list_runs(root: Path):
    runs = []
    cands = []
    if (root / "run.json").exists():
        cands.append(("_", root))
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and (sub / "run.json").exists():
                cands.append((sub.name, sub))
    for rid, d in cands:
        try:
            data = json.loads((d / "run.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        runs.append(dict(id=rid, name=data.get("name", rid), status=data.get("status"),
                         created=data.get("created"), updated=data.get("updated"),
                         steps=len(data.get("steps", []))))
    runs.sort(key=lambda r: r.get("updated") or 0, reverse=True)
    return runs


def make_handler(root: Path):
    root = root.resolve()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="text/plain", cache=False):
            self.send_response(code)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline' data: blob:")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _file(self, p: Path, cache=False):
            if not p.is_file():
                return self._send(404, b"not found")
            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/json", "image/svg+xml"):
                ctype += "; charset=utf-8"
            self._send(200, p.read_bytes(), ctype, cache)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if path == "/api/runs":
                return self._send(200, json.dumps(list_runs(root)).encode(), "application/json")
            if path.startswith("/r/"):
                parts = path[3:].split("/", 1)
                if len(parts) != 2:
                    return self._send(404, b"not found")
                rid, rel = parts
                base = root if rid == "_" else (root / rid)
                target = (base / rel).resolve()
                if not target.is_relative_to(root):          # (et non startswith : /out2 commence par /out)
                    return self._send(403, b"forbidden")
                return self._file(target, cache=rel.startswith("steps/"))
            self._send(404, b"not found")

        do_HEAD = do_GET

    return H


def serve(root, port=8765, open_browser=False, block=True, host="127.0.0.1"):
    root = Path(root)
    srv = ThreadingHTTPServer((host, port), make_handler(root))
    url = f"http://{host}:{port}/"
    if block:
        print(f"Interface web : {url}   (Ctrl+C pour arrêter)")
        if open_browser:
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print()
        return srv
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    if open_browser:
        webbrowser.open(url)
    return srv
