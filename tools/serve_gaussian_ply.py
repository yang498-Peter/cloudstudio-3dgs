#!/usr/bin/env python3
"""Serve one Gaussian PLY to browser viewers with CORS and byte ranges."""

from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8793, type=int)
    args = parser.parse_args()

    source = args.file.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    route = f"/{quote(source.name)}"

    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Range")
            self.send_header(
                "Access-Control-Expose-Headers",
                "Accept-Ranges, Content-Length, Content-Range",
            )

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            if urlparse(self.path).path != route:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(source.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            request_path = urlparse(self.path).path
            if request_path in {"/", "/index.html"}:
                model_url = f"http://{args.host}:{args.port}{route}"
                editor_url = "https://superspl.at/editor?load=" + quote(
                    model_url, safe=""
                )
                payload = (
                    "<!doctype html><meta charset='utf-8'>"
                    "<title>CloudStudio Gaussian PLY</title>"
                    "<style>body{font:16px Segoe UI,sans-serif;max-width:900px;"
                    "margin:60px auto;padding:0 24px}a{font-size:20px}</style>"
                    f"<h1>{html.escape(source.name)}</h1>"
                    f"<p>{source.stat().st_size / 1024**2:.1f} MiB</p>"
                    f"<a href='{html.escape(editor_url)}'>Open in SuperSplat</a>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self._cors()
                self.end_headers()
                self.wfile.write(payload)
                return
            if request_path != route:
                self.send_error(404)
                return

            size = source.stat().st_size
            start = 0
            end = size - 1
            range_header = self.headers.get("Range")
            if range_header:
                if not range_header.startswith("bytes=") or "," in range_header:
                    self.send_error(416)
                    return
                bounds = range_header[6:].split("-", 1)
                try:
                    if bounds[0]:
                        start = int(bounds[0])
                    if bounds[1]:
                        end = int(bounds[1])
                except ValueError:
                    self.send_error(416)
                    return
                if start < 0 or end < start or start >= size:
                    self.send_error(416)
                    return
                end = min(end, size - 1)

            length = end - start + 1
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self._cors()
            self.end_headers()
            with source.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def log_message(self, format: str, *values: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Gaussian PLY ready: http://{args.host}:{args.port}{route}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
