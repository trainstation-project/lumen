"""Local HTTP server for the dashboard: serves its static files and streams benchmark runs."""

import argparse
import functools
import http.server
import json
import pathlib
import threading
import urllib.parse

import bench

import lumen

ROOT = pathlib.Path(__file__).parent
running = threading.Lock()
DEVICES = ("mps",)


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static file handler plus ``POST /api/benchmark/<op>?device=...`` that runs one benchmark."""

    def do_POST(self):
        """Runs the named op's benchmark, streaming each result row and then the saved data as NDJSON."""
        prefix = "/api/benchmark/"
        url = urllib.parse.urlsplit(self.path)
        if not url.path.startswith(prefix):
            return self.reply(404, {"error": f"no endpoint {url.path}"})
        name = url.path[len(prefix) :]
        device = urllib.parse.parse_qs(url.query).get("device", ["mps"])[0]
        if device not in DEVICES:
            return self.reply(400, {"error": f"no benchmark on {device!r} yet; there is one on {', '.join(DEVICES)}"})
        if name not in bench.OPS:
            return self.reply(404, {"error": f"no benchmark for {name!r}; there are {', '.join(bench.OPS)}"})
        if bench.torch is None:
            return self.reply(503, {"error": bench.NO_TORCH})
        if not running.acquire(blocking=False):
            return self.reply(409, {"error": "a benchmark is already running; try again when it finishes"})
            # Headers go out before running so rows can stream; later failures become an error line.
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                data = bench.update(bench.load(), name, lambda row: self.line({"row": row}))
                bench.save(data, [name])
                self.line({"data": data})
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                self.line({"error": f"{type(e).__name__}: {e}"})
        finally:
            running.release()

    def line(self, body):
        """Writes one NDJSON line and flushes it to the client."""
        self.wfile.write(json.dumps(body, ensure_ascii=False).encode() + b"\n")
        self.wfile.flush()

    def reply(self, status, body):
        """Sends a complete JSON response with ``status``."""
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    """Parses ``--port`` and serves the dashboard on localhost until interrupted."""
    parser = argparse.ArgumentParser(description="Serve the lumen dashboard and run its benchmarks on this machine.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    lumen.config.static_allocator_bytes = 4 << 30
    handler = functools.partial(Handler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"lumen dashboard on http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
