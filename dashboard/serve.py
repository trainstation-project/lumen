import argparse
import functools
import http.server
import json
import pathlib
import threading

import bench

import lumen

ROOT = pathlib.Path(__file__).parent
running = threading.Lock()


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        prefix = "/api/benchmark/"
        if not self.path.startswith(prefix):
            return self.reply(404, {"error": f"no endpoint {self.path}"})
        name = self.path[len(prefix) :]
        if name not in bench.OPS:
            return self.reply(404, {"error": f"no benchmark for {name!r}; there are {', '.join(bench.OPS)}"})
        if not running.acquire(blocking=False):
            return self.reply(409, {"error": "a benchmark is already running; try again when it finishes"})
        try:
            data = bench.update(bench.load(), name)
            bench.save(data)
            self.reply(200, data)
        except Exception as e:
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            running.release()

    def reply(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description="Serve the lumen dashboard and run its benchmarks on this machine.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    lumen.config.static_allocator_bytes = 24 << 30
    handler = functools.partial(Handler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"lumen dashboard on http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
