"""Command line tool that writes the graph and plan of a compiled function as an HTML page."""

import argparse
import importlib
import importlib.util
import pathlib
import re
import sys

import lumen

from .common import DTYPES


def _examples():
    """Built-in example functions with their default input specs, by name."""
    import lumen.functional as F

    def attention(x, wq, wk, wv, w1, w2):
        """Single-head attention, then an MLP, each with a residual."""
        q, k, v = (
            F.matmul(x, wq, "float32", "float32"),
            F.matmul(x, wk, "float32", "float32"),
            F.matmul(x, wv, "float32", "float32"),
        )
        scores = F.matmul(q, k.t(), "float32", "float32") * (1.0 / q.shape[-1] ** 0.5)
        x = x + F.matmul(F.softmax(scores, -1), v, "float32", "float32")
        return x + F.matmul(F.relu(F.matmul(x, w1, "float32", "float32")), w2, "float32", "float32")

    def mlp(x, w1, w2):
        """Two-layer ReLU MLP followed by a softmax."""
        return F.softmax(F.matmul(F.relu(F.matmul(x, w1, "float32", "float32")), w2, "float32", "float32"), -1)

    seq, dim, hidden = 128, 256, 1024
    return {
        "attention": (
            attention,
            [f"f32[{seq},{dim}]"] + [f"f32[{dim},{dim}]"] * 3 + [f"f32[{dim},{hidden}]", f"f32[{hidden},{dim}]"],
        ),
        "mlp": (mlp, [f"f32[{seq},{dim}]", f"f32[{dim},{hidden}]", f"f32[{hidden},{dim}]"]),
    }


def _load(spec):
    """``path/to/file.py:fn`` or ``package.module:fn``."""
    target, _, name = spec.rpartition(":")
    if not target or not name:
        raise SystemExit(f"expected FILE.py:FUNCTION or MODULE:FUNCTION, got {spec!r}")
    if target.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location(pathlib.Path(target).stem, target)
        module = importlib.util.module_from_spec(mod_spec)
        sys.path.insert(0, str(pathlib.Path(target).resolve().parent))
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, name)


def _tensor(spec, device):
    """An input from ``dtype[d0,d1,...]``: ones, so every op is defined."""
    m = re.fullmatch(r"(\w+)\[([\d, ]*)\]", spec.strip())
    if not m:
        raise SystemExit(f"expected an input like f32[64,128], got {spec!r}")
    dtype = DTYPES.get(m[1], m[1])
    shape = [int(d) for d in m[2].split(",") if d.strip()]
    return lumen.ones(shape, dtype=dtype, device=device)


def main():
    """Parses the command line, compiles the function on ones-filled inputs and writes its HTML page."""
    parser = argparse.ArgumentParser(
        description="Write the graph and plan of a function compiled for a device as an HTML page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("function", nargs="?", help="FILE.py:FUNCTION or MODULE:FUNCTION")
    parser.add_argument("--example", choices=sorted(_examples()), help="a built-in function instead")
    parser.add_argument(
        "--input", action="append", default=[], help="an input as dtype[shape], e.g. f32[64,128]; repeat per argument"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--runs", type=int, default=5, help="profiled runs (GPU times are medians)")
    parser.add_argument("-o", "--out", default=None, help="the HTML page (default: <function>.html)")
    parser.add_argument("--json", default=None, help="also write the data as JSON here")
    parser.add_argument(
        "--fragment", action="store_true", help="no doctype: for a host that wraps the page (an artifact)"
    )
    args = parser.parse_args()
    if args.example:
        fn, inputs = _examples()[args.example]
        inputs = args.input or inputs
    elif args.function:
        fn, inputs = _load(args.function), args.input
    else:
        parser.error("give a FUNCTION or --example")
    tensors = [_tensor(s, args.device) for s in inputs]
    out = args.out or f"{fn.__name__}.html"
    data = lumen.compile(fn).dump_graph(out, *tensors, runs=args.runs, json_path=args.json, fragment=args.fragment)
    fused = data["views"]["fused"]["nodes"]
    print(
        f"{out}: {sum(n['kind'] in ('step', 'fusion') for n in fused)} plan steps "
        f"({sum(n['kind'] == 'fusion' for n in fused)} fusions), "
        f"{sum(n['kind'] == 'node' for n in data['views']['traced']['nodes'])} traced primitives"
    )


if __name__ == "__main__":
    main()
