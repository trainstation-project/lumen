"""A compiled function's graph as one self-contained HTML page:
``lumen.compile(fn).dump_graph(path)``. The page shows the fused plan and
the traced graph as node graphs, and for every node its types, buffers, the
kernels it launched (with GPU time) and their Metal source, generated
(fusions) or hand-written (``lumen/ops/<op>/mps.metal``).

From the command line, for a function and inputs of the given types:

    python -m lumen.graph.viz --example attention -o attention.html
    python -m lumen.graph.viz my_model.py:block --input 'f32[64,128]' --input 'f32[128,128]'
"""

import argparse
import datetime
import html
import importlib
import importlib.util
import json
import pathlib
import re
import statistics
import sys

import lumen
from lumen._C import Plan
from lumen.profiler import ProfilerActivity, profile

OPS = pathlib.Path(lumen.__file__).parent / "ops"

DTYPES = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "f64": "float64",
    "bool": "bool",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}
SHORT = {v: k for k, v in DTYPES.items()}


def type_text(dtype, shape):
    return f"{SHORT.get(dtype, dtype)}[{','.join(map(str, shape))}]"


# ---------------------------------------------------------------------
# where a primitive kernel's source is: (file under lumen/ops, anchor)
# ---------------------------------------------------------------------

_SOURCES = [
    (r"^matmul_small_", "dot_general/mps.metal", "inline void matmul_sg_impl"),
    (r"^matmul_(f16|bf16|f32)(_f32_(f16|bf16|f32))?$", "dot_general/mps.metal", "inline void matmul_sg_impl"),
    (r"^matmul_(u64|i64)$", "dot_general/mps.metal", "inline void matmul_wide_impl"),
    (r"^matmul_", "dot_general/mps.metal", "inline void matmul_impl"),
    (r"^reduce_\w+_rows_", "reduce/mps.metal", "inline void reduce_rows"),
    (r"^reduce_\w+_cols_", "reduce/mps.metal", "inline void reduce_cols"),
    (r"^reduce_\w+_grouped_", "reduce/mps.metal", "inline void reduce_grouped"),
    (r"^reduce_", "reduce/mps.metal", "inline void reduce("),
    (r"^gather_", "layout/mps.metal", "#define GATHER"),
    (r"^transpose_", "layout/mps.metal", "#define TRANSPOSE"),
    (r"^fill_(strided_)?u\d+$", "fill/mps.metal", "inline void fill"),
    (r"^fill_\d$", "factory/mps.metal", "#define FULL"),
    (r"^iota_", "factory/mps.metal", "#define IOTA"),
    (r"^convert_", "elementwise/mps.metal", "#define CONVERT"),
    (r"^select_", "elementwise/mps.metal", "#define SELECT"),
    (r"^neg_", "elementwise/mps.metal", "#define NEG"),
    (r"^(exp|log|sqrt|tanh|logistic)_", "elementwise/mps.metal", "#define UNARY"),
    (r"^(max|eq|lt)_", "elementwise/mps.metal", "#define ORDERED"),
    (r"^(add|sub|mul|div)_", "elementwise/mps.metal", "#define BINARY"),
]


def source_of(kernel):
    """``(file, anchor)`` of a primitive kernel's source, or None."""
    for pattern, path, anchor in _SOURCES:
        if re.match(pattern, kernel):
            return path, anchor
    return None


# ---------------------------------------------------------------------
# profiling: each plan step's kernels and GPU time
# ---------------------------------------------------------------------


def profile_steps(plan, run, runs, device):
    """For each step of ``plan``, its kernels in launch order as ``[name,
    median GPU us]`` over ``runs`` calls of ``run`` (running it) on
    ``device``."""
    sync = lumen.mps.synchronize if device == "mps" else (lambda: None)
    run()
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
        for _ in range(runs):
            run()
            sync()
    events = prof.events()
    by_parent = {}
    for e in events:
        if e["kind"] == "gpu":
            by_parent.setdefault(e["parent"], []).append(e)
    nsteps = len(plan.steps())
    times = [[] for _ in range(nsteps)]  # per step, per run: [(kernel, us), ...]
    for run in (e for e in events if e["name"] == "lumen::plan"):
        steps = sorted(
            (
                e
                for e in events
                if e["parent"] == run["id"] and e["kind"] == "op" and not e["name"].startswith("lumen::")
            ),
            key=lambda e: e["start_us"],
        )
        if len(steps) != nsteps:
            continue  # cannot line the ranges up with the steps
        for i, step in enumerate(steps):
            gpu = sorted(by_parent.get(step["id"], []), key=lambda e: e["start_us"])
            times[i].append([(e.get("kernel") or e["name"], e["duration_us"]) for e in gpu])
    result = []
    for runs_of_step in times:
        if not runs_of_step:
            result.append([])
            continue
        kernels = [name for name, _ in runs_of_step[0]]
        result.append(
            [[name, statistics.median(r[k][1] for r in runs_of_step if len(r) > k)] for k, name in enumerate(kernels)]
        )
    return result


# ---------------------------------------------------------------------
# views: nodes and edges
# ---------------------------------------------------------------------


def _kernel_entries(kernels):
    out = []
    for name, us in kernels:
        src = source_of(name)
        out.append({"name": name, "us": us, "file": src[0] if src else None, "anchor": src[1] if src else None})
    return out


def plan_view(plan, kernels):
    """The plan as a graph: its inputs, steps and outputs as nodes, an edge
    from each buffer's latest writer to each step reading it."""
    nodes, edges, writer = [], [], {}

    def node_for(buffer, dtype, shape):
        if buffer not in writer:
            nodes.append(
                {
                    "id": len(nodes),
                    "kind": "input" if buffer.startswith(("in", "s")) else "buffer",
                    "label": buffer,
                    "type": type_text(dtype, shape),
                    "buffer": buffer,
                }
            )
            writer[buffer] = nodes[-1]["id"]
        return writer[buffer]

    for i, step in enumerate(plan.steps()):
        sources = [(node_for(b, d, s), b) for b, d, s in step["inputs"]]
        out_buffer, dtype, shape = step["output"]
        fusion = step["fusion"]
        node = {
            "id": len(nodes),
            "kind": "fusion" if fusion else "step",
            "step": i,
            "label": step["label"],
            "text": step["text"],
            "type": type_text(dtype, shape),
            "buffer": out_buffer,
            "inputs": [[b, type_text(d, s)] for b, d, s in step["inputs"]],
            "kernels": _kernel_entries(kernels[i] if i < len(kernels) else []),
            "fusion": fusion,
        }
        nodes.append(node)
        for src, b in sources:
            edges.append([src, node["id"], b])
        writer[out_buffer] = node["id"]
        for b, _, _ in step["extra_outputs"]:  # a multi-output fusion's other outputs
            writer[b] = node["id"]
    for buffer in sorted((b for b in writer if b.startswith("out")), key=lambda b: int(b[3:])):
        src = writer[buffer]
        if nodes[src]["kind"] in ("input", "buffer"):
            continue
        nodes.append(
            {"id": len(nodes), "kind": "output", "label": buffer, "type": nodes[src]["type"], "buffer": buffer}
        )
        edges.append([src, nodes[-1]["id"], ""])
    return {"nodes": nodes, "edges": edges}


def graph_view(graph, unfused, kernels):
    """The traced graph, a node per primitive, each with the kernels its
    step in the unfused plan launched (steps are the live nodes in order)."""
    nodes, edges, of_var = [], [], {}
    for k, v in enumerate(graph.inputs()):
        dtype, shape = graph.type_of(v)
        nodes.append({"id": len(nodes), "kind": "input", "label": f"in{k}", "type": type_text(dtype, shape), "var": v})
        of_var[v] = nodes[-1]["id"]
    steps = unfused.steps()
    s = 0
    for node in graph.nodes():
        dtype, shape = graph.type_of(node["output"])
        entry = {
            "id": len(nodes),
            "kind": "node",
            "label": node["primitive"],
            "text": node["text"],
            "type": type_text(dtype, shape),
            "var": node["output"],
            "inputs": [[f"%{v}", type_text(*graph.type_of(v))] for v in node["inputs"]],
            "kernels": [],
            "fusion": None,
        }
        # The unfused plan's steps are the live nodes in graph order (it may
        # drop dead ones and alias reshapes), plus steps the device's
        # compiler added (a transpose putting a dot_general's operand in
        # matmul form): those go with the node after them.
        added = []
        for t in range(s, len(steps)):
            if steps[t]["primitive"] == node["primitive"] and list(steps[t]["output"][1:]) == [dtype, list(shape)]:
                added += [k for u in range(s, t + 1) for k in (kernels[u] if u < len(kernels) else [])]
                entry["kernels"] = _kernel_entries(added)
                entry["buffer"] = steps[t]["output"][0]
                s = t + 1
                break
        else:
            entry["note"] = (
                "no kernel of its own in the unfused plan (an aliasing reshape, a repeat of an earlier node, or dead)"
            )
        nodes.append(entry)
        of_var[node["output"]] = entry["id"]
        for v in node["inputs"]:
            edges.append([of_var[v], entry["id"], f"%{v}"])
    for k, v in enumerate(graph.outputs()):
        dtype, shape = graph.type_of(v)
        nodes.append({"id": len(nodes), "kind": "output", "label": f"out{k}", "type": type_text(dtype, shape)})
        edges.append([of_var[v], nodes[-1]["id"], f"%{v}"])
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------


def collect(graph, plan, inputs, title, runs=5, device=None, run=None):
    """Everything the page shows, as a JSON-able dict: ``graph``, its
    ``plan`` (fused, for ``device``, default the inputs') and its unfused
    plan, both profiled on ``inputs`` on that device (the graph's, then any
    the plan adds), ``plan`` by ``run`` if given (else ``plan.run``)."""
    device = device or (str(inputs[0].device) if inputs else "cpu")
    # The unfused plan for the same device: its steps are the graph's
    # primitives (after the device's rewrites), each its own kernel.
    unfused = Plan(graph) if device in ("cpu", "meta") else Plan(graph, device, fuse=False)
    timed = device not in ("cpu", "meta")
    graph_inputs = inputs[: len(graph.inputs())]
    run = run or (lambda: plan.run(inputs, device))
    fused_kernels = profile_steps(plan, run, runs, device) if timed else []
    unfused_kernels = profile_steps(unfused, lambda: unfused.run(graph_inputs, device), runs, device) if timed else []
    files = sorted(
        {
            k["file"]
            for view in (fused_kernels, unfused_kernels)
            for step in view
            for k in _kernel_entries(step)
            if k["file"]
        }
    )
    return {
        "title": title,
        "device": device,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "inputs": [type_text(*graph.type_of(v)) for v in graph.inputs()],
        "graph_text": str(graph),
        "fused_text": str(plan),
        "unfused_text": str(unfused),
        "workspace": {"fused": plan.workspace_bytes, "unfused": unfused.workspace_bytes},
        "views": {"fused": plan_view(plan, fused_kernels), "traced": graph_view(graph, unfused, unfused_kernels)},
        "sources": {f: (OPS / f).read_text() for f in files},
        "prelude": (OPS / "mps.metal").read_text(),
    }


def write(data, path, json_path=None, fragment=False):
    """Write the page for ``data`` to ``path`` (and ``data`` as JSON to
    ``json_path``). ``fragment`` leaves out the doctype, for a host that
    wraps the page in its own document (a published artifact)."""
    page = TEMPLATE.replace("__TITLE__", html.escape(data["title"])).replace(
        "__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    )
    if not fragment:
        page = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n' + page
    pathlib.Path(path).write_text(page)
    if json_path:
        pathlib.Path(json_path).write_text(json.dumps(data, indent=1))


# ---------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------


def _examples():
    def attention(x, wq, wk, wv, w1, w2):
        """Single-head attention, then an MLP, each with a residual."""
        q, k, v = x @ wq, x @ wk, x @ wv
        scores = (q @ k.t()) * (1.0 / q.shape[-1] ** 0.5)
        x = x + scores.softmax(-1) @ v
        return x + (x @ w1).relu() @ w2

    def mlp(x, w1, w2):
        return ((x @ w1).relu() @ w2).softmax(-1)

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
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
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


TEMPLATE = (pathlib.Path(__file__).parent / "viz.html").read_text()

if __name__ == "__main__":
    main()
