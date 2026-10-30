"""Builds the data of the graph visualization page (traced graph, fused plan, kernel timings) and writes the HTML."""

import datetime
import html
import json
import pathlib
import statistics

import lumen
from lumen.graph._C import FUSION_SEPARATOR
from lumen.graph_compiler._C import Plan
from lumen.profiler import ProfilerActivity, profile

from .common import KERNELS, LUMEN, source_of, type_text


def profile_steps(plan, run, runs, device):
    """For each step of ``plan``, its kernels in launch order as ``[name,
    median GPU us]`` over ``runs`` calls of ``run`` (running it) on
    ``device``."""
    sync = lumen.Stream(device).synchronize
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
    times = [[] for _ in range(nsteps)]
    by_id = {e["id"]: e for e in events}

    def in_op(e):
        """Whether event ``e`` is nested inside an op-level profiler range."""
        p = by_id.get(e["parent"])
        while p is not None:
            if p["kind"] == "op":
                return True
            p = by_id.get(p["parent"])
        return False

    # Top-level op ranges are the plan's steps in launch order, nsteps per run; any other count can't be matched up.
    ranges = sorted(
        (e for e in events if e["kind"] == "op" and not in_op(e) and not e["name"].startswith("lumen::")),
        key=lambda e: e["start_us"],
    )
    if nsteps == 0 or len(ranges) % nsteps:
        ranges = []
    for start in range(0, len(ranges), nsteps or 1):
        for i, step in enumerate(ranges[start : start + nsteps]):
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


def _kernel_entries(kernels):
    """``(name, us)`` kernels as dicts with the source file and anchor of each primitive kernel."""
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
        """Node id of the latest writer of ``buffer``, adding an input or buffer node on first use."""
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
            "scope": step["scope"],
            "sources": step["sources"],
        }
        nodes.append(node)
        for src, b in sources:
            edges.append([src, node["id"], b])
        writer[out_buffer] = node["id"]
        for b, _, _ in step["extra_outputs"]:
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
            "scope": node["scope"],
            "sources": node["sources"],
        }
        # Walk the unfused plan's steps in order to the next one computing this node; skipped steps' kernels are credited to it.
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


def code_of(views):
    """The text of each file of the traced program the nodes of ``views``
    were traced at (their ``sources``), those it can read."""
    files = {f for nodes in views for n in nodes for f, _ in n.get("sources", [])}
    code = {}
    for f in sorted(files):
        try:
            code[f] = pathlib.Path(f).read_text()
        except (OSError, UnicodeDecodeError):
            pass
    return code


def collect(graph, plan, inputs, title, runs=5, device=None, run=None, scalars=()):
    """Everything the page shows, as a JSON-able dict: ``graph``, its
    ``plan`` (fused, for ``device``, default the inputs') and its unfused
    plan, both profiled on ``inputs`` on that device (the graph's, then any
    the plan adds), ``plan`` by ``run`` if given (else ``plan.run``).
    ``scalars`` are the graph's runtime scalars (``Plan``'s)."""
    device = device or (str(inputs[0].device) if inputs else "cpu")
    host = str(device).split(":")[0] in ("cpu", "meta")
    unfused = Plan(graph) if host else Plan(graph, device, fuse=False, scalars=list(scalars))
    timed = not host
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
    views = {"fused": plan_view(plan, fused_kernels), "traced": graph_view(graph, unfused, unfused_kernels)}
    return {
        "title": title,
        "device": device,
        "fusion_separator": FUSION_SEPARATOR,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "inputs": [type_text(*graph.type_of(v)) for v in graph.inputs()],
        "graph_text": str(graph),
        "fused_text": str(plan),
        "unfused_text": str(unfused),
        "workspace": {"fused": plan.workspace_bytes, "unfused": unfused.workspace_bytes},
        "views": views,
        "sources": {f: (LUMEN / f).read_text() for f in files},
        "code": code_of(view["nodes"] for view in views.values()),
        "prelude": (KERNELS / "mps" / "kernels.metal").read_text(),
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


# The page's HTML template, with __TITLE__ and __DATA__ placeholders.
TEMPLATE = (pathlib.Path(__file__).parent / "page.html").read_text()
