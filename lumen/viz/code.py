"""Dumps the generated code of every kernel launch of a compiled program to a directory."""

import importlib.util
import pathlib
import re

import lumen

from .common import KERNELS, _traced_at, source_of, type_text


def _slug(text):
    """``text`` reduced to a filename-safe slug of at most 60 characters."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "step"


def _snippet(text, anchor):
    """The ``#define`` macro or function in ``text`` starting at the line beginning with ``anchor``, or all of ``text``."""
    lines = text.splitlines()
    start = next((k for k, line in enumerate(lines) if line.startswith(anchor)), None)
    if start is None:
        return text
    # A macro ends at the first line without a continuation backslash; a function at its closing brace.
    if anchor.startswith("#define"):
        end = next((k for k in range(start, len(lines)) if not lines[k].endswith("\\")), len(lines) - 1)
    else:
        end = next((k for k in range(start, len(lines)) if lines[k] == "}"), len(lines) - 1)
    return "\n".join(lines[start : end + 1]) + "\n"


def _launch_code(step, kernel, code):
    """Source and file extension of a launch's kernel: captured code, a generated kernel, a fusion or a primitive's snippet."""
    from lumen.graph_compiler._C import _kernel_source

    if code is not None:
        return code, "py" if "@triton.jit" in code else "mlir"
    source = _kernel_source(kernel)
    if source is not None:
        return source, "metal"
    fusion = step["fusion"]
    if fusion and fusion["source"]:
        return fusion["source"], "metal"
    src = source_of(kernel)
    if src is not None:
        path, anchor = src
        return f"// {path}\n" + _snippet((KERNELS / path).read_text(), anchor), "metal"
    return step["text"] + "\n", "txt"


def _header(step, at, ext):
    """Comment header for a dumped kernel: the step, where it was traced and its graph."""
    fusion = step["fusion"]
    graph = fusion["body"] if fusion else step["text"]
    mark = "#" if ext in ("txt", "py") else "//"
    lines = [f"step {at}: {step['label']}", *_traced_at(step["sources"]), *graph.rstrip().splitlines()]
    return "".join(f"{mark} {line}\n" for line in lines) + "\n"


def _literal(value):
    """Python literal for ``value``, spelling non-finite floats as ``float(...)``."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return f'float("{value}")'
    return repr(value)


def program_source(graph, name="program"):
    """Python source of a function named ``name`` that re-traces ``graph`` via ``prims.bind``."""
    names = {v: f"in{k}" for k, v in enumerate(graph.inputs())}
    lines = ["from lumen.graph import prims", "", ""]
    lines += [f"# {names[v]}: {type_text(*graph.type_of(v))}" for v in graph.inputs()]
    lines.append(f"def {name}({', '.join(names[v] for v in graph.inputs())}):")
    traced = None
    for node in graph.nodes():
        at = _traced_at(node["sources"])
        if at and at != traced:
            lines += [f"    # {line}" for line in at]
            traced = at
        names[node["output"]] = f"v{node['output']}"
        operands = [f'"{node["primitive"]}"', *(names[v] for v in node["inputs"])]
        operands += [f"{key}={_literal(value)}" for key, value in node["params"].items()]
        lines.append(f"    {names[node['output']]} = prims.bind({', '.join(operands)})")
    outputs = [names[v] for v in graph.outputs()]
    lines.append(f"    return {', '.join(outputs)}" + ("," if len(outputs) == 1 else ""))
    return "\n".join(lines) + "\n"


def _pallas(step, at, module, spec):
    """Python script that rebuilds a launch's Mosaic program from its Pallas template ``module`` and JSON ``spec``,
    or, for a generated program (``module`` ``generated``), its source run on the device's memory (``spec``)."""
    if module == "generated":
        return _header(step, at, "txt") + spec
    return (
        _header(step, at, "txt")
        + f"import json\n\nfrom {module} import program\n\n"
        + f"spec = json.loads({spec!r})\nmosaic = program(spec)\n"
    )


def dump_code(graph, plan, run, device, directory):
    """Runs ``run`` capturing its kernel launches and writes each launch's code, the graph, the plan and an index
    to ``directory``; returns the written kernel files."""
    from lumen.graph_compiler._C import _capture_launches, _captured_launches

    _capture_launches()
    try:
        run()
        lumen.Stream(device).synchronize()
    finally:
        launches = _captured_launches()
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    steps = plan.steps()
    index, written = [], []
    templates = set()
    # Drop launches outside this plan's steps.
    launches = [launch for launch in launches if launch[0] < len(steps)]
    for n, (at, label, kernel, code, template) in enumerate(launches):
        text, ext = _launch_code(steps[at], kernel, code)
        name = f"{n:04d}-{_slug(label)}-{_slug(kernel)}.{ext}"
        (directory / name).write_text(_header(steps[at], at, ext) + text)
        index.append(f"{n:04d}  step {at:4d}  {label}  {kernel}  {name}")
        written.append(directory / name)
        if template is not None:
            module, spec = template
            pallas = name.rsplit(".", 1)[0] + ".pallas.py"
            (directory / pallas).write_text(_pallas(steps[at], at, module, spec))
            if module == "generated":
                index.append(f"{n:04d}  {' ' * 9}  its Pallas: its generated program  {pallas}")
                continue
            index.append(f"{n:04d}  {' ' * 9}  its Pallas: {module}'s program(spec)  {pallas}")
            templates.add(module)
    for module in sorted(templates):
        source = pathlib.Path(importlib.util.find_spec(module).origin)
        (directory / "pallas").mkdir(exist_ok=True)
        (directory / "pallas" / f"{module}.py").write_text(source.read_text())
        index.append(f"pallas/{module}.py  the template {module}")
    if str(device).startswith("mps"):
        (directory / "prelude.metal").write_text((KERNELS / "mps" / "kernels.metal").read_text())
        index.append("prelude.metal  the Metal prelude every kernel above is compiled with")
    (directory / "graph.txt").write_text(str(graph) + "\n")
    (directory / "program.py").write_text(program_source(graph))
    (directory / "plan.txt").write_text(str(plan) + "\n")
    index[:0] = [
        "graph.txt  the traced program",
        "program.py  the traced program as Python (prims.bind a node), tracing to the same graph",
        "plan.txt  its plan: the steps below and their buffers",
    ]
    (directory / "index.txt").write_text("\n".join(index) + "\n")
    return written
