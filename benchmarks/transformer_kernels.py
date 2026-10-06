"""The MPS kernels of ``examples/transformer.py``'s training step, by GPU
time (``lumen.profiler``'s gpu events), per step: each kernel's label, its
Metal kernel, its shapes, launches and time; then the step's total and
kernel count.

    python benchmarks/transformer_kernels.py [--steps N] [--top K]

``--op`` times one op on its own instead, compiled for MPS, for sweeping a
kernel's parameters: a matmul ``a @ b`` of the shapes and dtype given.

    python benchmarks/transformer_kernels.py --op matmul --shapes 1024x64 64x64 --dtype bfloat16
    python benchmarks/transformer_kernels.py --op matmul_tn --shapes 1024x64 1024x64

(``matmul_tn``: ``a.t() @ b``, a weight's gradient.)
"""

import argparse
import pathlib
from collections import defaultdict

import numpy as np

import lumen
from lumen.profiler import ProfilerActivity, profile

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def gpu_events(run, steps):
    """The gpu events of ``steps`` calls of ``run`` (after a warm-up), each
    with the shapes of its op's ``inputs``: timed with the MPS activity
    alone (recording the host's ops too slows dispatch, and the GPU clocks
    down between kernels), the shapes from one more step recorded with it,
    by launch order."""

    def events(steps, activities):
        lumen.mps.synchronize()
        with profile(activities=activities, record_shapes=True) as prof:
            for _ in range(steps):
                run()
            lumen.mps.synchronize()
        return [e for e in prof.events() if e["kind"] == "gpu"]

    for _ in range(20):
        run()
    shaped = events(1, [ProfilerActivity.CPU, ProfilerActivity.MPS])
    timed = events(steps, [ProfilerActivity.MPS])
    assert len(timed) == steps * len(shaped), "every step launches the same kernels"
    # Each launch's median over the steps (the GPU's clock varies).
    for k, e in enumerate(shaped):
        e["duration_us"] = float(np.median([t["duration_us"] for t in timed[k :: len(shaped)]]))
    return shaped


def table(events, top):
    """Each (label, kernel, shapes)'s launches and time a step (each
    launch's median), most first; the total."""
    rows = defaultdict(lambda: [0, 0.0])
    for e in events:
        shapes = " ".join("x".join(map(str, s)) + f":{d}" for d, s in e["inputs"])
        key = (e["name"], e.get("kernel", ""), shapes)
        rows[key][0] += 1
        rows[key][1] += e["duration_us"]
    total = sum(r[1] for r in rows.values())
    print(f"per step: {total:8.1f} us GPU, {len(events)} kernels")
    print(f"{'us/step':>8} {'%':>5} {'n':>3} {'us/launch':>9}  kernel / label / inputs")
    for (name, kernel, shapes), (n, us) in sorted(rows.items(), key=lambda r: -r[1][1])[:top]:
        print(f"{us:8.1f} {100 * us / total:5.1f} {n:3d} {us / n:9.1f}  {kernel}  {name[:50]}  {shapes}")
    return total


def transformer(steps, top):
    """examples/transformer.py's model and optimizer, compiled (its source
    up to its training loop), its training step profiled on random
    batches."""
    source = (EXAMPLES / "transformer.py").read_text()
    source = source[: source.index("for step in range(STEPS + 1)")]
    import sys

    sys.path.insert(0, str(EXAMPLES))
    scope = {"__name__": "transformer_example"}
    exec(compile(source, str(EXAMPLES / "transformer.py"), "exec"), scope)
    x, y, _ = scope["batch"]()
    run = lambda: scope["train_step"](scope["model"], scope["opt"], x, y)  # noqa: E731
    return table(gpu_events(run, steps), top)


def op(name, shapes, dtype, steps, top, repeats=16):
    """``name`` of operands of ``shapes`` and ``dtype``, ``repeats`` of them
    in one compiled function (on as many operands: back to back, the GPU
    kept busy, as in a step), compiled as examples/transformer.py compiles
    (deterministic)."""
    lumen.config.compiler.deterministic = True
    rng = np.random.default_rng(0)
    args = [
        lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to(dtype=dtype).to("mps")
        for _ in range(repeats)
        for s in shapes
    ]
    fns = {"matmul": lambda a, b: a @ b, "matmul_tn": lambda a, b: a.t() @ b, "matmul_nt": lambda a, b: a @ b.t()}
    f = lumen.compile(lambda *xs: [fns[name](*xs[k : k + len(shapes)]) for k in range(0, len(xs), len(shapes))])
    return table(gpu_events(lambda: f(*args), steps), top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--op", choices=["matmul", "matmul_tn", "matmul_nt"])
    parser.add_argument("--shapes", nargs="*", default=[])
    parser.add_argument("--dtype", default="bfloat16")
    a = parser.parse_args()
    if a.op:
        shapes = [tuple(int(d) for d in s.split("x")) for s in a.shapes]
        op(a.op, shapes, a.dtype, a.steps, a.top)
    else:
        transformer(a.steps, a.top)
