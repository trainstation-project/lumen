"""Compiled execution: ``lumen.compile`` traces a function into a graph of
primitive ops (bindings in ``lumen/graph/python.rs``) and runs the graph.

* ``lumen.graph.prims`` (re-exported as ``lumen.prims``): the strict
  primitives, modeled on ``jax.lax``.
* ``Plan(graph, device)``: the graph compiled for execution by the
  device's graph compiler (``lumen/compiler``: on MPS, loop fusion into
  generated Metal kernels), memory planned (``print`` it to read it);
  ``lumen.compile`` builds and caches these.
* ``lumen.graph.viz``: a compiled function's graph and plans as an HTML
  page, with each node's kernels, GPU time and Metal source
  (``lumen.compile(fn).dump_graph("graph.html")``).
* ``lumen.graph.tracer``: tracing, and the torch-like API traced tensors
  have (``x @ w``, ``x.softmax(-1)``, ``lumen.where``, ...), written on the
  primitives.
"""

from lumen._C import Graph, Plan

# The tracer first: it imports prims, whose functions its class body uses.
from lumen.graph.tracer import (
    TracedTensor,
    compile,
    exp,
    log,
    make_graph,
    matmul,
    maximum,
    minimum,
    rsqrt,
    rms_norm,
    sigmoid,
    softmax,
    tanh,
    where,
)
from lumen.graph import prims

__all__ = [
    "Graph", "Plan", "TracedTensor", "compile", "make_graph", "prims",
    "where", "matmul", "maximum", "minimum", "exp", "log", "rsqrt", "tanh", "sigmoid", "softmax", "rms_norm",
]
