"""Compiled execution: ``lumen.compile`` traces a function into a graph of
primitive ops (bindings in ``lumen/graph/python.rs``) and runs the graph.

* ``lumen.graph.prims`` (re-exported as ``lumen.prims``): the strict
  primitives, modeled on ``jax.lax``.
* ``Plan(graph)``: the graph compiled for execution (memory planned,
  ``print`` it to read it); ``lumen.compile`` builds and caches these.
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
    promote_types,
    result_type,
    rsqrt,
    sigmoid,
    softmax,
    tanh,
    where,
)
from lumen.graph import prims

__all__ = [
    "Graph", "Plan", "TracedTensor", "compile", "make_graph", "prims", "promote_types", "result_type",
    "where", "matmul", "maximum", "minimum", "exp", "log", "rsqrt", "tanh", "sigmoid", "softmax",
]
