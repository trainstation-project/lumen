"""Graph compilers: ``Plan(graph, device)``, a graph compiled for execution
by the device's graph compiler (bindings in ``lumen/graph_compiler/python.rs``;
on MPS, loop fusion into generated Metal kernels, ``lumen/codegen``), its
memory planned (``print`` it to read it). ``lumen.compile`` builds and
caches these; ``lumen.config.compiler`` holds the compilers' flags.
"""

from lumen._C import Plan

__all__ = ["Plan"]
