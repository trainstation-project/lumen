# tracing

The Python tracer: it runs a function on traced tensors and records each primitive into the Rust `Graph` (`lumen.graph.ir._C`), which `lumen.graph_compiler` then compiles. `lumen.graph` re-exports `TracedTensor`, `compile`, `make_graph` and `prims`.

## Files

- `tracer.py` is the tracer: `TracedTensor` (the tensor type inside a trace, with torch-style methods, optional axis `names` and an optional `placement`: a sharded value's `shape` is global, its var the local shard; `distribute`, `to_local`, `full_tensor`, `redistribute`), `compile` and `Compiled`, `make_graph`, scopes, pytree helpers, broadcasting (`_broadcast`: positional, or by name when several operands are named), dimension helpers (`_dim`, `_dims`: positions or axis names) and dtype helpers (`_lift`, `_accum_dtype`, `_common_dtype`, `_dot`) the rest of the Python code uses.
- `prims.py` has one Python wrapper per primitive; each calls `bind`, which records the node into the current graph and names its result by the primitive's rule (`_C.names`) when an operand is named; with a sharded operand, `sharding.propagate` records it on the local shards.
- `__init__.py` marks the Python package.

How compiled calls trace, cache plans and treat scalars is in the parent README (`lumen/graph/README.md`).
