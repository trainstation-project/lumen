# graph

The traced dataflow graph and the Python tracer that builds it. This is the start of the pipeline: `lumen.compile` traces a Python function into a `Graph` here, then hands it to `lumen/graph_compiler` (through `lumen.graph_compiler._C.Plan`), which runs its passes and lowers it to a plan.

## Subfolders

- `ir/`: the Rust graph (`Graph`, `Node`, `TensorType`, `Var`) and its primitives (`Primitive`, their type inference). `mod.rs` re-exports it, so the rest of the crate names them `crate::graph::Graph`.
- `propagation/`: how each primitive's result names and sharding follow from its operands' (Rust), and the Python that runs primitives on sharded operands by those rules (`sharding.py`, with `cost.py`, `redistribute.py` and `gradients.py`).
- `tracing/`: the Python tracer (`TracedTensor`, `compile`, `make_graph`) and the primitive wrappers (`prims`) that record nodes.

## Files

- `mod.rs` declares the subfolders and re-exports `ir`.
- `custom.rs` is the bridge from a `custom_call` kernel at run time back to the Python function registered for it (`set_hook`, `call`).
- `python.rs` builds the `lumen.graph._C` native module: the `Graph` class (`ir/python.rs`), `names` and `strategies` (`propagation/python.rs`), and the custom-op hook (`_register_custom_op`, the function `custom.rs` calls).
- `__init__.py` exports `Graph`, `TracedTensor`, `compile`, `make_graph` and `prims`.
- `_C.pyi` types the native module.

## How it works

- `compile(fn, device)` returns a `Compiled`. On the first call with each signature (tensor dtypes and shapes, values of non-float arguments) it traces `fn` and builds a `Plan`; later calls with that signature reuse it.
- A tensor argument's or weight's names and placement become its traced input's (`_input`; a sharded one's input has its local shard's shape) and are part of the signature: the same tensor named or placed otherwise traces anew. Results come back named and placed as their traced values.
- Float arguments are runtime scalars: plan inputs whose value each call passes. Ints and other values are static and part of the signature.
- `lumen.nn.Module` weights are placed on the device when the first compiled function taking them compiles, and every compiled function taking them shares that memory. A call with meta tensors compiles without running.
- Scalar-only work on runtime scalars and constants (`_Host`) is recorded into a separate host graph that runs on the CPU before the device plan; device ops read its results as scalar inputs.
- Results are views into the plan's workspace and are overwritten by the next call.

## Conventions

- Every `Graph::apply` type-checks through `Primitive::infer`; a new primitive needs a variant, a name, inference, its name rule and sharding strategies (`propagation/`), a `tracing/prims.py` wrapper, a `python.rs` conversion and autograd rules in `lumen/autograd/rules.py`.
- Nodes are kept in topological order; passes rely on it.
- `Scope` and `Lines` are interned `'static` slices, so nodes are cheap to clone.
- Placements are Python-side metadata too: the graph holds local values and the explicit collectives `sharding` recorded, so autograd differentiates it as any local program. Every process must choose the same strategies (they depend on global shapes alone) and issue the same collectives.
- Names are Python-side metadata only: the `Graph`, its passes and plans never see them, so a named program traces to the graph its positional version gives. Ops that rename without computing (`rename`, `refine_names`) return a view of the same value, no node; a named split or merge (`unflatten`, `flatten` by names), an alignment by name and indexing that drops axes name their reshape's result themselves, as no reshape rule can.
- `Node::backward` marks gradient nodes traced by autograd; passes that rebuild nodes (rematerialization, scheduling, the attention backward pass, precision preservation) carry or read it, so keep it set on rewritten nodes.
