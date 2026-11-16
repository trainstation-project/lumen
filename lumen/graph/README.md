# graph

The traced dataflow graph and the Python tracer that builds it. This is the start of the pipeline: `lumen.compile` traces a Python function into a `Graph` here, then hands it to `lumen/graph_compiler` (through `lumen.graph_compiler._C.Plan`), which runs its passes and lowers it to a plan.

## Files

- `mod.rs` holds the Rust graph: `TensorType` (dtype and shape), `Var` (an index into the type table), `Node` (one primitive applied to input vars, producing one output, with its scope, source lines, label and a `backward` flag), and `Graph` (types, inputs, nodes in topological order, outputs). It also has `prune` (drop nodes no output needs, keep all inputs), `placements` and `slot` (values stored inside another value's buffer: view slices and operands of in-place concatenates) and `precision_warnings` (accumulations rounded to a narrow dtype and later cast back up).
- `primitive.rs` holds the `Primitive` enum with `UnaryOp`, `BinaryOp` and `Reduction`, their names, and `Primitive::infer`, the output type inference every `Graph::apply` runs. It also defines the label markers `FUSION_SEPARATOR`, `NEURAL_ENGINE` and `UPDATES` used in fused node names.
- `custom.rs` is the bridge from a `custom_call` kernel at run time back to the Python function registered for it (`set_hook`, `call`).
- `python.rs` binds `Graph` to Python (`input`, `apply`, `prune`, `set_outputs`, `run` on the reference interpreter) and converts primitive names plus a params dict to `Primitive` and back.
- `tracer.py` is the tracer: `TracedTensor` (the tensor type inside a trace, with torch-style methods), `compile` and `Compiled`, `make_graph`, scopes, pytree helpers (`_tree_map` over tensors, modules, lists, tuples and classes registered with `register_tree`, such as `lumen.named.NamedTensor`) and dtype helpers (`_lift`, `_accum_dtype`, `_common_dtype`, `_dot`) the rest of the Python code uses.
- `prims.py` has one Python wrapper per primitive; each calls `bind`, which records the node into the current graph.
- `__init__.py` exports `Graph`, `TracedTensor`, `compile`, `make_graph` and `prims`.
- `_C.pyi` types the native module. `tests.rs` holds unit tests.

## How it works

- `compile(fn, device)` returns a `Compiled`. On the first call with each signature (tensor dtypes and shapes, values of non-float arguments) it traces `fn` and builds a `Plan`; later calls with that signature reuse it.
- A `register_tree` argument is flattened into a marker holding its class and static part (part of the signature) and its tensors, which become ordinary tensor arguments; `fn` gets it rebuilt (`_flattened`, `_regrouping`).
- Float arguments are runtime scalars: plan inputs whose value each call passes. Ints and other values are static and part of the signature.
- `lumen.nn.Module` weights are placed on the device when the first compiled function taking them compiles, and every compiled function taking them shares that memory. A call with meta tensors compiles without running.
- Scalar-only work on runtime scalars and constants (`_Host`) is recorded into a separate host graph that runs on the CPU before the device plan; device ops read its results as scalar inputs.
- Results are views into the plan's workspace and are overwritten by the next call.

## Conventions

- Every `Graph::apply` type-checks through `Primitive::infer`; a new primitive needs a variant, a name, inference, a `prims.py` wrapper, a `python.rs` conversion and autograd rules in `lumen/autograd/rules.py`.
- Nodes are kept in topological order; passes rely on it.
- `Scope` and `Lines` are interned `'static` slices, so nodes are cheap to clone.
- `Node::backward` marks gradient nodes traced by autograd; passes that rebuild nodes (rematerialization, scheduling, the attention backward pass, precision preservation) carry or read it, so keep it set on rewritten nodes.
