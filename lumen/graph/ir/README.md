# ir

The Rust graph the tracer builds and every pass rewrites: values, nodes and the primitives they apply.

## Files

- `mod.rs` declares the files and re-exports their types, so the crate names them `crate::graph::Graph`.
- `types.rs` holds `TensorType` (dtype and shape) and `Var` (an index into a graph's type table).
- `node.rs` holds `Node` (one primitive applied to input vars, producing one output, with its scope, source lines, label and a `backward` flag), `Scope`, `Range` and `Lines`, and their interning as `'static` slices (`intern`, `intern_lines`, `intern_scope`).
- `graph.rs` holds `Graph` (types, inputs, nodes in topological order, outputs): building it (`input`, `apply`, `set_outputs`, the scope, origin and label setters), reading it back (`type_of`, `inputs`, `nodes`, `outputs`, `label`) and its display.
- `prune.rs`: `prune` drops nodes no output needs, keeping all inputs.
- `placements.rs`: `View`, `slot` and `placements`, the values stored inside another value's buffer (view slices and operands of in-place concatenates).
- `precision.rs`: `precision_warnings`, accumulations rounded to a narrow dtype and later cast back up.
- `primitive.rs` holds the `Primitive` enum with `UnaryOp`, `BinaryOp` and `Reduction`, their names, and `Primitive::infer`, the output type inference every `Graph::apply` runs. It also defines the label markers `FUSION_SEPARATOR`, `NEURAL_ENGINE` and `UPDATES` used in fused node names.
- `python.rs` (with the `python` feature) binds `Graph` to Python as `lumen.graph._C.Graph` (`input`, `apply`, `prune`, `set_outputs`, `nodes`, `run` on the reference interpreter), and `scope_names`.
- `convert.rs` (with the `python` feature) converts a primitive's name plus params dict to `Primitive` (`primitive`) and back (`primitive_dict`).
- `tests.rs` holds unit tests of primitive type inference, graph display and the reference interpreter.

## Conventions

- Every `Graph::apply` type-checks through `Primitive::infer`; nodes are kept in topological order, and passes rely on it.
- `Scope` and `Lines` are interned `'static` slices, so nodes are cheap to clone.
