# ir

The Rust graph the tracer builds and every pass rewrites: values, nodes and the primitives they apply.

## Files

- `mod.rs` holds the Rust graph: `TensorType` (dtype and shape), `Var` (an index into the type table), `Node` (one primitive applied to input vars, producing one output, with its scope, source lines, label and a `backward` flag), and `Graph` (types, inputs, nodes in topological order, outputs). It also has `prune` (drop nodes no output needs, keep all inputs), `placements` and `slot` (values stored inside another value's buffer: view slices and operands of in-place concatenates) and `precision_warnings` (accumulations rounded to a narrow dtype and later cast back up).
- `primitive.rs` holds the `Primitive` enum with `UnaryOp`, `BinaryOp` and `Reduction`, their names, and `Primitive::infer`, the output type inference every `Graph::apply` runs. It also defines the label markers `FUSION_SEPARATOR`, `NEURAL_ENGINE` and `UPDATES` used in fused node names.
- `tests.rs` holds unit tests of primitive type inference, graph display and the reference interpreter.

## Conventions

- Every `Graph::apply` type-checks through `Primitive::infer`; nodes are kept in topological order, and passes rely on it.
- `Scope` and `Lines` are interned `'static` slices, so nodes are cheap to clone.
