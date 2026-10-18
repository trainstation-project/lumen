# `lumen/graph`

The IR. `lumen.compile` traces a Python function once per input signature
into a `Graph` of strict, JAX `lax`-style `Primitive`s (a jaxpr); the
graph compiler (`lumen/graph_compiler`) rewrites and fuses it and compiles
it into a static `Plan`; the CPU interpreter `lumen/ops/reference.rs` defines
what each primitive computes. This folder holds the Rust IR, its Python
bindings, the Python tracer that builds graphs (`TracedTensor`,
`compile`), the primitive functions (`prims`), and the HTML graph viewer.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `Graph`, `Node`, `Var`, `TensorType`, `View`, `Range`/`Scope`/`Lines`; `apply`, `prune`, `placements`, `slot`, `label`, `precision_warnings`; `intern*` helpers; jaxpr-style `Display`. |
| `primitive.rs` | The `Primitive` enum, `name()`, `infer()` (type checking), `accum_dtypes()`, `Display`; `FUSION_SEPARATOR`, `NEURAL_ENGINE`. |
| `custom.rs` | The hook through which a `CustomCall` step calls its (Python) function by handle. |
| `python.rs` | `lumen._C.Graph` bindings (build by primitive name + params dict, `nodes()`, `attentions()`, `run()` on the reference executor), `_register_custom_op`, `FUSION_SEPARATOR`. |
| `_C.pyi` | Type stub for `python.rs`. |
| `__init__.py` | Re-exports `Graph`, `TracedTensor`, `compile`, `make_graph`, `prims` (tracer imported before `prims`). |
| `tracer.py` | Tracing: `TracedTensor` (torch-like methods), `compile`, `make_graph`, trace-global state (scopes, sources, tapes, RNG). |
| `prims.py` | One Python function per primitive (`jax.lax`-like); `bind` records a node. |
| `viz.py` | `dump_graph`'s backend: profiles fused and unfused plans and writes one self-contained HTML page; also a CLI (`python -m lumen.graph.viz`). |
| `viz.html` | The page template (`__TITLE__`, `__DATA__` placeholders), its CSS and JS. |
| `tests.rs` | Unit tests: type rules, display, reference-executor values. |

## Core types (`mod.rs`)

* `TensorType { dtype, shape }` — a value's static type (`ShapedArray`);
  no strides or device. Displays as `f32[2,3]`.
* `Var = usize` — a value: an index into `Graph::types`.
* `Node { primitive, inputs, output, scope, sources, label }` — one SSA
  definition `output = primitive(inputs...)`.
* `Graph { types, inputs, nodes, outputs, scope, sources }` — SSA, nodes
  in definition order. Build with `input(ty)`, `apply(prim, &inputs)` and
  `set_outputs`. `apply` calls `Primitive::infer` on the operand types,
  so **every node is type-checked at construction**; an error is a
  `String` (a `ValueError` in Python).
* `prune()` — dead-code elimination keeping inputs; returns the new graph
  and old→new `Var` map. The tracer prunes after every trace.

### Metadata carried by nodes

* `scope: Scope` — the `record_function` ranges (`Range { name, call }`)
  open when traced, outermost first; the plan opens the same profiler
  ranges around the node's step.
* `sources: Lines` — `(file, line)`s of the traced program (first frame
  outside the lumen package) that computed it.
* `label: Option<&'static str>` — a profile name set by a pass (e.g. a
  split-K dot's partials); `Graph::label(node)` falls back to the
  primitive name, or `cast(float32 -> bfloat16)` / `dot_general → cast(...)`.
* Passes building new nodes call `set_origin(node)` / `set_origins(nodes)`
  (or `set_scope`) first so the new nodes inherit scope and sources.
* Scopes, lines and labels are `&'static`, interned once per distinct value
  (`intern`, `intern_lines`, `intern_scope`; leaked deliberately).

## Primitives (`primitive.rs`)

Strict: elementwise operands already share shape *and* dtype; no implicit
broadcast or promotion; results are values.

| Kind | Primitives |
| --- | --- |
| Elementwise binary | `Add Sub Mul Div Max Eq Lt` (`Eq`/`Lt` → bool) |
| Elementwise unary | `Neg Exp Log Sqrt Tanh Logistic`, `Cast { new_dtype }` |
| Ternary | `Select` |
| Reductions / scans | `ReduceSum { axes, accum_dtype }`, `ReduceMax { axes }`, `Cumsum { axis, reverse, accum_dtype }` |
| Contraction | `DotGeneral { lhs/rhs_contracting, lhs/rhs_batch, accum_dtype, output_dtype }` |
| Layout | `Reshape`, `BroadcastInDim`, `Transpose`, `Slice { .., view }`, `Concatenate { dimension, in_place }` |
| Indexing | `DynamicSlice`, `DynamicUpdateSlice`, `Gather { axis }`, `ScatterAdd { axis }` |
| Creation | `Full`, `Iota`, `RandomBits { shape, offset }` (counter-based Philox) |
| Host/device | `Wait`, `ToHost`, `ToDevice` |
| Fused ops | `LinearCrossEntropy` (cut cross-entropy, `[2, B]` f32 out) |
| Compiler-made | `Fusion { name, label, body }`, `FusionOutput { index, ty }` |
| User | `CustomCall { label, kernel, mutated, overlappable }` |

* `accum_dtype` on dots/sums/cumsum is explicit: the operands' dtype, or
  float32 for narrower floats. `output_dtype` on a dot is the operands' or
  `accum_dtype`, rounded once. `accum_dtypes()` reports these (for a
  fusion, all of its body's).
* `Fusion`: `body` is a `Graph` run as one kernel; its value is the
  body's first output, others are read by `FusionOutput` nodes (XLA
  `get-tuple-element`). `label` joins member names with
  `FUSION_SEPARATOR` (`" → "`). A name starting with `NEURAL_ENGINE`
  (`"coreml_"`) is a Core ML program, not a Metal kernel.
* `CustomCall`: `kernel` is a handle from `_register_custom_op`; operands
  at `mutated` are written in place; value is `mutated[0]`'s new value,
  later ones via `FusionOutput`. `overlappable` `(m, j)` pairs allow the
  planner to alias mutated `m` with read `j`. At run time
  `custom::call` dispatches through the hook `python.rs` installs.
* `infer()` checks arity and types for each primitive (e.g. no bool
  arithmetic, float-only transcendental ops, some bf16 math rejected with a
  hint to convert).

### Placements: `View`, `Slice.view`, `Concatenate.in_place`

Two primitive flags let a value live inside another value's memory. Both
default to false from the tracer and are set by device compiler passes:

* `Slice { view: true }` — the slice is its operand's memory read as a
  strided view; no step computes it.
* `Concatenate { in_place: true }` — each operand is written directly at
  its place in the result by the kernel computing it; no copy step.

`Graph::placements()` derives, from these flags alone, for every value
`Some((host_var, View { offset, strides }))` or `None`. `slot(node, k)`
gives operand `k`'s view in a concatenate's result. Passes, kernel
codegen (`codegen::mps` output views) and the plan all read placements
from here — do not track aliasing elsewhere.

### `precision_warnings`

Flags a dot/sum that accumulates wide, outputs narrow, then is cast back
wide through dtype-preserving ops. The tracer turns them into Python
warnings at the source line.

## Python side

* `prims.py` — `bind(name, *operands, **params)` lifts operands, checks
  they belong to the current trace, calls `Graph.apply` and records the
  node (`tracer._record`: source line, scope, autodiff tapes). Shape/dtype
  rule violations raise `ValueError` from Rust. `_custom_call` emits
  `custom_call` + `fusion_output`s.
* `python.rs` maps primitive names and param dicts to `Primitive`
  (`primitive()`); `fusion` is not constructible from Python.
  `nodes()` returns dicts incl. a fusion's kernel name, body text and Metal
  source; `attentions()` exposes what the compiler's attention matcher
  sees.
* `tracer.py`:
  * `TracedTensor` — torch-like operators, layout methods, dtype casts,
    in-place ops (`copy_`, `add_`, `__setitem__`, ...), `.backward()`.
    Views keep `_base`/`_ops` and re-derive from the base after it is
    assigned; writes through a view become `dynamic_update_slice` etc.
    (`_scatter`).
  * `_trace(fn, args, device)` — graph inputs are tensor args, then
    runtime float scalars (weak-typed 0-d float32), then module weights,
    then the RNG state if `lumen.random` is used. Assigned weights /
    scalars / tensor args become extra outputs (written back after each
    call). Then prune and emit precision warnings.
  * `compile(fn, device=None)` — caches a `Plan` per signature, device and
    `repr(config.compiler)`; runs in a single preallocated workspace;
    handles meta-tensor compile-only calls, weight packing for merged dots,
    Neural Engine invalidation, and `compiled.dump_graph(...)`.
  * `make_graph(fn)` — returns the traced `Graph` (like `jax.make_jaxpr`).
* `viz.py` / `viz.html` — `collect` profiles the fused plan and an unfused
  plan (`fuse=False`) and gathers graph/plan text, per-node kernels with
  GPU time and Metal sources; `write` fills the template.

## Invariants & gotchas

* No implicit dtype changes anywhere: operands must already match; casts
  are explicit `Cast` nodes.
* `Graph` fields are `pub(crate)`; outside the crate use `inputs()`,
  `nodes()`, `outputs()`, `type_of()`.
* After creating nodes in a pass, carry `label` over explicitly where it
  matters (`prune` does via `set_label`).
* A new primitive needs: the enum variant, `name()`, `infer()`, `Display`
  params, `python.rs` parsing, a `prims.py` function, a reference
  implementation in `lumen/ops/reference.rs`, and device support.
* Tracing happens only inside `lumen.compile`; `prims.bind` and custom ops
  raise outside a trace.

## Related

* [`lumen/graph_compiler`](../graph_compiler/README.md) — passes, fusion, `Plan` and scheduling over this IR.
* [`lumen/codegen`](../codegen/README.md) — Metal / Core ML generated from `Fusion` bodies.
* [`lumen/custom_op`](../custom_op/README.md) — builds `CustomCall` nodes.
* `lumen/ops/reference.rs` — the CPU interpreter defining each primitive's semantics.
* [`lumen/autograd`](../autograd/README.md) — uses the tracer's tapes for `.backward()`.
* [`lumen/profiler`](../profiler/README.md) — `record_function` ranges that become node scopes.
