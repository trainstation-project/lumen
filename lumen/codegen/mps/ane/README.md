# `lumen/codegen/mps/ane`

Lowering of a fusion body to a Core ML ML Program, the only public way onto
the Apple Neural Engine. With `lumen.config.compiler.neural_engine` on, the
MPS compiler (`lumen/graph_compiler/mps/ane/mod.rs`) carves out regions of
large float16 dots of fixed weights (and the float16 work after them) as
`Primitive::Fusion`s named `coreml_*` (`graph::NEURAL_ENGINE`), and calls
`mil::lower` here to turn each body into a serialized Core ML model spec.
Compiling the spec, asking Core ML which device runs each op, and running
it all happen in `graph_compiler/mps/ane` and the Objective-C++ shim; this
folder only produces bytes.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Declares `mil` and `proto`. |
| `proto.rs` | `Message`: a minimal protobuf writer (varints, length-delimited fields, packed ints/floats, map `entry`), builder style. No decoding. |
| `mil.rs` | `lower(graph, constants) -> (spec bytes, Owners)`, plus `io_shape`, `mil_dtype`, `array_dtype` (Core ML type codes) used by the caller. |

## How `lower` works

* Builds a `Model` (spec version 8) whose `ML Program` has one function
  `main` with one `CoreML7` block. Field numbers are written by hand as
  integers in `Message` calls; there is no generated protobuf schema.
* Graph inputs become function inputs `in0, in1, ...`, except those with a
  `constants[k]` byte buffer, which are baked in as `const` ops (the frozen
  weights). Outputs are `out0, out1, ...` via `identity` (or `reshape` for
  scalars: Core ML I/O is never rank 0, see `io_shape`).
* Each graph node maps to MIL ops in `Lower::lower_node`: elementwise
  (`add`, `real_div`, `maximum`, `sigmoid`, `select`, ...), `cast`,
  `reshape`, `transpose`, `reduce_sum`/`reduce_max`, `slice_by_index`,
  `concat`, `gather`, `matmul` for `DotGeneral` (`Lower::dot`). Anything
  else is an error ("Core ML programs do not run ...") and the region is
  not offloaded.
* Values are kept *compact*: `full`, `iota` and `broadcast_in_dim` produce
  tensors with size-1 dimensions that rely on MIL broadcasting;
  `Lower::full(v)` materializes a value at its real shape with `tile` only where an op
  needs it.
* `Owners` records, for each MIL op output name, the graph node index it
  came from (`None` for inputs/outputs). The caller matches this against
  Core ML's compute plan to keep only the nodes placed on the Neural
  Engine.

## Invariants & gotchas

* Supported dtypes: float16, float32, int32, bool for values
  (`mil_dtype`); float16, float32, int32 for I/O arrays (`array_dtype`).
  Others are rejected with an error, never converted.
* `gather` indices are clamped into range with `maximum`/`minimum` before
  the MIL `gather` (whose `validate_indices` is false).
* Integer constants (shapes, axes, perms) are emitted as int32.
* The Neural Engine accumulates float16 dots narrower than float32, which
  is why the feature is opt-in (see the caller's module docs).
* Not compiled without Metal: the parent `codegen::mps` is behind
  `cfg(lumen_mps_linked)`.

## Related

* [`lumen/codegen/mps`](../README.md) — the parent; Metal kernels for everything not offloaded.
* [`lumen/graph_compiler/mps/ane`](../../../graph_compiler/mps/ane/README.md) — chooses regions, compiles/caches programs, runs them in the plan.
* [`lumen/graph`](../../../graph/README.md) — `Graph`, `Primitive`, `NEURAL_ENGINE`.
