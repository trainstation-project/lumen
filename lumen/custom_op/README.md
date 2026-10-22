# `lumen/custom_op`

User-defined ops inside `lumen.compile`: `lumen.custom_op.custom_op` wraps a Python
function as an op that the compilers treat as opaque, and `MetalKernel`
lets that function launch a Metal kernel of its own into lumen's MPS
stream. A custom op returns nothing: its results are the arguments it
mutates in place. Tracing records it as one `Primitive::CustomCall` node
(see [`lumen/graph`](../graph/README.md)); when the plan reaches that step it
calls the Python function back with real tensors.

## Files

| File | Contents |
| --- | --- |
| `__init__.py` | `custom_op(name, fn=None, *, mutates_args, overlappable=None)` decorator and the `CustomOp` class; re-exports `MetalKernel`. |
| `mod.rs` | Declares `mps`, `python` (feature `python`) and `tests`. |
| `mps.rs` | `MetalKernel` (compile Metal source, `launch` on MPS tensors) and `Grid` (`Threads` or `Groups`, 3-D; `Elementwise`, a generated loop's threads and threadgroup size). |
| `python.rs` | The `MetalKernel` pyclass (in `lumen.custom_op._C`, re-exported by `__init__.py`) wrapping `mps::MetalKernel`; `register`. |
| `_C.pyi` | Type stub for `MetalKernel`. |
| `tests.rs` | Launches an `axpy` kernel into the MPS stream (skips without Metal). |

## Flow

1. `@custom_op("ns::name", mutates_args=[...])` validates at decoration
   time: at least one mutated argument, each a parameter of `fn`, no
   return annotation other than `None`, and every `overlappable` key mutated and
   its values non-mutated parameters.
2. Calling a `CustomOp` outside a trace raises (`tracer._TRACES` empty).
   Inside one, tensor arguments become operands (in argument order), other
   arguments are captured as constants. Tensors nested in
   lists/tuples/dicts are rejected.
3. A `run(tensors)` closure rebinds the arguments with real tensors and
   calls `fn`; it is registered via `lumen.graph._C._register_custom_op`
   (`graph/python.rs`), which returns an integer handle.
4. `prims._custom_call(operands, name, handle, mutated, overlappable)`
   records the `custom_call` node (plus a `fusion_output` per extra
   mutated operand); each mutated operand is then
   `copy_`-assigned its new value, so later reads in the trace see it.
5. At run time the plan (`graph_compiler/plan.rs`) calls
   `graph::custom::call(handle, args)`, which goes through the hook into
   Python with the GIL.

## `MetalKernel`

* `MetalKernel(source, name=None)` compiles via `lumen_mps_compile_kernel`
  in the shim. If `name` is omitted the source must contain exactly one
  kernel; the chosen name is exposed as `.name`. Each instance gets a
  unique pipeline key `"<lib>.custom_op.MetalKernel#<n>"`.
* `launch(tensors, args=(), *, grid=None, threadgroups=None)`: exactly one
  of `grid` (threads) or `threadgroups`; sizes padded to 3-D with 1s.
  Tensors bind as buffers `0..`, then each `args` entry (bytes, or any
  object with `.tobytes()`, e.g. a NumPy scalar) as by-value bytes. Every
  tensor must be on MPS. The launch goes through `kernels::mps::launch` with
  the kernel name as profiler label.

## Invariants & gotchas

* `mutates_args` must be accurate: a mutation not declared is not
  respected by the compiler.
* `overlappable={out: [inputs]}` lets the planner give mutated `out` the
  memory of a read input; only valid if the function reads each element
  of the input before writing `out` at that index and never reads `out`'s
  old value.
* Without Metal (`lumen_mps_linked` unset), `MetalKernel::new` errors
  ("lumen was built without MPS") and `launch` is a no-op.
* Custom op handles are never freed, so graphs holding them stay valid.

## Related

* [`lumen/graph`](../graph/README.md) — `CustomCall`, `prims._custom_call`, `custom.rs` hook, `_register_custom_op`.
* [`lumen/graph_compiler`](../graph_compiler/README.md) — `plan.rs` runs the custom-call step.
* [`lumen/kernels/mps`](../kernels/mps/README.md) — `launch` and the shim (`shim.mm`) that compiles kernels.
