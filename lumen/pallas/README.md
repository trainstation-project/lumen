# `lumen/pallas`

lumen's Pallas: TPU kernels written as Pallas kernels are, with Pallas's
API, but traced by lumen and lowered to Mosaic (a TPU core's kernel
language, which libtpu compiles) by lumen. No JAX: this ports Pallas's
front end, its lowering rules (`jax/_src/pallas/mosaic/lowering.py`) and
its pipeline (`pipeline.py`) — JAX is Apache 2.0.

```python
from lumen import pallas as pl

def body(x, y, o, acc):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc[...] = 0.0
    acc[...] += pl.dot(x[...], y[...])
    ...

pl.emit_pipeline(body, grid=(mb, nb, kb),
                 in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)), ...],
                 out_specs=[...])(x_hbm, y_hbm, o_hbm)
```

A kernel's values are lumen's own traced tensors (`TracedTensor`): its
body records into a `Kernel` (Rust) through the same tracer `lumen.compile`
uses (`tracer.tracing`), so lumen's ops work in kernels as they do in
programs (`@`, `lumen.functional`, `.to(dtype)`, indexing, `lumen.zeros`),
each a graph `Primitive` whose types are inferred as the graph's. Its refs
are `Ref`s; the TPU's own ops (`bitcast`, shifts, `bitwise_and`/`or`,
`multiple_of`) are functions of this module.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The kernel IR: `Kernel` (parameters, scratch, a body `Block` of `Node`s), `Type` (`Value`, `Ref` of a `Space`, `Semaphore` array), `Op` (graph primitives, bit ops, ref `Slice`/`Reshape`/`RefBitcast`, `Load`, `Store`, `DmaStart`/`DmaWait`, `If`, `For`), and the builder that checks each node as it is added. |
| `mosaic.rs` | `lower(kernel, name)`: the kernel as a Mosaic module in libtpu's serialized form (generic ops named `stable_mosaic.*`, version 17). Pallas's lowering rules for each primitive (`arith`, `math`, `vector`, `tpu.matmul`, `scf`). |
| `python.rs` | `lumen._C.Kernel`, the bindings the Python API drives; it answers the tracer as a `Graph` does (`apply`, `type_of`, `_set_source`, `_set_scope`). |
| `__init__.py` | `lumen.pallas`: `kernel` (trace and lower a kernel on a device's memory), `Ref` (loads, stores, `.at` windows, `reshape`, `bitcast`), `ds`, `VMEM`, `SemaphoreType`, `when`, `fori_loop`, `make_async_copy` / `async_copy`, `bitcast`, `shift_left`, `shift_right_logical`, `bitwise_and`, `bitwise_or`, `multiple_of`, `dot`. |
| `pipeline.py` | `emit_pipeline`, `BlockSpec`, `program_id`, `num_programs`: Pallas's two-slot double-buffered pipeline over a grid (see the module doc for its schedule). |
| `_C.pyi` | Type stub for `Kernel`. |
| `tests.rs` | Lowering tests, without a TPU: a copy kernel, a loop accumulating a matmul in VMEM, bfloat16 word unpacking, exact constants, and what kernels refuse. |

## The IR

* **Values** are scalars (shape `[]`) or vectors; a graph `Primitive` makes
  one from others (`Kernel::apply`, its type from `Primitive::infer`).
* **Refs** are buffers: HBM (`Any`), `Vmem`, `Smem`. A window of one
  (`Slice`, static or i32 starts, dimensions squeezed) is a ref; VMEM refs
  load and store whole, SMEM refs an element at an index.
* **Blocks**: an `If` runs its block on a bool; a `For` runs its block for
  `i` in `lo..hi` carrying values (its block's arguments and results). A
  value made in a block is not visible after it.
* **Kernels** take, in order: an SMEM vector of int32 scalars, optionally
  host rows, the memory as read, the memory as written, then their scratch
  (VMEM buffers and semaphore arrays, `Kernel::alloc`, made as the body is
  traced). Their function has no grid: pipelines are loops.

## Lowering notes

* Every op generic, named `stable_mosaic.<dialect>.<op>`, properties as
  attributes; `scf.if` with an empty `else`; multi-result ops as `%t:k`.
* float32 dots multiply in full float32 (`contract_precision<fp32>`), other
  floats on the matrix unit's default; the result is cast to the
  primitive's `output_dtype` if it is not the accumulator's.
* Constants are exact: integers wrapped to their width, floats to 17
  digits (hex bits if not finite).
* Not lowered yet (they error): `iota`, `transpose`, `concatenate`,
  reductions to a scalar, vector-to-scalar reshapes, gathers and the rest;
  dots with batch dimensions.

## Related

* [`../ops/tpu/`](../ops/tpu/README.md) — how a kernel is wrapped, compiled and run on a device's memory.
* [`../ops/dot_general/`](../ops/dot_general/README.md) — the dot kernel (`tpu.py`), written on this.
* [`../graph/`](../graph/README.md) — the tracer (`tracer.tracing`) and `Primitive`s.
