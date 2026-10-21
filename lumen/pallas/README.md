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
are `Ref`s; the TPU's own ops (`bitcast`, shifts, bitwise ops,
`multiple_of`) and `lax`'s that lumen has no name for (`select_n`,
`integer_pow`, `erf_inv`, control flow) are functions of this module.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The kernel IR: `Kernel` (parameters, scratch, a body `Block` of `Node`s), `Type` (`Value`, `Ref` of a `Space`, `Semaphore` array, tiled `View`), `Op` (graph primitives, bit ops, ref `Slice`/`Reshape`/`RefBitcast`, `Load`, `Store`, `StridedLoad`/`StridedStore`, `DmaStart`/`DmaWait`, `If` (with an `else` and values), `For`, `While`), and the builder that checks each node as it is added. |
| `mosaic.rs` | `lower(kernel, name)`: the kernel as a Mosaic module in libtpu's serialized form (generic ops named `stable_mosaic.*`, version 17). Pallas's lowering rules for each primitive (`arith`, `math`, `vector`, `tpu.matmul`, `scf`). |
| `python.rs` | `lumen.pallas._C.Kernel`, the bindings the Python API drives; it answers the tracer as a `Graph` does (`apply`, `type_of`, `_set_source`, `_set_scope`). |
| `__init__.py` | `lumen.pallas`: `kernel` (trace and lower a kernel on a device's memory), `Ref` (loads, stores, strided indexing, `.at` windows, `reshape`, `bitcast`), `ds`, `VMEM`, `SMEM`, `SemaphoreType`, control flow (`when`, `cond`, `switch`, `fori_loop`, `while_loop`, `scan` of a carry, `loop`), `make_async_copy` / `async_copy`, bit ops, `multiple_of`, `dot`, `dot_general`, `select_n`, `integer_pow`, `reduce_precision`, `erf_inv`, `population_count`, `clz`. |
| `call.py` | `pallas_call`: a kernel as an op of a lumen program, its operands staged as JAX stages them (whole in VMEM or SMEM, as HBM refs, or pipelined in blocks over a grid, named or not), `input_output_aliases`. |
| `pipeline.py` | `emit_pipeline`, `BlockSpec` (`Blocked`, `Element`, `None` dimensions), `program_id`, `num_programs`, `axis_index`, `axis_size`: Pallas's two-slot double-buffered pipeline over a grid (see the module doc for its schedule), with JAX's checks of block shapes and index maps. A block past its operand's end reads the operand viewed larger where that is its tiles' padding, else (at a static position) copies the operand's rows alone, the rest of the slot undefined. |
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
  attributes; `scf.if` with an empty `else` if it has none; multi-result
  ops as `%t:k`.
* As Pallas: bool selects, comparisons and reductions by logic or through
  int32 (Mosaic has no i1 vector ops of those); NaN checks as `!(x == x)`
  (no `uno`); casts between float8s through float32.
* float32 dots multiply in full float32 (`contract_precision<fp32>`), other
  floats on the matrix unit's default; the result is cast to the
  primitive's `output_dtype` if it is not the accumulator's.
* Constants are exact: integers wrapped to their width, floats to 17
  digits (hex bits if not finite).
* `take_along_axis` (`F.gather`) as `tpu.dynamic_gather`, as Pallas: along
  one of the last two dimensions, indices of the operand's shape, extended
  to its elements' width (Mosaic's).
* Not lowered yet (they error): `jnp.take`-like gathers and scatters, cumulative sums,
  dots of several contracting dimensions
  (libtpu 0.0.48's Mosaic aborts on them, under JAX too), `BoundedSlice`
  block dimensions (as under JAX on a TPU), `Element` padding below an
  operand.

## Related

* [`../ops/tpu/`](../ops/tpu/README.md) — how a kernel is wrapped, compiled and run on a device's memory.
* [`../ops/dot_general/`](../ops/dot_general/README.md) — the dot kernel (`tpu.py`), written on this.
* [`../graph/`](../graph/README.md) — the tracer (`tracer.tracing`) and `Primitive`s.
