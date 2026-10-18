# `lumen/ops/fill`

The eager `fill_` op (PyTorch: `aten::fill_`): set every element of a tensor
(contiguous or strided view) to a scalar, with a kernel per device. Called by
`Tensor::fill_`, `Tensor::zeros` and `Tensor::ones` in
`lumen/tensor/tensor.rs` via `fill_op`. This is a device op on an existing
tensor, not a graph primitive; graph plans' `full` lives in
[`lumen/ops/factory`](../factory/).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `FillKernel`, the `FILL` op and its static registry, `FILL_PY` (python feature), `fill_op`, `fill_layout`. |
| `cpu.rs` | CPU kernel: waits on the tensor, then writes contiguously or via `for_each_index` + `flat_offset`. |
| `cuda.py` | CUDA kernel in the CuTe DSL (`_FillCUDAKernel`), registered from Python as `<LIBRARY_NAME>::fill_` for `"cuda"`. |

## Subfolders

[`mps/`](mps/) — the Metal fill kernels, their Objective-C++ shim and the Rust wrapper.

## Dispatch

`fill_op(t, value)`:

1. Returns at once if `t.numel() == 0`.
2. With the `python` feature, if a Python kernel is registered for the
   device (`crate::ops::python::handle_for`), launches it with the layout
   from `fill_layout` and returns if it ran.
3. Otherwise `FILL.dispatch(t.device())`: `Cpu` -> `cpu::fill`, `Mps` ->
   `mps::fill` (when linked), `Meta` -> no-op, `Cuda` -> no Rust kernel
   (CUDA is served only by `cuda.py`).

The Rust kernels convert the `Scalar` with `T::from_scalar` under
`dispatch_dtype!`.

`fill_layout(shape, strides)` canonicalises a view for Python kernels: drop
size-1 dims, sort by stride ascending, merge contiguous neighbours. A
contiguous tensor becomes `([numel], [1])`; a 0-d tensor `([1], [1])`.

## CUDA kernel (`cuda.py`)

Persistent grid: `BLOCK_SIZE = 256`, blocks capped at
`SMs * threads per SM / 256`; each thread stores a register vector of
`vector_size` elements and strides over the vectors. Compiled kernels are
cached by `(dtype, shape, strides, vector_size)`. `bool` is stored as `Uint8`.

## Related

- [`lumen/ops/python.rs`](../python.rs) — Python kernel registry (`FILL_PY`).
- [`lumen/ops/factory`](../factory/) — the graph primitive `full`.
- `lumen/stream/mps` — the stream the MPS fill is encoded into.
