# `lumen/codegen/triton`

Triton kernels generated from a loop fusion's body, as `codegen/mps`'s
loops are Metal: one `@triton.jit` function a fusion, each program a block
of `BLOCK` output elements, every value computed at the indices it is read
at (`Emitter`, the MPS emitter's index arithmetic, written in Python).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `fusible` (what a loop holds: elementwise ops, casts, select, reshape, broadcast_in_dim, transpose, slice, full and iota, on dtypes Triton has); `kernel` (a body's source and name, `fusion_<hash>`), which reads input `k` through a `LOAD{k}(ptr, i, mask)` prologue and writes through a `STORE(ptr, i, value, mask)` epilogue, both constexpr arguments, so its source never depends on a layout; `load` and `store` (those functions for a tensor's shape and strides: `ptr + i` when contiguous, `i` through its strides otherwise); `fill` (a CUDA tensor's `fill_`: the value's bits a runtime argument, so one kernel a dtype, written through `STORE`); `bits`, `triton_type`. |
| `gemm.rs` | `gemm`: the GEMM's sizes (`L`, `M`, `K`, `N`) and its prologues and epilogue for its operands' and output's layouts. `PROLOGUE_A(ptr, l, m, k, mask)` and `PROLOGUE_B(ptr, l, k, n, mask)` load a tile at its batch, row and column indices through the operand's batch, free and contracting dimensions and strides; `EPILOGUE(ptr, l, m, n, value, mask)` rounds a tile to the output's dtype and stores it. Any strides: dimensions need not merge. |
| `__init__.py` | `compiled` (a source compiled once, registered with `linecache`, as Triton reads a function's source), `functions` (prologues and epilogues by parameter) and `launch(name, source, inputs, scalars, out, functions)`: a loop over `out`, runtime scalars passed by value. |

## Invariants & gotchas

* Exactly as traced: floats divide and take square roots correctly rounded
  (`tl.div_rn`, `tl.sqrt_rn`), multiply and add are never contracted into
  an FMA (`enable_fp_fusion=False`), `max` propagates NaN, and an integer
  divided by 0 is -1. `exp`, `log`, `tanh` and `logistic` are libdevice's;
  float16 and bfloat16 ones compute in float32 and round once.
* Constants are written as their bits, so every dtype's value is exact.
* Gathers, concatenates, random bits and reductions are not generated yet:
  such a step has no Triton kernel and errors.
