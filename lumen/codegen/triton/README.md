# `lumen/codegen/triton`

Triton kernels generated from a loop fusion's body, as `codegen/mps`'s
loops are Metal: one `@triton.jit` function a fusion, each program a block
of `BLOCK` output elements, every value computed at the indices it is read
at (`Emitter`, the MPS emitter's index arithmetic, written in Python).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `fusible` (what a loop holds: elementwise ops, casts, select, reshape, broadcast_in_dim, transpose, slice, full and iota, on dtypes Triton has), `kernel` (a body's source and name, `fusion_<hash>`), `triton_type`. |
| `__init__.py` | `launch(name, source, inputs, scalars, out)`: compiles a kernel's source once (registered with `linecache`, as Triton reads a function's source) and launches it over `out`, runtime scalars passed by value. |

## Invariants & gotchas

* Exactly as traced: floats divide and take square roots correctly rounded
  (`tl.div_rn`, `tl.sqrt_rn`), multiply and add are never contracted into
  an FMA (`enable_fp_fusion=False`), `max` propagates NaN, and an integer
  divided by 0 is -1. `exp`, `log`, `tanh` and `logistic` are libdevice's;
  float16 and bfloat16 ones compute in float32 and round once.
* Constants are written as their bits, so every dtype's value is exact.
* Gathers, concatenates, random bits and reductions are not generated yet:
  such a step has no Triton kernel and errors.
