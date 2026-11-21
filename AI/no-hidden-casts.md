---
name: no-hidden-casts
description: Use whenever writing or changing an op, a primitive's type rules, a graph compiler pass, codegen or a kernel on any device. Values keep the dtypes the user's program gives them; no code casts on its own.
---

# No casts the user didn't write

A program runs in exactly the dtypes it was traced with. A cast happens only where the user's program has one
(`prims.cast`, `.to(dtype=...)`, `F.matmul(..., accum_dtype, output_dtype)` and the like). Ops, passes, codegen
and kernels never add one: no rounding to a narrower dtype, no widening "for accuracy", no detour through
another dtype.

## Rules

1. **Ops compute in their operands' dtype.** A bfloat16 `exp` is a bfloat16 `exp`; an int32 max is an int32
   max. Don't widen to float32 (or float64) and round back, and don't route integers through floats.
2. **Accumulation is what the program names.** A dot or reduction accumulates in its `accum_dtype`
   (required on `dot_general` and `reduce_sum`), kept at every level of a split or hierarchical reduction.
   Widening rules go by size (narrower than float32), never by a list of dtypes.
3. **Ambiguous casts are errors, not choices.** When a cast has more than one reasonable result, the type
   check rejects it and says which explicit casts to write. A 32- or 64-bit integer to a float narrower
   than float32 is one: it could round once or through float32, so the user casts to float32 first
   (`Primitive::infer` for `Cast`).
4. **Passes don't change dtypes.** A rewrite may move, fold or fuse a cast the program has, but the values
   it produces are the traced program's, bit for bit. A pass never inserts a cast to make a kernel fit.
5. **Kernels take the dtypes they're given.** If a backend can't run an op in a dtype, it fails with a
   clear "does not run on <device> yet" error (or the backend's unsupported pass sends it elsewhere). It
   doesn't cast.
6. **A wider intermediate must be bit-exact.** Lowering an op or a cast the program wrote may use a wider
   type only when the result is bit for bit the op's single rounding in its own dtype: a bfloat16 add or
   multiply in float32 then rounded, an int16 to bfloat16 cast through an exact float32. A bfloat16 `exp`
   in float32 is not (it is more accurate than the program), so it is an error asking the user to convert.

## Exceptions the user chose

These are compiler options, not hidden casts. Don't add others without asking.

- `online_softmax` (on by default): a softmax's max and sum in one rescaled pass.
- `linear_reordering` and `preserve_precision` (off by default).
- The Triton GEMM launches with `enable_fp_fusion=True`, so its epilogue may fuse a multiply and an add into one FMA.

## When in doubt

If correct results seem to need a cast the user didn't write, ask the user or raise an error that names
the explicit casts to write. Don't add the cast silently.
