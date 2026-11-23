# named

Contraction by axis name, `lumen.named` (usually imported as `N`): `dot` and `contract` of named tensors, in
the style of haliax and xarray. Names are not a tensor type of their own: they are optional metadata on
`lumen.Tensor` and `TracedTensor` (`x.rename("batch", "seq", "embed")`), and every other op takes them where it
takes dimensions (`lumen.functional`, the tensor methods). Contraction has no positional op whose dimension
arguments a name could replace, so it lives here.

## Files

- `ops.py` holds the ops:
  - `dot(a, b, over)`: one `dot_general` summing the axes `over`; the other shared names are batch axes,
    matched by name, and each operand's own axes are kept (a linear layer by name: `N.dot(x, w, "embed")`);
  - `contract(*operands, keep=())`: an einsum by name, every shared name summed unless kept, as a chain of
    `dot_general`s left to right.
- `__init__.py` re-exports both. `lumen/__init__.py` exposes the package as `lumen.named`.

## How it connects

- The result's names come from `dot_general`'s rule in `lumen/graph/propagation/mod.rs` (batch, then the left
  operand's free axes, then the right's), as every op's do; the ops here only pick the dimension numbers.
- Sizes are checked by name with `tracer._union`, the same check elementwise broadcasting by name runs.

## Conventions

- Operands must be named; an unnamed one raises `TypeError`. Unknown names raise `KeyError`, size mismatches
  `ValueError`.
- `dot` and `contract` accumulate as `lumen.functional.matmul` would (float32 for narrower floats, float64
  and integers their own) and round to the operands' dtype, unless `accum_dtype`/`output_dtype` say otherwise;
  the result's axes are `dot_general`'s order, so no transpose follows.
- Like every lumen op, they run only inside `lumen.compile`.
