# `lumen/functional`

The ops as functions: `import lumen.functional as F`, named after PyTorch's
`torch` / `torch.nn.functional`. Every function records primitives
(`lumen.graph.prims`) into the graph being traced by `lumen.compile`, so it
takes traced tensors and Python scalars; nothing here runs eagerly. A traced
tensor itself only has operators (`+ - * / == < []`, `-x`; `@` raises: a
matmul names its dtypes), layout methods
and dtype casts (`lumen/graph/tracer.py`); everything else lives here.
Patterns like softmax, RMS norm, cross entropy and attention are traced as
plain primitives and recognized by the MPS graph compiler, which fuses them
into its own kernels.

## Files

| File | Contents |
| --- | --- |
| `__init__.py` | The public `F` namespace: re-exports `functions.*`, `flash_attention`, `naive_attention`, `cross_entropy`. |
| `functions.py` | Arithmetic/comparison (`add` … `where`), elementwise (`exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `relu`), `dropout`, reductions (`sum`, `mean`, `amax`, `max`), `cumsum`, `softmax`, `log_softmax`, `rms_norm`, `matmul`. |

## Subfolders

* [`attention/`](attention/README.md) — `flash_attention` (a `Function` with FlashAttention-2 backward) and `naive_attention` (written out).
* [`cross_entropy/`](cross_entropy/README.md) — `cross_entropy` (a `Function` differentiated from each row's log-sum-exp).

## Invariants & gotchas

* **Exact dtypes**: operands share a dtype (a Python scalar takes its tensor
  operand's and must be of its kind); float functions require float tensors;
  `sum`/`mean` reduce in the input's dtype. Convert explicitly
  (`F.sum(x.float())`).
* `matmul(input, other, accum_dtype, output_dtype)` and
  `einsum(equation, x, y, accum_dtype=..., output_dtype=...)` take both
  dtypes from the caller: nothing infers them (`@` on traced tensors
  raises). A `[..., K] · [K, N]` matmul folds the batch into rows rather
  than broadcasting the weight.
* `cumsum` accumulates in its result dtype; a float16/bfloat16 input with
  `dtype="float32"` is widened by the kernel, any other dtype change casts
  first.
* `rms_norm` normalizes the last dimension only, in the accumulation dtype
  (float32 for narrower floats), then casts back and multiplies by `weight`;
  `eps` defaults to the dtype's machine epsilon.
* `dropout` draws its mask from `lumen.random` (`_dropped`): random bits
  below `p * 2**32` are dropped; the mask is recomputed where applied, never
  stored.

## Related

* [`lumen/graph`](../graph/) — `prims.py` and the tracer these build on.
* [`lumen/autograd`](../autograd/README.md) — `Function`, and the passes that rewrite attention / cross entropy before autodiff.
* `lumen/graph_compiler/mps/` — the fusions (`diamonds.rs` for softmax/norm rows) that run these patterns.
