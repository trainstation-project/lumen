# `lumen/functional/attention`

Scaled dot-product attention in flash-attn's layout: `query [B, Sq, N, H]`,
`key [B, Sk, Nkv, H]`, `value [B, Sk, Nkv, Hv]` → `[B, Sq, N, Hv]`, with
grouped-query attention (`N` a multiple of `Nkv`), `scale` defaulting to
`1 / sqrt(H)`, a bottom-right-aligned causal mask (key `j` hidden from query
`i` when `j > i + Sk - Sq`), and `dropout_p` on the probabilities. Both
functions trace plain primitives; the MPS compiler matches them and runs the
forward as one flash-attention kernel (`lumen.config.compiler.flash_attention`).

## Files

| File | Contents |
| --- | --- |
| `__init__.py` | Re-exports `flash_attention`, `naive_attention`. |
| `flash.py` | `flash_attention` and `_FlashAttention(Function)`: the forward saves `q, k, v, o` and each row's log-sum-exp; the backward is FlashAttention-2's, recomputing probabilities on `S^T = K Q^T` and summing dK/dV over each GQA group. |
| `naive.py` | `naive_attention`: the same math written out (`softmax`, `dropout`, two `dot_general`s), no custom backward. |
| `utils.py` | `_expand_heads` (repeat KV heads for their query-head group) and `_scores` (`q k^T * scale`, causal-masked, in the accumulation dtype). |

## Two ways to get the flash backward

* `flash_attention` records one `autograd.Function` node, so its backward is
  `_FlashAttention.backward` directly.
* `naive_attention` (or attention written by hand) records primitives; when
  differentiated, `lumen/autograd/attention.py` finds it with the compiler's
  matcher and substitutes an equivalent FlashAttention-2 backward.

## Invariants

* Scores and softmax are in the accumulation dtype (`_accum_dtype`: float32
  or wider); probabilities are cast to the inputs' dtype before `p @ v`; the
  output is in the inputs' dtype.
* Backward dots accumulate and write in the accumulation dtype; gradients are
  cast back to `q`'s dtype once at the end.
* The dropout mask comes from `lumen.random` and is redrawn where needed,
  never saved; `dropout_p` must be in `[0, 1)` for `flash_attention`.

## Related

* [`lumen/autograd`](../../autograd/README.md) — `Function`, the attention substitution pass.
* `lumen/graph_compiler/attention.rs`, `lumen/ops/attention/mps/` — matcher and Metal kernels.
* `tests/test_attention.py`.
