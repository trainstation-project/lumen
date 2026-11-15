# functional/attention

Attention variants of `lumen.functional`. All three are traced as ordinary primitives (dots, softmax, masks); the graph compiler recognizes the pattern and runs it as one flash-attention kernel where the device supports it, gated by the compiler's `flash_attention` setting.

## Files

- `__init__.py` exports `flash_attention`, `naive_attention` and `attention_decode`.
- `flash.py` holds `flash_attention` and `_FlashAttention`, an `autograd.Function`. The forward also produces each query row's log-sum-exp; the backward is FlashAttention-2's, recomputing probabilities from that log-sum-exp instead of saving them. With dropout, the mask is drawn again where the backward needs it, never stored.
- `naive.py` holds `naive_attention`: the same attention written out as plain primitives, with autograd going through the generic rules (and `lumen/autograd/attention.py`'s substitution).
- `decode.py` holds `attention_decode`: new query tokens against a KV cache, query row `i` seeing keys `0..=start + i`, where `start` is an int or a 0-d integer tensor.
- `utils.py` holds `_expand_heads` (repeat each key/value head for its group of query heads) and `_scores` (`q k^T * scale` in the accumulation dtype, with the optional causal mask).

## Conventions

- Layout is flash-attn's: `query` `[B, Sq, N, H]`, `key` `[B, Sk, Nkv, H]`, `value` `[B, Sk, Nkv, Hv]`, result `[B, Sq, N, Hv]`. `N` must be a multiple of `Nkv` (grouped-query attention); gradients of query heads sharing a key/value head are summed back.
- `scale` defaults to `1 / sqrt(H)`.
- `is_causal` aligns the mask to the bottom right: key `j` is masked for query `i` when `j > i + Sk - Sq`, so with a KV cache each new query sees every earlier key.
- Scores and the softmax are computed in float32 or wider (`_accum_dtype`); the probabilities are cast back to the inputs' dtype before the dot with `value`.
- Inputs must be float and share one dtype; shapes are checked up front with a `ValueError`.
- `lumen/autograd/attention.py` imports these functions to build its replacement nodes, so a signature change here must be reflected there.
