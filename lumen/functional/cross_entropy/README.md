# `lumen/functional/cross_entropy`

`F.cross_entropy(input, target)`: PyTorch's
`cross_entropy(..., reduction="none")` over the last dimension — each row's
`logsumexp(input) - input[target]`, shaped like `input` without its last
dimension, in `input`'s dtype. `target` holds int32/int64 classes in `[0, n)`
and must have shape `input.shape[:-1]`.

## Files

| File | Contents |
| --- | --- |
| `__init__.py` | Re-exports `cross_entropy`. |
| `logits.py` | `cross_entropy` and `_CrossEntropy(Function)`: forward computes `lse = max + log(sum(exp(x - max)))` and reads each row's target logit with a `gather` of the flattened rows; backward is `(exp(x - lse) - one_hot(target)) * g`. |
| `utils.py` | `_one_hot(target, n, dtype)`: an `eq` against `iota`, cast to `dtype`. |

## How it is compiled and differentiated

* The MPS compiler runs the forward as one row kernel (online max and sum of
  exponentials), writing each row's log-sum-exp for the gradient.
* `_CrossEntropy` saves `x, target, lse`; the target gets no gradient.
* When `lumen.config.compiler.fused_linear_cross_entropy_chunk_size` is set
  and the logits come only from a 2-D float32 `h @ w'` (an LM head),
  `lumen/autograd/cross_entropy.py` replaces this node and the dot with a
  chunked fused linear cross entropy: logits are recomputed a chunk of rows
  at a time and never materialized whole. It matches on `_CrossEntropy`
  itself, so it only applies to logits fed through this function.

## Related

* [`lumen/autograd`](../../autograd/README.md) — `Function`, the chunked FLCE pass.
* [`lumen/functional`](../README.md).
