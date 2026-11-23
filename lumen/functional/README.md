# functional

The public functional API, `lumen.functional` (usually imported as `F`): torch-style operations on traced tensors. Each function validates its arguments and binds primitives from `lumen.graph.tracing.prims` into the current trace; nothing here runs eagerly. The graph compiler later fuses and lowers what these functions emit.

## Files and folders

- `__init__.py` re-exports everything listed in `__all__`: the functions of `functions.py`, the attention variants and `cross_entropy`. `lumen/__init__.py` re-exports `cat` as `lumen.cat`.
- `functions.py` holds the operations:
  - elementwise arithmetic and comparisons with torch's broadcasting and dtype promotion (`add`, `mul`, `div`, `eq`, `where`, `maximum`, `clamp`, `pow`, ...), float math (`exp`, `log`, `sqrt`, `tanh`, `sigmoid`, the trigonometric functions, `erfinv`, ...) and logical ops;
  - reductions (`sum`, `mean`, `amax`, `amin`, `max`, `min`, `any`, `all`, `argmax`, `argmin`) and `cumsum`;
  - `softmax`, `log_softmax`, `rms_norm` (an `autograd.Function`, `_RMSNorm`, differentiated from each row's `1 / rms`), `dropout`;
  - `matmul` with torch's rules for 1-d operands, `einsum` of two operands;
  - layout and indexing (`cat`, `stack`, `triu`, `diagonal`, `pad`, `gather`).
- `attention/` holds the attention variants (see its README).
- `cross_entropy/` holds `cross_entropy` (see its README).

## How it connects

- Shared tracing helpers come from `lumen/graph/tracing/tracer.py`: `_lift` (Python scalars and tensors to traced values), `_common_dtype`, `_accum_dtype`, `_require_float`, `_dot`.
- `dropout` draws its mask from `lumen.random`'s counter-based generator (`_dropped`).
- `lumen/autograd` substitutes recorded attention, RMS norm and cross entropy patterns with the `Function`s defined here (`_RMSNorm`, `_CrossEntropy`, `_FlashAttention`).

## Conventions

- Functions follow torch's names and semantics unless the docstring of the function says otherwise; reductions and `cumsum` accumulate in a wider dtype where torch does.
- Ops are traced exactly as written, with no hidden casts beyond the documented promotion. Float matmuls accumulate in float32 (float64 stays float64; integers accumulate in their own dtype), per `tracer._accum_dtype`.
- Some ops are traced as other primitives (`gt` as `other < input`, `ge` as `other <= input`), so the graph has fewer primitive kinds.
- Every argument naming a dimension (`dim`, a reduction's dims, `cat`'s, `rms_norm`'s `normalized_shape`) is an `AxisType` (`lumen/tensor/names.py`): a position or an axis name of a named tensor, resolved by `names.axis`. Elementwise ops broadcast named operands by name (`tracer._broadcast`); `cat` aligns named tensors to the first's order, and `stack` with a name names its new, first axis.
- These functions only work inside `lumen.compile`.
