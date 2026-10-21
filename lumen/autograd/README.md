# `lumen/autograd`

Automatic differentiation, after JAX's `jax/_src/interpreters/ad.py`, with a
PyTorch-style `Function` for custom derivatives. It is pure Python and works
only on *traced* programs: lumen always runs inside `lumen.compile`, so every
transformation here runs while a function is being traced, reads the tape of
primitives that tracing recorded, and records the derivative as more
primitives into the same `Graph`. There is no eager autograd and no runtime
tape; the gradient is part of the compiled program.

Entry points: `lumen.grad`, `lumen.value_and_grad`, `lumen.vjp`, `lumen.jvp`
(re-exported from `lumen/__init__.py`), `lumen.autograd.linearize`,
`lumen.autograd.backward_pass`, and `TracedTensor.backward()`
(`lumen/graph/tracer.py`), which calls `transforms.backward` and accumulates
into each argument's / weight's `.grad`.

## Files

| File | Contents |
| --- | --- |
| `__init__.py` | Re-exports `Function`, `backward_pass`, `grad`, `jvp`, `linearize`, `value_and_grad`, `vjp`. |
| `core.py` | The rule registries `primitive_jvps` / `primitive_transposes`, `UndefinedPrimal`, `defjvp` / `deflinear`, and small helpers (`_full`, `_sum`, `_value`, `_accumulate`). |
| `rules.py` | One JVP rule per differentiable primitive and a transpose rule per linear one (JAX's `lax.py`): add/sub/neg, cast, reshape, transpose, reductions, cumsum, broadcast, slice, concatenate, select, (dynamic) slices/updates, gather/scatter_add, mul/div, max, exp/log/sqrt/tanh/logistic, `dot_general`, wait and host/device transfers. |
| `transforms.py` | The transformations: `jvp`, `linearize`, `backward` (used by `.backward()`), `backward_pass`, `vjp`, `value_and_grad`, `grad`. |
| `function.py` | `Function` / `FunctionCtx` (`torch.autograd.Function`, JAX `custom_vjp`), and how a Function node is JVP'd and transposed. |
| `attention.py` | Pre-autodiff pass: each attention the compiler's matcher finds becomes one `_Attention` node with a FlashAttention-2 backward. |
| `cross_entropy.py` | Pre-autodiff pass: a cross entropy of a 2-D float32 dot's logits becomes one `_LinearCrossEntropy` node with a chunked backward (fused linear cross entropy). |
| `dots.py` | Pre-autodiff pass: dots sharing an lhs over distinct weights become one `_MergedDot` node, differentiated as the single concatenated dot. |

## The tape

`lumen/graph/tracer.py` keeps a stack `_TAPES`; every primitive bound while
tracing is appended to each open tape as `(name, inputs, params, out_var)`.
`transforms._record(fn, *args)` opens a fresh tape around `fn`, and
`lumen.compile` opens one for the whole trace (also kept in `_BACKWARD` for
`.backward()`). A `Function` call appears as a single entry
`("autograd.Function", input_vars, {"function", "ctx", "entries", "nargs"}, out_vars)`:
`Function.apply` hides the tapes while `forward` runs, so its inner
primitives are in the graph but not on the tape.

## Data flow

1. **Forward mode** (`jvp`): walk the tape in order, calling each primitive's
   JVP rule (`_jvp_node`), or a Function's own `jvp`. Tangents are `None` for
   zero; non-float values get none.
2. **Linearize** (`_linearize_tape`): first the substitution passes, in this
   order — `attention.substitute`, then `cross_entropy.substitute`, then
   `dots.substitute`. Then JVP the tape again with placeholder seeds
   (`_full(p, 0)`), recording into a new tape; a Function becomes one linear
   node with placeholder outputs. Partial evaluation keeps the nodes that
   depend on a seed: the *linear* program.
3. **Transpose** (`backward_pass`): walk the linear nodes in reverse, applying
   each primitive's transpose rule (linear operands passed as
   `UndefinedPrimal`), or a Function's `backward`; cotangents for one value
   are summed with `_accumulate`.
4. `vjp` = linearize + a pullback over `backward_pass`; `value_and_grad` /
   `grad` call `vjp` with a cotangent of 1 and require a float scalar output.

Derivative nodes are recorded in the scopes of their forward node
(`tracer._backward_scope`, `tracer._SCOPE_OF`) so profiler/graph ranges read
e.g. `attention (backward)`.

## Custom derivatives: `Function`

```python
class Square(lumen.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * x * 2
```

* `apply` binds arguments to `forward`'s signature (defaults applied);
  `forward` must return a traced tensor or a tuple of them.
* `ctx.save_for_backward(*tensors, **named)` sets `saved_tensors` and
  `saved_named_tensors`; any other attribute can be stashed on `ctx`.
* `backward(ctx, *grads)` returns the positional arguments' gradients (a
  tuple), a dict by name (keyword-only arguments too), or positional ones
  followed by a dict; `None` or a missing name means no gradient.
* `jvp(ctx, *tangents, **kw_tangents)` is only needed for forward mode.
* Every returned gradient/tangent is type-checked (`_check_type`): it must
  have exactly its value's dtype and shape.

Users in the tree: `_FlashAttention` (`lumen/functional/attention/flash.py`),
`_CrossEntropy` (`lumen/functional/cross_entropy/logits.py`), and the three
substitution nodes here.

## The substitution passes

Each rewrites tape entries into a `Function` node whose `forward` is never
called (the forward is already in the graph); only its `backward` matters.

* **`attention.py`** — uses `graph.attentions()` (the Rust matcher in
  `lumen/graph_compiler/common/attention.rs`) to find attentions however written,
  replaces their member nodes with one `_Attention` node and adds the
  log-sum-exp (`max + log(sum)`) to the graph. The backward recomputes the
  probabilities on `S^T = K Q^T`, handling the matched scale/rounding chain,
  causal offset and dropout mask; the MPS compiler runs it as two kernels
  (dK/dV, dQ). Off when `lumen.config.compiler.flash_attention` is false.
* **`cross_entropy.py`** — when
  `lumen.config.compiler.fused_linear_cross_entropy_chunk_size` is set
  (default `None`: off), finds a `_CrossEntropy` node whose logits come only
  from a 2-D `dot_general` accumulated in and written as float32, drops the
  dot, and adds `_LinearCrossEntropy`, whose backward recomputes logits
  `chunk` rows at a time from the saved log-sum-exp; `dw` is summed over
  chunks in float32 and rounded once.
* **`dots.py`** — when `lumen.config.compiler.merge_dots` is on, groups
  `dot_general`s with the same lhs, params and weight type (except the one
  free dimension) over distinct non-weak leaves, and differentiates them as
  `x @ [w1 | w2 | ...]`: one dot for `dx`, one for the weights, sliced back.

## Invariants & gotchas

* **Exact dtypes**: no rule widens or narrows silently. Tangents/cotangents
  have their primal's dtype; dots pass explicit `accum_dtype` and
  `output_dtype`. `_cast_transpose` is the one place that avoids a rounding:
  if a cotangent is a narrow cast of dots accumulating in the wider dtype
  (`_widened`), it re-emits those dots writing the accumulator dtype instead.
* Everything raises outside a trace (`_check`: "call it inside a function
  passed to lumen.compile").
* A primitive without a rule raises `NotImplementedError`; `custom_call`
  (custom ops) is never differentiable.
* Circular imports: `attention.py`/`cross_entropy.py` import
  `lumen.functional` lazily inside `backward`.

## Related

* [`lumen/graph`](../graph/) — `tracer.py` (tapes, `TracedTensor.backward`) and `prims.py`.
* [`lumen/functional`](../functional/README.md) — ops that define `Function`s.
* [`lumen/graph_compiler`](../graph_compiler/) — attention matcher, fusion of the backward kernels.
* `tests/test_autograd.py`, `tests/test_attention.py`.
