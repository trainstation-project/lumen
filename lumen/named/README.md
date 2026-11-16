# named

Named tensors, `lumen.named` (usually imported as `N`): a tensor whose axes have names, and ops that take
axis names instead of positions, in the style of haliax and xarray. Ops broadcast and align operands by name,
so the positional order of a tensor's axes stops mattering to the user. Names are resolved while tracing into
positional `lumen.functional` ops (or primitives) on the underlying tensor: the traced graph is the one the
positional program would give, so names cost nothing when the plan runs.

## Files

- `tensor.py` holds `NamedTensor`: a tensor plus a tuple of unique axis names, one per axis, in positional
  order. It has `.names`, `.shape` (a dict by name), `.dtype`, `.size(name)`, `.axis(name)`, `.tensor(*order)`
  (the positional tensor in an order of names, one `...` standing for the rest), `.align_to(*order)`, Python
  operators broadcasting by name, `.to(dtype)`, indexing with a dict (`x[{"seq": slice(0, 4)}]`), the
  reductions `sum`, `mean`, `max`, `min`, `split`, `merge`, `rename`, and `.backward()` and `.grad`. It
  registers the class with the tracer (`tracer.register_tree`).
- `ops.py` holds the ops:
  - elementwise binary ops and comparisons (`add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, `eq`,
    `ne`, `lt`, `le`, `gt`, `ge`, `logical_and`, `logical_or`, `where`, `clamp`), broadcast by name;
  - unary math (`exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `relu`, `sin`, `cos`, ...) and `cast`;
  - reductions over a name or names, which they drop (`sum`, `mean`, `max`, `min`, `logsumexp`, `any`, `all`),
    `argmax`/`argmin` along one name, and `softmax`/`log_softmax` along one name;
  - contraction: `dot(a, b, over)` (one `dot_general`; shared names not summed are batch axes) and
    `contract(*operands, keep=())` (an einsum by name: shared names are summed unless kept);
  - layout: `rename`, `align_to`, `split` (one axis into several, a reshape), `merge` (several into one),
    `index` (ints, unit-step slices, and gathers with a named tensor of indices), `concatenate`, `stack`;
  - `arange(axis, n)`, a named iota (causal masks: `arange("q", n) >= arange("k", n)`).
- `__init__.py` re-exports `NamedTensor` and every op. `lumen/__init__.py` exposes the package as
  `lumen.named`.

## How it connects

- `lumen/graph/tracer.py` walks `NamedTensor` as a container (`register_tree`, like JAX's pytree nodes): a
  compiled function takes one as a top-level argument (flattened into its tensor plus a static marker of its
  names, so the names are part of the signature) and returns them anywhere in its result; `lumen.grad`,
  `vjp` and `jvp` take and return them, their gradients named as their primals.
- `lumen.grad` accepts a `NamedTensor` of no axes as the scalar it differentiates.

## Conventions

- Names are strings, unique per tensor. Unknown names raise `KeyError`; repeated names and size mismatches
  `ValueError`; a positional tensor of rank 1 or more as an operand `TypeError` (0-d tensors and Python scalars
  broadcast as scalars).
- Broadcasting matches axes by name, never by position, and sizes must agree (no size-1 stretching). The
  result is stored in the order of the operand with the most axes, then the other operands' new names, so the
  widest operand is read without a transpose.
- `dot` and `contract` accumulate as `lumen.functional.matmul` would (float32 for narrower floats, float64
  and integers their own) and round to the operands' dtype, unless `accum_dtype`/`output_dtype` say otherwise;
  the result's axes are `dot_general`'s order (batch, then the left operand's, then the right's), so no
  transpose follows.
- Ops pick no layout of their own: a consumer that needs a particular positional order (a kernel reading the
  last axis contiguous) asks for it with `align_to` or `tensor(*order)`.
- Like every lumen op, named ops run only inside `lumen.compile`.
- `lumen.nn.Module` weights stay positional tensors; name them where they are used
  (`NamedTensor(self.w, ("embed", "mlp"))`).
