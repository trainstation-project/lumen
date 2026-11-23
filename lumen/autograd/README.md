# autograd

Automatic differentiation, written in Python and run during tracing. The transforms read the nodes the tracer (`lumen/graph/tracing/tracer.py`) recorded while running a function and bind new primitives for the derivative into the same graph, so forward and backward reach the graph compiler as one program. They raise when called outside a trace.

## Files

- `__init__.py` exports the public API: `Function`, `grad`, `value_and_grad`, `jvp`, `vjp`, `linearize` and `backward_pass`. `lumen/__init__.py` re-exports `grad`, `jvp`, `value_and_grad` and `vjp`.
- `transforms.py` holds the transforms. `jvp` runs forward mode over the recorded tape. `linearize` builds the linear program at the primals; `vjp`, `grad`, `value_and_grad` and `TracedTensor.backward` (through `backward`) transpose that program in reverse order (`backward_pass`).
- `core.py` holds the rule registries `primitive_jvps` and `primitive_transposes`, the helpers `defjvp` and `deflinear` that fill them, and `UndefinedPrimal`, the stand-in for a linear operand during transposition.
- `rules.py` registers the JVP and transpose rule of every differentiable primitive on import (elementwise, casts, layout, slicing, gather and scatter, scans, reductions, `dot_general` and the collectives).
- `function.py` holds `Function` and `FunctionCtx`, for ops with a hand-written `forward`, `backward` and `jvp`. A `Function` call enters the tape as one node named `autograd.Function`.
- `attention.py`, `rms_norm.py`, `cross_entropy.py` and `dots.py` each hold a `substitute(tape)` pass that pattern-matches the recorded tape before linearizing:
  - `attention.py` replaces a written-out attention with one node whose backward is flash attention's (gated by `config.compiler.flash_attention`).
  - `rms_norm.py` replaces a written-out RMS norm with one node differentiated from each row's `1 / rms`.
  - `cross_entropy.py` replaces a cross entropy over a dot's logits with a linear cross entropy whose backward works a chunk of rows at a time (`config.compiler.fused_linear_cross_entropy_chunk_size`).
  - `dots.py` merges dots that share their lhs into one node differentiated as one dot (gated by `config.compiler.merge_dots`).

## How it connects

- Primitives come from `lumen.graph.tracing.prims`; tape nodes, scopes and tree helpers come from `lumen.graph.tracing.tracer`.
- The substitutes build their replacement nodes from `lumen.functional` (`_RMSNorm`, `_CrossEntropy`, the flash attention functions).
- The order of substitution is fixed in `_linearize_tape`: attention, then RMS norm, then cross entropy, then dots.

## Conventions

- `all_reduce` by max or min differentiates as `reduce_max` does: the tangents of the processes holding the result, averaged over ties.
- Rules follow JAX's design: a JVP rule per primitive, a transpose rule for each linear primitive, and reverse mode by transposing the linearized program.
- Primals, tangents, cotangents and outputs are trees walked by `tracer._tree_map`: tensors, modules, lists and tuples. Gradients, cotangents and tangents come back named as their primals or outputs; a named tangent or cotangent given is aligned to its value by name (`tracer._like`). Rules work on values rebuilt from vars (`core._value`), unnamed.
- Only float leaves get tangents; integer and bool values have none.
- Sharded values (placements) need nothing of the rules: the tape holds local values and explicit collectives, whose existing rules transpose them. The transforms translate at the edges (`lumen/graph/propagation/sharding.py`): a cotangent of a replicated output enters on one process of each group (the others' zero), so the copies' gradients add up to it once; a replicated leaf's gradient (partial over its processes) is all-reduced, a partial sum leaf's partitioned; a cotangent of a partial average is divided among the parts, of a partial max or min given to the parts equal to the whole (ties share), and such a leaf's parts' gradients are summed; gradients come back placed as their leaves. A `Function` with a sharded argument runs on replicated arguments, its results replicated.
- Tangents and cotangents must match their primal's dtype and shape exactly; the transforms raise `TypeError` otherwise.
- `rules.py` must be imported for its side effect before any transform runs; `transforms.py` does this.
- A new differentiable primitive needs its rules registered in `rules.py` with `defjvp`, `deflinear` or a direct entry in the registries.
