# propagation

How a primitive's result metadata follows from its operands': its axis names and its sharding. The rules are Rust (one `match` over every primitive, so a new one must say how each flows through it); `sharding.py` applies them while tracing.

## Files

Rust (Python calls them as `_C.names` and `_C.strategies`):

- `mod.rs` declares the files and re-exports `Named`, `Op`, `Placement` and `Strategy`.
- `names.rs`: `Primitive::names`, a result's axis names (PyTorch's name inference: elementwise ops keep the names their named operands share, reductions drop the reduced ones, `transpose` permutes, `dot_general` keeps batch and free names, `gather` splices in the indices').
- `placement.rs`: `Placement` (DTensor's `Shard`, `Replicate`, `Partial` of an `Op`) and `Strategy` (the operands' placements on one mesh dim and the result's).
- `rules.rs`: strategy families many primitives share: elementwise ops by how far they are linear in their partial operands (`Linear`), reductions making a sharded reduced dim partial, the dim groups a reshape maps whole.
- `strategies.rs`: `Primitive::strategies`, each primitive's sharding strategies on one mesh dim (PyTorch DTensor's: elementwise ops on operands sharded alike and partial ones as far as they are linear, reductions, `dot_general`'s einsum strategies, `transpose`, `reshape` and `broadcast_in_dim` where a sharded dim maps whole, `slice`, `dynamic_slice` and `cumsum` of other dims, `gather`, `take_along_axis`, `scatter_add`, `dynamic_update_slice` and `concatenate` of other dims or of partial sums, and a `gather` along a sharded dim as a masked lookup giving a partial sum, DTensor's `MaskPartial`; none for factories, custom kernels and collectives, which run replicated).

Both `names` and `strategies` `match` every primitive, so a new one must say how each flows through it.

Python:

- `sharding.py` runs primitives on sharded operands (`propagate`, called by `bind`): the cheapest of the primitive's strategies (`_C.strategies`, the product over mesh dims), the operands moved to it, the primitive on the local shards (with its shape params made local, `_local_params`; a `gather` along a sharded dim masked, `_masked_gather`), its result placed. Choices use the largest block's shape, so every process makes the same ones; a reshape takes evenly split shards only. A primitive without strategies runs on replicated operands; a collective takes local values only. `mesh_of` gives the one mesh sharded operands share.
- `cost.py`: `moves`, the steps taking a placement to another (shards undone first, then partial values reduced, then replicated ones sliced or partitioned; a dim sharded over several mesh dims gathered innermost first and sliced outermost first), and `cost`, the bytes they move (DTensor's model, plus a fixed cost a collective).
- `redistribute.py`: `moved` records those steps on a value (all-gather, all-to-all, all-reduce, reduce-scatter from `lumen.distributed`, local slices, partitions; a dim that doesn't split evenly padded to whole blocks around a collective), and `distribute`, `redistribute` and `replicate`, which `TracedTensor` and autograd call.
- `gradients.py` translates gradients between placements and the local program autograd transposes (`cotangent`, `gradient`, `tangent`, `placed_like`), partial averages, maxes and mins included (`_weighted`).
- `__init__.py` marks the Python package.
