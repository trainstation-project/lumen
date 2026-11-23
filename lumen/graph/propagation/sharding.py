"""Primitives on sharded traced tensors, PyTorch DTensor's sharding propagation at trace time. When an
operand of a primitive is placed over a device mesh (``TracedTensor.placement``), ``propagate`` lists the
primitive's strategies (``Primitive::strategies``, per mesh dimension), picks the one whose
redistribution of the operands moves the fewest bytes (DTensor's cost model), records the collectives that
move the operands' local shards to it (``redistribute``), records the primitive on the local shards and
places its result. The graph holds local values and explicit collectives only: autograd differentiates it
as any other program, whose gradients ``cotangent`` and ``gradient`` translate to and from placements."""

import itertools

from lumen.graph._C import Graph
from lumen.graph._C import strategies as _strategies
from lumen.graph.propagation import cost
from lumen.graph.propagation.redistribute import moved, placed
from lumen.graph.tracing import prims
from lumen.tensor import placements as _placements
from lumen.tensor.placements import Partial, Replicate, Shard, Sharding, local_shape, replicated

# Primitives that move data between processes: they take local values, never sharded ones.
COLLECTIVES = frozenset(
    "all_reduce all_gather reduce_scatter all_to_all broadcast reduce gather_to scatter_from send recv "
    "device_wait".split()
)


def _infer(name, operands, params):
    """The dtype and global shape of primitive ``name``'s result on the global types of ``operands``."""
    graph = Graph()
    var = graph.apply(name, [graph.input(x.dtype, list(x.shape)) for x in operands], params)
    dtype, shape = graph.type_of(var)
    return dtype, tuple(shape)


def mesh_of(values):
    """The one mesh the sharded ones of traced ``values`` are placed over; a ValueError if they differ."""
    meshes = list(dict.fromkeys(x.placement.mesh for x in values if x.placement is not None))
    if len(meshes) > 1:
        raise ValueError(f"operands are sharded over different meshes: {[m.mesh for m in meshes]}")
    return meshes[0]


def _encode(p):
    """Placement ``p`` as ``_C.strategies`` takes it: ``(kind, dim, op)``."""
    if isinstance(p, Shard):
        return ("shard", p.dim, "")
    if isinstance(p, Partial):
        return ("partial", 0, p.reduce_op)
    return ("replicate", 0, "")


def _decode(kind, dim, op):
    """The placement ``_C.strategies`` gives as ``(kind, dim, op)``."""
    return Shard(dim) if kind == "shard" else Partial(op) if kind == "partial" else Replicate()


def _local_params(name, params, shape):
    """``params`` (global, as ``prims.bind`` gets them) for primitive ``name`` on local shards giving a result
    of local ``shape``: as they are, but for a param that is a shape (a sharded axis of a slice starts at 0)."""
    if name == "slice":
        return {**params, "limit_indices": tuple(s + size for s, size in zip(params["start_indices"], shape))}
    if name == "reshape":
        return {**params, "new_sizes": tuple(shape)}
    if name == "broadcast_in_dim":
        return {**params, "shape": tuple(shape)}
    if name == "dynamic_slice":
        return {**params, "slice_sizes": tuple(shape)}
    return params


def _choose(name, specs, operands, params, shape):
    """The operands' and the result's placements of the cheapest strategy of primitive ``name`` for operands placed
    by ``specs``, the result of global ``shape``: the product over mesh dimensions of each one's
    strategies (all-replicated first: the first of equally cheap ones wins), those the shapes can take."""
    mesh = specs[0].mesh
    shapes = [x.shape for x in operands]
    k = len(operands)
    per_dim = []
    for i in range(mesh.ndim):
        current = [s.placements[i] for s in specs]
        listed = _strategies(name, params, [_encode(p) for p in current], shapes, shape)
        listed = [(tuple(_decode(*p) for p in ins), _decode(*out)) for ins, out in listed]
        options = [((Replicate(),) * k, Replicate()), *listed]
        per_dim.append(list(dict.fromkeys(options)))
    best = None
    for combo in itertools.product(*per_dim):
        ins = [Sharding(mesh, tuple(c[0][j] for c in combo)) for j in range(k)]
        out = Sharding(mesh, tuple(c[1] for c in combo))
        even = name == "reshape"
        if not cost.valid(out, shape, even) or not all(cost.valid(s, x, even) for s, x in zip(ins, shapes)):
            continue
        total = sum(cost.cost(a, b, x.shape, x.dtype) for a, b, x in zip(specs, ins, operands))
        if best is None or total < best[0]:
            best = (total, ins, out)
    return best[1], best[2]


def _masked_gather(x, ids, shape, spec, axis, local):
    """The gather along ``axis`` of local ``x``, the block of a value of global ``shape`` placed by ``spec``
    (sharded along ``axis``), at replicated indices ``ids``: each index looked up in this process's block if
    it falls there, zeros where it doesn't, so the processes' results sum to the whole gather (DTensor's
    ``MaskPartial``; Megatron's vocab-parallel embedding). Its result has the ``local`` shape."""
    if x.shape[axis] == 0:
        return prims.full(local, False if x.dtype == "bool" else 0, x.dtype)
    start = prims.full(ids.shape, _placements.offsets(shape, spec)[axis], ids.dtype)
    shifted = prims.sub(ids, start)
    below = prims.lt(shifted, prims.full(ids.shape, 0, ids.dtype))
    above = prims.lt(prims.full(ids.shape, x.shape[axis] - 1, ids.dtype), shifted)
    outside = prims.or_(below, above)
    inside = prims.select(outside, prims.full(ids.shape, 0, ids.dtype), shifted)
    gathered = prims.gather(x, inside, axis)
    mask = prims.broadcast_in_dim(outside, gathered.shape, tuple(range(axis, axis + ids.ndim)))
    return prims.select(mask, prims.full(gathered.shape, False if x.dtype == "bool" else 0, x.dtype), gathered)


def propagate(name, operands, params):
    """Primitive ``name`` on traced ``operands`` (some sharded, the others global values every process
    holds, taken as replicated) with ``params`` (in global terms, as ``prims.bind`` gets them): its
    cheapest strategy's collectives and the primitive on the local shards recorded; its result placed.
    A primitive without strategies runs on replicated operands."""
    if name in COLLECTIVES:
        raise TypeError(f"{name}: a collective takes local tensors: call to_local() on sharded operands")
    mesh = mesh_of(operands)
    specs = [x.placement or replicated(mesh) for x in operands]
    _, shape = _infer(name, operands, params)
    ins, out = _choose(name, specs, operands, params, shape)
    local = local_shape(shape, out)
    moved = [moved(x, a, b) for x, a, b in zip(operands, specs, ins)]
    if name == "gather" and Shard(params["axis"]) in ins[0].placements:
        result = _masked_gather(*moved, operands[0].shape, ins[0], params["axis"], local)
    else:
        result = prims.bind(name, *moved, **_local_params(name, params, local))
    if result.shape != local:
        raise AssertionError(f"{name}: strategy {ins} -> {out} gave a local shape {result.shape}, not {local}")
    return placed(result, out, shape)
