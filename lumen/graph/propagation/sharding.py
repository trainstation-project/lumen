"""Primitives on sharded traced tensors, PyTorch DTensor's sharding propagation at trace time. When an
operand of a primitive is placed over a device mesh (``TracedTensor.placement``), ``propagate`` lists the
primitive's strategies (``Primitive::strategies``, per mesh dimension), picks the one whose
redistribution of the operands moves the fewest bytes (DTensor's cost model), records the collectives that
move the operands' local shards to it (``redistribute``), records the primitive on the local shards and
places its result. The graph holds local values and explicit collectives only: autograd differentiates it
as any other program, whose gradients ``cotangent`` and ``gradient`` translate to and from placements."""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

from lumen.graph._C import Graph
from lumen.graph._C import strategies as _strategies
from lumen.graph.tracing import prims, tracer
from lumen.tensor import placements as _placements
from lumen.tensor.placements import Partial, Replicate, Shard, Sharding, local_shape, replicated

# Primitives that move data between processes: they take local values, never sharded ones.
COLLECTIVES = frozenset(
    "all_reduce all_gather reduce_scatter all_to_all broadcast reduce gather_to scatter_from send recv "
    "device_wait".split()
)

# A collective's fixed cost, in bytes moved: of two strategies moving as much, the one with fewer
# collectives wins.
LATENCY = 1024


def _local(x):
    """Traced ``x``'s local value: unplaced and unnamed, with no op."""
    if x.placement is None and x.names is None:
        return x
    return tracer._view_as(x, None, None, x.graph.type_of(x.var)[1])


def _placed(x, spec, shape):
    """Local traced ``x`` as the local shard of a value of global ``shape`` placed by ``spec``."""
    return tracer._view_as(x, None, spec, shape)


def _itemsize(dtype: str) -> int:
    """Bytes per element of ``dtype`` in memory (a sub-byte dtype takes a byte)."""
    if dtype == "bool":
        return 1
    return max(1, int("".join(c for c in dtype.split("_")[0] if c.isdigit())) // 8)


def _valid(spec: Sharding, shape: Sequence[int], even: bool = False) -> bool:
    """Whether a value of global ``shape`` can be placed by ``spec``: each sharded dimension one of its own,
    and, with ``even``, splitting evenly (a reshape's blocks map to blocks only then)."""
    dims = [p.dim for p in spec.placements if isinstance(p, Shard)]
    return all(d < len(shape) for d in dims) and (not even or _placements.even(shape, spec))


def _nested(spec: Sharding) -> bool:
    """Whether ``spec`` shards one tensor dimension over several mesh dimensions."""
    dims = [p.dim for p in spec.placements if isinstance(p, Shard)]
    return len(set(dims)) != len(dims)


def _steps(src: Sharding, dst: Sharding) -> list[tuple[int, Any, Any]]:
    """The moves taking a value placed by ``src`` to ``dst``, in order: ``(mesh dim, from, to)``, each
    changing one mesh dimension's placement. Shards are undone first (an all-to-all straight into a target
    dimension no mesh dimension shards, else an all-gather), then partial values reduced (a reduce-scatter
    into such a dimension, else an all-reduce), then replicated ones sliced or partitioned, so no tensor
    dimension is ever sharded twice."""
    if _nested(src) or _nested(dst):
        return _nested_steps(src, dst)
    current = list(src.placements)
    steps = []

    def free(p):
        """Whether ``p`` shards a tensor dimension no mesh dimension shards now."""
        return isinstance(p, Shard) and all(not (isinstance(c, Shard) and c.dim == p.dim) for c in current)

    # Each pass sees the placements the passes before it left: a value replicated by one is moved on by the last.
    for kind in (Shard, Partial, Replicate):
        for i, b in enumerate(dst.placements):
            a = current[i]
            if a == b or not isinstance(a, kind):
                continue
            to = b if kind is Replicate or free(b) else Replicate()
            steps.append((i, a, to))
            current[i] = to
    return steps


def _nested_steps(src: Sharding, dst: Sharding) -> list[tuple[int, Any, Any]]:
    """``_steps`` when a tensor dimension is sharded over several mesh dimensions in ``src`` or ``dst``: each
    tensor dimension whose sharding mesh dimensions change is gathered whole (the innermost split first), then
    partial values reduced, then replicated ones sliced (the outermost split first) or partitioned."""

    def sharded_by(placements, d):
        """The mesh dimensions sharding tensor dimension ``d``."""
        return [i for i, p in enumerate(placements) if p == Shard(d)]

    current = list(src.placements)
    dims = {p.dim for p in (*src.placements, *dst.placements) if isinstance(p, Shard)}
    moved = {d for d in dims if sharded_by(src.placements, d) != sharded_by(dst.placements, d)}
    steps = []
    for i in reversed(range(len(current))):
        a = current[i]
        if isinstance(a, Shard) and a.dim in moved:
            steps.append((i, a, Replicate()))
            current[i] = Replicate()
    for i, b in enumerate(dst.placements):
        a = current[i]
        if isinstance(a, Partial) and a != b:
            steps.append((i, a, Replicate()))
            current[i] = Replicate()
    for i, b in enumerate(dst.placements):
        if current[i] == Replicate() and b != Replicate():
            steps.append((i, Replicate(), b))
            current[i] = b
    return steps


def cost(src: Sharding, dst: Sharding, shape: Sequence[int], dtype: str) -> float:
    """The bytes a value of global ``shape`` and ``dtype`` placed by ``src`` moves per process to be
    placed by ``dst``, plus ``LATENCY`` a collective (DTensor's ``redistribute_cost``): an all-gather moves
    ``(n - 1) / n`` of its result, an all-reduce twice its operand's share, a reduce-scatter and an
    all-to-all its operand's share; slicing and partitioning move nothing."""
    size = math.prod(_placements.block_shape(shape, src)) * _itemsize(dtype)
    total = 0.0
    for i, a, b in _steps(src, dst):
        # A mesh dimension of one process moves nothing, but costs as one of more would: strategies are
        # chosen alike on any mesh.
        n = src.mesh.size(i)
        share = (n - 1) / n
        if isinstance(a, Shard) and isinstance(b, Replicate):
            size *= n
            total += LATENCY + size * share
        elif isinstance(a, Shard):
            total += LATENCY + size * share + 1
        elif isinstance(a, Partial) and isinstance(b, Replicate):
            total += LATENCY + 2 * size * share
        elif isinstance(a, Partial):
            total += LATENCY + size * share
            size /= n
        elif isinstance(b, Shard):
            size /= n
    return total


def _partition(v, op, first):
    """Local ``v``, a replicated value, as a part of it partial by ``op``: for a sum, ``v`` on the
    ``first`` process of the mesh dimension and zeros on the others (exact, unlike DTensor's ``v / n``),
    still read on every process so all of them differentiate it alike; ``v`` for the others."""
    if op != "sum":
        return v
    keep = prims.full(v.shape, bool(first), "bool")
    return prims.select(keep, v, prims.full(v.shape, False if v.dtype == "bool" else 0, v.dtype))


def _reduced(v, op, ranks, scatter=None):
    """Local ``v``, partial by ``op`` over ``ranks``, reduced: on every one of them, or, with ``scatter``
    (a tensor dimension), split along it, each keeping its block."""
    from lumen import distributed as dist

    reduce = "sum" if op == "avg" else op
    if scatter is None:
        out = dist.all_reduce(v, reduce, group=ranks)
    else:
        out = dist.reduce_scatter(v, scatter, reduce, group=ranks)
    if op == "avg" and len(ranks) > 1:
        out = prims.div(out, prims.full(out.shape, len(ranks), out.dtype))
    return out


def _padded(v, dim, size):
    """Local ``v`` padded with zeros at the end of ``dim`` to ``size``: written into zeros, no concatenate."""
    if v.shape[dim] == size:
        return v
    shape = list(v.shape)
    shape[dim] = size
    zeros = prims.full(tuple(shape), False if v.dtype == "bool" else 0, v.dtype)
    return prims.dynamic_update_slice(zeros, v, [0] * v.ndim)


def _narrowed(v, dim, start, size):
    """Local ``v``'s ``size`` entries along ``dim`` from ``start`` (``v`` itself when that is all of it)."""
    if (start, size) == (0, v.shape[dim]):
        return v
    starts, limits = [0] * v.ndim, list(v.shape)
    starts[dim], limits[dim] = start, start + size
    return prims.slice(v, starts, limits)


def _extent(shape, placements, mesh, d, i, at):
    """The size of tensor dimension ``d`` of a value of global ``shape`` placed by ``placements`` that this
    process's peers along mesh dimension ``i`` split between them: what the mesh dimensions before ``i``
    sharding it leave of it (the same on each of those peers)."""
    size = shape[d]
    for k in range(i):
        if placements[k] == Shard(d):
            size = _placements.chunk(size, mesh.size(k), at[k])[1]
    return size


def _moved(x, src: Sharding, dst: Sharding):
    """The local value of traced ``x`` (placed by ``src``, or a global value if unplaced) placed by ``dst``
    instead: the collectives, slices and partitions of ``_steps`` recorded on it. A dimension that doesn't
    split evenly is padded to whole blocks around a collective and the padding sliced off after."""
    from lumen import distributed as dist

    v = _local(x)
    at = _placements.coordinate(src.mesh)
    current = list(src.placements)
    for i, a, b in _steps(src, dst):
        ranks = _placements.group(src.mesh, i)
        n = len(ranks)
        if isinstance(a, Shard):
            # The whole of the dimension this mesh dimension splits, and its blocks' size.
            whole = _extent(x.shape, current, src.mesh, a.dim, i, at)
            block = -(-whole // n)
            v = _padded(v, a.dim, block)
        if isinstance(a, Shard) and isinstance(b, Replicate):
            v = _narrowed(dist.all_gather(v, a.dim, group=ranks), a.dim, 0, whole)
        elif isinstance(a, Shard):
            # Non-nested only (``_steps``): the target dimension is whole here.
            size = v.shape[b.dim]
            v = dist.all_to_all(_padded(v, b.dim, n * -(-size // n)), b.dim, a.dim, group=ranks)
            v = _narrowed(_narrowed(v, a.dim, 0, whole), b.dim, 0, _placements.chunk(size, n, at[i])[1])
        elif isinstance(a, Partial):
            if isinstance(b, Shard):
                size = v.shape[b.dim]
                v = _padded(v, b.dim, n * -(-size // n))
                v = _narrowed(_reduced(v, a.reduce_op, ranks, b.dim), b.dim, 0, _placements.chunk(size, n, at[i])[1])
            else:
                v = _reduced(v, a.reduce_op, ranks)
        elif isinstance(b, Shard):
            v = _narrowed(v, b.dim, *_placements.chunk(v.shape[b.dim], n, at[i]))
        else:
            v = _partition(v, b.reduce_op, at[i] == 0)
        current[i] = b
    return v


def _infer(name, operands, params):
    """The dtype and global shape of primitive ``name``'s result on the global types of ``operands``."""
    graph = Graph()
    var = graph.apply(name, [graph.input(x.dtype, list(x.shape)) for x in operands], params)
    dtype, shape = graph.type_of(var)
    return dtype, tuple(shape)


def _mesh(values):
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
        if not _valid(out, shape, even) or not all(_valid(s, x, even) for s, x in zip(ins, shapes)):
            continue
        total = sum(cost(a, b, x.shape, x.dtype) for a, b, x in zip(specs, ins, operands))
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
    mesh = _mesh(operands)
    specs = [x.placement or replicated(mesh) for x in operands]
    _, shape = _infer(name, operands, params)
    ins, out = _choose(name, specs, operands, params, shape)
    local = local_shape(shape, out)
    moved = [_moved(x, a, b) for x, a, b in zip(operands, specs, ins)]
    if name == "gather" and Shard(params["axis"]) in ins[0].placements:
        result = _masked_gather(*moved, operands[0].shape, ins[0], params["axis"], local)
    else:
        result = prims.bind(name, *moved, **_local_params(name, params, local))
    if result.shape != local:
        raise AssertionError(f"{name}: strategy {ins} -> {out} gave a local shape {result.shape}, not {local}")
    return _placed(result, out, shape)


def redistribute(x, mesh, placements):
    """Traced ``x`` (sharded) placed by ``placements`` over its mesh (``mesh``, if given, must be it)."""
    if x.placement is None:
        raise TypeError("redistribute: the tensor is not sharded: distribute() it")
    if mesh is not None and mesh != x.placement.mesh:
        raise ValueError(f"redistribute: the tensor is sharded over {x.placement.mesh.mesh}, not {mesh.mesh}")
    spec = _placements.sharding(x.placement.mesh, placements, x.ndim, x.names)
    local_shape(x.shape, spec)
    if spec == x.placement:
        return x
    out = _placed(_moved(x, x.placement, spec), spec, x.shape)
    out.names = x.names
    return out


def distribute(x, mesh, placements):
    """Traced ``x``, a global value every process holds, placed by ``placements`` over ``mesh``."""
    if x.placement is not None:
        raise TypeError("distribute: the tensor is sharded already: redistribute() it")
    spec = _placements.sharding(mesh, placements, x.ndim, x.names)
    local_shape(x.shape, spec)
    out = _placed(_moved(x, replicated(mesh), spec), spec, x.shape)
    out.names = x.names
    return out


def replicate(x):
    """Traced ``x``'s whole value as a local value: ``x`` gathered and reduced if it is sharded."""
    return x if x.placement is None else _moved(x, x.placement, replicated(x.placement.mesh))


def _dual(spec: Sharding, leaf: bool = False) -> Sharding:
    """How a gradient of a value placed by ``spec`` is laid out in the local program autograd transposes:
    a shard's as a shard, a replicated value's partial (every process differentiates its own copy), a
    partial sum's replicated. A partial average, max or min is replicated as a cotangent (``_weighted``
    scales each part's), and summed as a leaf's gradient (``leaf``): each part's gradient is its share of
    the whole's."""
    out = []
    for p in spec.placements:
        if isinstance(p, Partial):
            out.append(Partial() if leaf and p.reduce_op != "sum" else Replicate())
        else:
            out.append(Partial() if isinstance(p, Replicate) else p)
    return Sharding(spec.mesh, tuple(out))


def _weighted(c, out):
    """Local cotangent ``c`` of traced ``out``'s whole value, as the cotangent of ``out``'s local part: for
    a value partial by an average, ``c`` over the parts' count; by a max or min, ``c`` shared evenly by
    the parts equal to the whole (``all_reduce``'s rule); else ``c``."""
    spec = out.placement
    dims = [i for i, p in enumerate(spec.placements) if isinstance(p, Partial) and p.reduce_op != "sum"]
    if not dims:
        return c
    ops = {p.reduce_op for p in spec.placements if isinstance(p, Partial)}
    if len(ops) > 1:
        raise NotImplementedError(f"differentiating a value partial by several reductions {sorted(ops)}: reduce it")
    (op,) = ops
    if op == "avg":
        n = math.prod(spec.mesh.size(i) for i in dims)
        return prims.div(c, prims.full(c.shape, n, c.dtype))
    part = _local(out)
    whole = part
    for i in dims:
        whole = _reduced(whole, op, _placements.group(spec.mesh, i))
    where = prims.cast(prims.eq(part, whole), c.dtype)
    count = where
    for i in dims:
        count = _reduced(count, "sum", _placements.group(spec.mesh, i))
    return prims.div(prims.mul(c, where), count)


def cotangent(c, out):
    """The local value autograd transposes from, for cotangent ``c`` of traced ``out``: ``c`` (sharded,
    or a global value) laid out as ``out``'s gradient (``_dual``): a replicated output's on one process of
    each group, so the gradients of every process's copies add up to the gradient once; a partial
    average's, max's or min's weighted to each part (``_weighted``)."""
    if out.placement is None:
        if c.placement is not None:
            raise TypeError("a cotangent of a local value must be local")
        return c
    dual = _dual(out.placement)
    return _weighted(_moved(c, c.placement or replicated(out.placement.mesh), dual), out)


def gradient(g, primal):
    """Local ``g``, the gradient autograd gave for traced ``primal``'s local value, placed as ``primal``:
    a replicated primal's partial gradients reduced, a partial one's partitioned (a partial average's,
    max's or min's parts' gradients summed into the whole's, which every part takes)."""
    if primal.placement is None:
        return g
    dual = _dual(primal.placement, leaf=True)
    return _placed(_moved(_placed(g, dual, primal.shape), dual, primal.placement), primal.placement, primal.shape)


def tangent(t, primal):
    """The local value of tangent ``t`` (sharded, or a global value) of traced ``primal``, placed as it."""
    if primal.placement is None:
        if t.placement is not None:
            raise TypeError("a tangent of a local value must be local")
        return t
    return _moved(t, t.placement or replicated(primal.placement.mesh), primal.placement)


def placed_like(v, ref):
    """Local ``v`` placed as traced ``ref`` (a tangent of a value or zeros), as ``ref``'s shape says."""
    return v if ref.placement is None else _placed(v, ref.placement, ref.shape)
