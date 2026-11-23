"""Moving traced values between placements: the collectives, slices and partitions taking a value's local shard
from one placement to another (``moved``), and the ``distribute``, ``redistribute`` and ``replicate`` the tracer and
autograd call."""

from __future__ import annotations

from lumen.graph.propagation import cost
from lumen.graph.tracing import prims, tracer
from lumen.tensor import placements as _placements
from lumen.tensor.placements import Partial, Replicate, Shard, Sharding, local_shape, replicated


def local(x):
    """Traced ``x``'s local value: unplaced and unnamed, with no op."""
    if x.placement is None and x.names is None:
        return x
    return tracer._view_as(x, None, None, x.graph.type_of(x.var)[1])


def placed(x, spec, shape):
    """Local traced ``x`` as the local shard of a value of global ``shape`` placed by ``spec``."""
    return tracer._view_as(x, None, spec, shape)


def _partition(v, op, first):
    """Local ``v``, a replicated value, as a part of it partial by ``op``: for a sum, ``v`` on the
    ``first`` process of the mesh dimension and zeros on the others (exact, unlike DTensor's ``v / n``),
    still read on every process so all of them differentiate it alike; ``v`` for the others."""
    if op != "sum":
        return v
    keep = prims.full(v.shape, bool(first), "bool")
    return prims.select(keep, v, prims.full(v.shape, False if v.dtype == "bool" else 0, v.dtype))


def reduced(v, op, ranks, scatter=None):
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


def moved(x, src: Sharding, dst: Sharding):
    """The local value of traced ``x`` (placed by ``src``, or a global value if unplaced) placed by ``dst``
    instead: the collectives, slices and partitions of ``cost.moves`` recorded on it. A dimension that doesn't
    split evenly is padded to whole blocks around a collective and the padding sliced off after."""
    from lumen import distributed as dist

    v = local(x)
    at = _placements.coordinate(src.mesh)
    current = list(src.placements)
    for i, a, b in cost.moves(src, dst):
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
            # Non-nested only (``cost.moves``): the target dimension is whole here.
            size = v.shape[b.dim]
            v = dist.all_to_all(_padded(v, b.dim, n * -(-size // n)), b.dim, a.dim, group=ranks)
            v = _narrowed(_narrowed(v, a.dim, 0, whole), b.dim, 0, _placements.chunk(size, n, at[i])[1])
        elif isinstance(a, Partial):
            if isinstance(b, Shard):
                size = v.shape[b.dim]
                v = _padded(v, b.dim, n * -(-size // n))
                v = _narrowed(reduced(v, a.reduce_op, ranks, b.dim), b.dim, 0, _placements.chunk(size, n, at[i])[1])
            else:
                v = reduced(v, a.reduce_op, ranks)
        elif isinstance(b, Shard):
            v = _narrowed(v, b.dim, *_placements.chunk(v.shape[b.dim], n, at[i]))
        else:
            v = _partition(v, b.reduce_op, at[i] == 0)
        current[i] = b
    return v


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
    out = placed(moved(x, x.placement, spec), spec, x.shape)
    out.names = x.names
    return out


def distribute(x, mesh, placements):
    """Traced ``x``, a global value every process holds, placed by ``placements`` over ``mesh``."""
    if x.placement is not None:
        raise TypeError("distribute: the tensor is sharded already: redistribute() it")
    spec = _placements.sharding(mesh, placements, x.ndim, x.names)
    local_shape(x.shape, spec)
    out = placed(moved(x, replicated(mesh), spec), spec, x.shape)
    out.names = x.names
    return out


def replicate(x):
    """Traced ``x``'s whole value as a local value: ``x`` gathered and reduced if it is sharded."""
    return x if x.placement is None else moved(x, x.placement, replicated(x.placement.mesh))
