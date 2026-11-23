"""Gradients of sharded values: translating between a value's placement and how its gradient is laid out in the
local program autograd transposes (``cotangent``, ``gradient``, ``tangent``, ``placed_like``)."""

from __future__ import annotations

import math

from lumen.graph.propagation.redistribute import local, moved, placed, reduced
from lumen.graph.tracing import prims
from lumen.tensor import placements as _placements
from lumen.tensor.placements import Partial, Replicate, Sharding, replicated


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
    part = local(out)
    whole = part
    for i in dims:
        whole = reduced(whole, op, _placements.group(spec.mesh, i))
    where = prims.cast(prims.eq(part, whole), c.dtype)
    count = where
    for i in dims:
        count = reduced(count, "sum", _placements.group(spec.mesh, i))
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
    return _weighted(moved(c, c.placement or replicated(out.placement.mesh), dual), out)


def gradient(g, primal):
    """Local ``g``, the gradient autograd gave for traced ``primal``'s local value, placed as ``primal``:
    a replicated primal's partial gradients reduced, a partial one's partitioned (a partial average's,
    max's or min's parts' gradients summed into the whole's, which every part takes)."""
    if primal.placement is None:
        return g
    dual = _dual(primal.placement, leaf=True)
    return placed(moved(placed(g, dual, primal.shape), dual, primal.placement), primal.placement, primal.shape)


def tangent(t, primal):
    """The local value of tangent ``t`` (sharded, or a global value) of traced ``primal``, placed as it."""
    if primal.placement is None:
        if t.placement is not None:
            raise TypeError("a tangent of a local value must be local")
        return t
    return moved(t, t.placement or replicated(primal.placement.mesh), primal.placement)


def placed_like(v, ref):
    """Local ``v`` placed as traced ``ref`` (a tangent of a value or zeros), as ``ref``'s shape says."""
    return v if ref.placement is None else placed(v, ref.placement, ref.shape)
