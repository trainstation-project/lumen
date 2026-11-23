"""The cost of moving a value between placements (DTensor's ``redistribute_cost``) and the moves that take it there
(``moves``), which ``sharding`` compares strategies by and ``redistribute`` records."""

from __future__ import annotations

import math
from typing import Any, Sequence

from lumen.tensor import placements as _placements
from lumen.tensor.placements import Partial, Replicate, Shard, Sharding

# A collective's fixed cost, in bytes moved: of two strategies moving as much, the one with fewer
# collectives wins.
LATENCY = 1024


def _itemsize(dtype: str) -> int:
    """Bytes per element of ``dtype`` in memory (a sub-byte dtype takes a byte)."""
    if dtype == "bool":
        return 1
    return max(1, int("".join(c for c in dtype.split("_")[0] if c.isdigit())) // 8)


def valid(spec: Sharding, shape: Sequence[int], even: bool = False) -> bool:
    """Whether a value of global ``shape`` can be placed by ``spec``: each sharded dimension one of its own,
    and, with ``even``, splitting evenly (a reshape's blocks map to blocks only then)."""
    dims = [p.dim for p in spec.placements if isinstance(p, Shard)]
    return all(d < len(shape) for d in dims) and (not even or _placements.even(shape, spec))


def _nested(spec: Sharding) -> bool:
    """Whether ``spec`` shards one tensor dimension over several mesh dimensions."""
    dims = [p.dim for p in spec.placements if isinstance(p, Shard)]
    return len(set(dims)) != len(dims)


def moves(src: Sharding, dst: Sharding) -> list[tuple[int, Any, Any]]:
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
    """``moves`` when a tensor dimension is sharded over several mesh dimensions in ``src`` or ``dst``: each
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
    for i, a, b in moves(src, dst):
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
