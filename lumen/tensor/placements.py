"""Sharding placements, the optional metadata of a ``lumen.Tensor`` and of a ``TracedTensor`` saying how a
tensor is laid out over a ``DeviceMesh`` (PyTorch DTensor's ``Shard``, ``Replicate`` and ``Partial``). A
sharded tensor's ``shape`` is its global (logical) shape; its storage, or its traced value, holds this
process's local shard, whose shape may differ between processes when a dimension doesn't split evenly. This module holds the placement types and the shape math both tensor types share.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from lumen.tensor import names as _names
from lumen.tensor.names import AxisType

# The reductions a ``Partial`` placement may be pending.
REDUCE_OPS = ("sum", "max", "min", "avg")


class Placement:
    """How a tensor is laid out along one dimension of a device mesh: ``Shard``, ``Replicate`` or
    ``Partial``."""


@dataclasses.dataclass(frozen=True)
class Shard(Placement):
    """Split along tensor dimension ``dim`` (a position or an axis name) into one block per process of the
    mesh dimension, in coordinate order, as ``torch.chunk`` splits (DTensor's): blocks of ``ceil(size / n)``,
    the last ones shorter or empty when the size doesn't split evenly. A dimension sharded over several mesh
    dimensions is split by each in mesh order, each splitting the block the ones before it left."""

    dim: AxisType


@dataclasses.dataclass(frozen=True)
class Replicate(Placement):
    """Every process of the mesh dimension holds the whole tensor."""


@dataclasses.dataclass(frozen=True)
class Partial(Placement):
    """Every process of the mesh dimension holds a part: the tensor is their reduction by ``reduce_op``
    (``"sum"``, ``"max"``, ``"min"`` or ``"avg"``), not yet done."""

    reduce_op: str = "sum"


@dataclasses.dataclass(frozen=True)
class Sharding:
    """A tensor's placement: a ``DeviceMesh`` and one ``Placement`` per mesh dimension, each ``Shard``'s
    dim a nonnegative position (``sharding`` resolves names)."""

    mesh: Any
    placements: tuple[Placement, ...]

    def __repr__(self):
        """The placements and the mesh's layout."""
        return f"Sharding({list(self.placements)}, mesh={self.mesh.mesh})"


def sharding(mesh: Any, placements: Sequence[Placement], ndim: int, names: tuple[str, ...] | None) -> Sharding:
    """``placements`` over ``mesh`` for a tensor of ``ndim`` dims named ``names`` as a ``Sharding``, each
    ``Shard``'s dim resolved to a position (``names.axis``); a ValueError or TypeError if they don't fit."""
    placements = tuple(placements)
    if len(placements) != mesh.ndim:
        raise ValueError(f"{len(placements)} placements {list(placements)} for a mesh of {mesh.ndim} dims")
    resolved = []
    for p in placements:
        if isinstance(p, Shard):
            p = Shard(_names.axis(p.dim, ndim, names, "Shard"))
            if not ndim:
                raise ValueError("Shard: a 0-d tensor has no dimension to shard")
        elif isinstance(p, Partial):
            if p.reduce_op not in REDUCE_OPS:
                raise ValueError(f"Partial: reduce_op is one of {REDUCE_OPS}, got {p.reduce_op!r}")
        elif not isinstance(p, Replicate):
            raise TypeError(f"a placement is Shard, Replicate or Partial, got {p!r}")
        resolved.append(p)
    return Sharding(mesh, tuple(resolved))


def replicated(mesh: Any) -> Sharding:
    """The placement of a tensor every process of ``mesh`` holds whole."""
    return Sharding(mesh, (Replicate(),) * mesh.ndim)


def chunk(size: int, n: int, k: int) -> tuple[int, int]:
    """Where block ``k`` of a dimension of ``size`` split over ``n`` processes starts, and its length:
    ``torch.chunk``'s blocks of ``ceil(size / n)``, the last ones shorter or empty."""
    c = -(-size // n)
    start = min(k * c, size)
    return start, min(start + c, size) - start


def _block(shape: Sequence[int], spec: Sharding, at: Sequence[int] | None) -> tuple[list[int], list[int]]:
    """The start and the size, per dimension, of the block of a tensor of global ``shape`` placed as
    ``spec`` that the process at mesh coordinate ``at`` holds; with ``at`` None, the largest block's size
    (every split rounded up), the same on every process, and starts of 0."""
    start, size = [0] * len(shape), list(shape)
    for i, p in enumerate(spec.placements):
        if isinstance(p, Shard):
            n = spec.mesh.size(i)
            if at is None:
                size[p.dim] = -(-size[p.dim] // n)
            else:
                offset, size[p.dim] = chunk(size[p.dim], n, at[i])
                start[p.dim] += offset
    return start, size


def local_shape(shape: Sequence[int], spec: Sharding) -> tuple[int, ...]:
    """The shape of this process's local shard of a tensor of global ``shape`` placed as ``spec``."""
    return tuple(_block(shape, spec, coordinate(spec.mesh))[1])


def block_shape(shape: Sequence[int], spec: Sharding) -> tuple[int, ...]:
    """The shape of the largest local shard of a tensor of global ``shape`` placed as ``spec``: every
    process computes the same one (choices made from it agree across processes)."""
    return tuple(_block(shape, spec, None)[1])


def global_shape(shape: Sequence[int], spec: Sharding) -> tuple[int, ...]:
    """The global shape of a tensor whose local shard, placed as ``spec``, has ``shape``, every sharded
    dimension split evenly (``from_local``'s default)."""
    full = list(shape)
    for i, p in enumerate(spec.placements):
        if isinstance(p, Shard):
            full[p.dim] *= spec.mesh.size(i)
    return tuple(full)


def even(shape: Sequence[int], spec: Sharding) -> bool:
    """Whether every sharded dimension of a tensor of global ``shape`` placed as ``spec`` splits evenly."""
    size = list(shape)
    for i, p in enumerate(spec.placements):
        if isinstance(p, Shard):
            if size[p.dim] % spec.mesh.size(i):
                return False
            size[p.dim] //= spec.mesh.size(i)
    return True


def coordinate(mesh: Any) -> list[int]:
    """This process's coordinate in ``mesh``; a RuntimeError if it isn't one of its processes."""
    at = mesh.get_coordinate()
    if at is None:
        raise RuntimeError(f"this process (rank {mesh.get_rank()}) is not in the mesh {mesh.mesh}")
    return at


def group(mesh: Any, dim: int) -> list[int]:
    """The global ranks of this process's peers along mesh dimension ``dim`` (its own included), in
    coordinate order: those whose coordinates differ from its own in ``dim`` alone."""
    at = coordinate(mesh)
    ranks = []
    for k in range(mesh.size(dim)):
        row = mesh.mesh
        for d, c in enumerate(at):
            row = row[k if d == dim else c]
        ranks.append(row)
    return ranks


def offsets(shape: Sequence[int], spec: Sharding) -> tuple[int, ...]:
    """Where this process's local shard of a tensor of global ``shape`` placed as ``spec`` starts, per
    dimension."""
    return tuple(_block(shape, spec, coordinate(spec.mesh))[0])
