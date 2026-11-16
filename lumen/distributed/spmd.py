"""JAX-style ``shard_map`` over a ``DeviceMesh``, ``PartitionSpec`` and the named collectives a mapped function
calls (``psum``, ``all_gather``, ``ppermute``, ...), with ``jax.lax``'s names and arguments.

lumen runs one process per device, so a mapped function is traced by every process for its own shard: the
function's arguments are the global tensors, which every process holds; each process slices its block out
inside the trace, runs the function on it, and gathers the outputs back so every process returns the global
results. Axis names resolve to the mapped mesh's sub-groups of this process, and the collectives lower to
``lumen.distributed``'s primitives over them."""

from __future__ import annotations

import functools
import math
from typing import Any, Callable

import lumen.distributed as dist
from lumen.autograd import Function
from lumen.distributed._C import DeviceMesh
from lumen.graph import prims, tracer

__all__ = [
    "P",
    "PartitionSpec",
    "all_gather",
    "all_to_all",
    "axis_index",
    "axis_size",
    "pmax",
    "pmean",
    "pmin",
    "ppermute",
    "psum",
    "psum_scatter",
    "shard_map",
]

# The meshes of the shard_maps whose functions are being traced, innermost last.
_MESHES: list[DeviceMesh] = []


class PartitionSpec(tuple):
    """How a tensor's dimensions split over mesh axes (``jax.sharding.PartitionSpec``): entry ``d`` is a mesh
    axis name, a tuple of names (their shards nested, the first outermost) or None (dimension ``d`` whole);
    dimensions past the last entry are whole too. A tensor is replicated along the axes it doesn't name."""

    def __new__(cls, *partitions: str | tuple[str, ...] | None) -> PartitionSpec:
        """A spec of one entry per leading dimension."""
        return super().__new__(cls, partitions)

    def __repr__(self) -> str:
        """``PartitionSpec('dp', None)``, as constructed."""
        return f"PartitionSpec({', '.join(map(repr, self))})"


P = PartitionSpec


def _names(axis_name: Any) -> tuple[str, ...]:
    """An axis name or a tuple (list) of them, as a tuple of names."""
    return tuple(axis_name) if isinstance(axis_name, (tuple, list)) else (axis_name,)


def _axes(axis_name: Any, op: str) -> tuple[DeviceMesh, tuple[str, ...]]:
    """The innermost shard_map's mesh and ``axis_name`` as a tuple of its axes; ``op`` names the caller in
    the ``NameError`` for a name no shard_map binds."""
    names = _names(axis_name)
    if not _MESHES:
        raise NameError(
            f"{op}(): unbound axis name {axis_name!r}: named collectives run inside a function mapped by shard_map"
        )
    mesh = _MESHES[-1]
    for name in names:
        if name not in mesh.mesh_dim_names:
            raise NameError(f"{op}(): unbound axis name {name!r}; the shard_map's mesh has axes {mesh.mesh_dim_names}")
    if len(set(names)) != len(names):
        raise ValueError(f"{op}(): axis names {names} repeat an axis")
    return mesh, names


def _group(mesh: DeviceMesh, names: tuple[str, ...]) -> DeviceMesh:
    """This process's sub-mesh along ``names``: its ranks in the order of the axes' linear index."""
    return mesh[names[0] if len(names) == 1 else names]


def _size(mesh: DeviceMesh, names: tuple[str, ...]) -> int:
    """How many processes the axes ``names`` span together."""
    return math.prod(mesh.size(name) for name in names)


def _index(mesh: DeviceMesh, names: tuple[str, ...]) -> int:
    """This process's linear index along the axes ``names``, the first outermost."""
    index = 0
    for name in names:
        index = index * mesh.size(name) + mesh.get_local_rank(name)
    return index


def _moveaxis(x, source: int, destination: int):
    """``x`` with dimension ``source`` moved to position ``destination``, the others in order."""
    order = [d for d in range(x.ndim) if d != source]
    order.insert(destination, source)
    return x if source == destination else prims.transpose(x, order)


def _map_leaves(f: Callable, x: Any) -> Any:
    """``f`` of each traced tensor in ``x`` (a tensor, or modules, lists and tuples of them), as JAX's
    collectives map over pytrees."""
    return tracer._tree_map(f, x)


def axis_index(axis_name: str | tuple[str, ...]) -> int:
    """This process's index along the mesh axis ``axis_name`` (a tuple of axes: their linear index, the first
    outermost), as a Python int: each process traces its own program (``jax.lax.axis_index``)."""
    mesh, names = _axes(axis_name, "axis_index")
    return _index(mesh, names)


def axis_size(axis_name: str | tuple[str, ...]) -> int:
    """How many processes the mesh axis ``axis_name`` (or a tuple of axes) spans (``jax.lax.axis_size``)."""
    mesh, names = _axes(axis_name, "axis_size")
    return _size(mesh, names)


def _reduce(x: Any, axis_name: Any, op: str, name: str) -> Any:
    """Every tensor in ``x`` combined with ``op`` across the processes of ``axis_name``; a Python number is
    one value per process (``psum(1, axis)`` is the axis size)."""
    mesh, names = _axes(axis_name, name)
    if isinstance(x, (int, float)):
        return x * _size(mesh, names) if op == "sum" else x
    group = _group(mesh, names)
    return _map_leaves(lambda t: dist.all_reduce(t, op, group), x)


def psum(x: Any, axis_name: str | tuple[str, ...]) -> Any:
    """``x`` summed across the processes of mesh axis ``axis_name`` (``jax.lax.psum``)."""
    return _reduce(x, axis_name, "sum", "psum")


def pmean(x: Any, axis_name: str | tuple[str, ...]) -> Any:
    """``x`` averaged across the processes of mesh axis ``axis_name`` (``jax.lax.pmean``)."""
    n = axis_size(axis_name)
    return _map_leaves(lambda t: t / n, psum(x, axis_name)) if not isinstance(x, (int, float)) else x


def pmax(x: Any, axis_name: str | tuple[str, ...]) -> Any:
    """``x``'s elementwise maximum across the processes of mesh axis ``axis_name`` (``jax.lax.pmax``); not
    differentiable."""
    return _reduce(x, axis_name, "max", "pmax")


def pmin(x: Any, axis_name: str | tuple[str, ...]) -> Any:
    """``x``'s elementwise minimum across the processes of mesh axis ``axis_name`` (``jax.lax.pmin``); not
    differentiable."""
    return _reduce(x, axis_name, "min", "pmin")


def all_gather(x: Any, axis_name: str | tuple[str, ...], *, axis: int = 0, tiled: bool = False) -> Any:
    """Every process's ``x`` along mesh axis ``axis_name``, in axis order: stacked along a new dimension
    ``axis``, or with ``tiled`` concatenated along dimension ``axis`` (``jax.lax.all_gather``)."""
    mesh, names = _axes(axis_name, "all_gather")
    group = _group(mesh, names)

    def gather(t):
        """``t`` gathered from every process of the group."""
        dim = axis % (t.ndim + (0 if tiled else 1))
        if not tiled:
            t = prims.reshape(t, [*t.shape[:dim], 1, *t.shape[dim:]])
        return dist.all_gather(t, dim, group)

    return _map_leaves(gather, x)


def psum_scatter(x: Any, axis_name: str | tuple[str, ...], *, scatter_dimension: int = 0, tiled: bool = False) -> Any:
    """``x`` summed across the processes of mesh axis ``axis_name``, each keeping its block of dimension
    ``scatter_dimension`` (split into one per process, in axis order); without ``tiled`` that dimension's size
    is the axis size and it is dropped (``jax.lax.psum_scatter``)."""
    mesh, names = _axes(axis_name, "psum_scatter")
    group, n = _group(mesh, names), _size(mesh, names)

    def scatter(t):
        """``t`` reduce-scattered over the group."""
        dim = scatter_dimension % t.ndim
        if not tiled and t.shape[dim] != n:
            raise ValueError(
                f"psum_scatter(): dimension {dim} has size {t.shape[dim]}, not the {n} processes of {axis_name!r}"
            )
        out = dist.reduce_scatter(t, dim, "sum", group)
        return out if tiled else prims.reshape(out, [*t.shape[:dim], *t.shape[dim + 1 :]])

    return _map_leaves(scatter, x)


def all_to_all(
    x: Any, axis_name: str | tuple[str, ...], split_axis: int, concat_axis: int, *, tiled: bool = False
) -> Any:
    """``x`` split along ``split_axis`` into one block per process of mesh axis ``axis_name``, block ``j`` sent
    to the ``j``-th; the blocks received concatenated along ``concat_axis`` in axis order. Without ``tiled``,
    ``split_axis``'s size is the axis size and the received blocks stack along a new ``concat_axis`` instead
    (``jax.lax.all_to_all``)."""
    mesh, names = _axes(axis_name, "all_to_all")
    group, n = _group(mesh, names), _size(mesh, names)

    def exchange(t):
        """``t``'s blocks exchanged over the group."""
        split, concat = split_axis % t.ndim, concat_axis % t.ndim
        if tiled:
            return dist.all_to_all(t, split, concat, group)
        if t.shape[split] != n:
            raise ValueError(
                f"all_to_all(): split_axis {split} has size {t.shape[split]}, not the {n} processes of {axis_name!r}"
            )
        # Blocks of one along `split` come back in sender order there, then move to `concat`.
        return _moveaxis(dist.all_to_all(t, split, split, group), split, concat)

    return _map_leaves(exchange, x)


def _permute(x, ranks: list[int], index: int, pairs: tuple[tuple[int, int], ...]):
    """What process ``index`` of the group ``ranks`` receives when each ``(source, destination)`` of ``pairs``
    sends its ``x``: the source's tensor, or zeros where none sends to it."""
    to = dict(pairs)
    source = {d: s for s, d in pairs}

    # Blocking sends deadlock around a cycle unless one process sends first: the cycle's smallest index.
    # A chain's head receives nothing and sends at once; everyone else receives, then passes theirs on.
    cycle, j = [index], to.get(index)
    while j is not None and j != index:
        cycle.append(j)
        j = to.get(j)
    send_first = index not in source or (j == index and index == min(cycle))

    def send():
        """Sends ``x`` to this process's destination, if it has one."""
        if index in to:
            dist.send(x, ranks[to[index]], group=ranks)

    if send_first:
        send()
    if index in source:
        out = dist.recv(x, ranks[source[index]], group=ranks)
        # Its sender's send always runs, so the receive must too, even where nothing reads it (a gradient
        # needs no forward value); otherwise a later receive from that sender would take this message.
        tracer.keep(out)
    else:
        out = prims.full(x.shape, 0, x.dtype)
    if not send_first:
        send()
    return out


class _Permute(Function):
    """``ppermute`` with its derivative: the cotangents travel the inverse permutation."""

    @staticmethod
    def forward(ctx, x, route):
        """``x`` permuted along ``route`` (the group's ranks, this process's index and the pairs)."""
        ctx.route = route
        return _permute(x, *route)

    @staticmethod
    def backward(ctx, ct):
        """Each destination's cotangent sent back to its source."""
        ranks, index, pairs = ctx.route
        return _permute(ct, ranks, index, tuple((d, s) for s, d in pairs)), None

    @staticmethod
    def jvp(ctx, tangent, route_tangent):
        """The tangent permuted as the value is."""
        return _permute(tangent, *ctx.route)


def ppermute(x: Any, axis_name: str | tuple[str, ...], perm) -> Any:
    """``x`` sent along mesh axis ``axis_name`` by ``perm``'s ``(source, destination)`` index pairs (no source
    or destination twice): each destination gets its source's ``x``, a process no pair sends to gets zeros
    (``jax.lax.ppermute``). It runs over sends and receives, ordered so blocking ones never deadlock."""
    mesh, names = _axes(axis_name, "ppermute")
    n, index = _size(mesh, names), _index(mesh, names)
    pairs = tuple((int(s), int(d)) for s, d in perm)
    sources, destinations = [s for s, _ in pairs], [d for _, d in pairs]
    if len(set(sources)) != len(sources) or len(set(destinations)) != len(destinations):
        raise ValueError(f"ppermute(): perm {perm} repeats a source or a destination")
    if not all(0 <= i < n for i in sources + destinations):
        raise ValueError(f"ppermute(): perm {perm} has an index outside the {n} processes of {axis_name!r}")
    route = (dist._ranks(_group(mesh, names)), index, pairs)

    def permute(t):
        """``t`` permuted; a process that only keeps its own value, or takes part in no pair, needs no
        message."""
        if (index, index) in pairs:
            return t
        if index not in sources + destinations:
            return prims.full(t.shape, 0, t.dtype)
        return _Permute.apply(t, route)

    return _map_leaves(permute, x)


def _spec_dims(spec: PartitionSpec | None, ndim: int, mesh: DeviceMesh, where: str) -> tuple[tuple[str, ...], ...]:
    """``spec`` for a tensor of ``ndim`` dimensions: the tuple of mesh axes each dimension splits over (empty
    for a whole one); ``where`` names the spec in errors."""
    spec = PartitionSpec() if spec is None else spec
    if len(spec) > ndim:
        raise ValueError(f"shard_map(): {where} {spec} has more entries than the tensor's {ndim} dimensions")
    dims = [() if entry is None else _names(entry) for entry in spec]
    named = [name for d in dims for name in d]
    for name in named:
        if name not in mesh.mesh_dim_names:
            raise NameError(f"shard_map(): {where} {spec} names axis {name!r}, not one of {mesh.mesh_dim_names}")
    if len(set(named)) != len(named):
        raise ValueError(f"shard_map(): {where} {spec} names an axis twice")
    return tuple(dims) + ((),) * (ndim - len(dims))


def _unnamed(mesh: DeviceMesh, dims: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """The mesh axes no dimension of ``dims`` splits over: those the tensor is replicated along."""
    named = {name for d in dims for name in d}
    return tuple(name for name in mesh.mesh_dim_names if name not in named)


def _block(x, mesh: DeviceMesh, dims: tuple[tuple[str, ...], ...]):
    """This process's block of the global ``x`` split as ``dims`` say: a slice inside the trace."""
    start, limit = [], []
    for d, names in enumerate(dims):
        n = _size(mesh, names)
        if x.shape[d] % n:
            raise ValueError(f"shard_map(): dimension {d} of size {x.shape[d]} doesn't split over axes {names} ({n})")
        size = x.shape[d] // n
        start.append(_index(mesh, names) * size)
        limit.append(start[-1] + size)
    return prims.slice(x, start, limit)


def _assemble(x, mesh: DeviceMesh, dims: tuple[tuple[str, ...], ...]):
    """The global tensor whose block ``x`` is, split as ``dims`` say: each split dimension gathered over its
    axes' sub-group. A new traced value even when nothing is gathered."""
    for d, names in enumerate(dims):
        if names:
            x = dist.all_gather(x, d, _group(mesh, names))
    return x if any(dims) else prims.reshape(x, x.shape)


class _Shard(Function):
    """A global input's block on this process. Each process's block cotangent goes back into the global
    cotangent every process holds: summed along the axes the input is replicated over, then gathered."""

    @staticmethod
    def forward(ctx, x, layout):
        """``x``'s block for ``layout`` (the mesh and each dimension's axes)."""
        ctx.layout = layout
        return _block(x, *layout)

    @staticmethod
    def backward(ctx, ct):
        """The global cotangent, from every process's block cotangent."""
        mesh, dims = ctx.layout
        unnamed = _unnamed(mesh, dims)
        if _size(mesh, unnamed) > 1:
            ct = dist.all_reduce(ct, "sum", _group(mesh, unnamed))
        return _assemble(ct, mesh, dims), None

    @staticmethod
    def jvp(ctx, tangent, layout_tangent):
        """The tangent's block."""
        return _block(tangent, *ctx.layout)


class _Unshard(Function):
    """A mapped output's global value, gathered from every process's block. The global cotangent, which every
    process holds, gives each process its block's, split evenly among the copies of the axes the output is
    replicated over (each copy's backward is summed, as JAX's shard_map transpose does)."""

    @staticmethod
    def forward(ctx, x, layout):
        """The global output whose block ``x`` is."""
        ctx.layout = layout
        return _assemble(x, *layout)

    @staticmethod
    def backward(ctx, ct):
        """This process's block of the global cotangent, divided by the number of replicas."""
        mesh, dims = ctx.layout
        ct = _block(ct, mesh, dims)
        copies = _size(mesh, _unnamed(mesh, dims))
        return (ct / copies if copies > 1 else ct), None

    @staticmethod
    def jvp(ctx, tangent, layout_tangent):
        """The tangent gathered as the value is."""
        return _assemble(tangent, *ctx.layout)


def _match(specs: Any, tree: Any, f: Callable, where: str) -> Any:
    """``tree`` with each traced tensor ``t`` replaced by ``f(t, spec)``: ``specs`` is a prefix of ``tree``,
    a spec (or None, replicated) standing for every tensor below it."""
    if specs is None or isinstance(specs, PartitionSpec):
        return tracer._tree_map(lambda t: f(t, specs, where), tree)
    if isinstance(specs, (tuple, list)) and isinstance(tree, (tuple, list)) and len(specs) == len(tree):
        return type(tree)(_match(s, t, f, f"{where}[{i}]") for i, (s, t) in enumerate(zip(specs, tree)))
    kind = f"a {type(tree).__name__} of {len(tree)}" if isinstance(tree, (tuple, list)) else type(tree).__name__
    raise ValueError(f"shard_map(): {where} {specs!r} doesn't match the structure it maps ({kind})")


def shard_map(f: Callable | None = None, *, mesh: DeviceMesh, in_specs: Any, out_specs: Any) -> Callable:
    """``f`` mapped over ``mesh``'s processes (``jax.shard_map``); also a decorator taking only the keywords.

    The mapped function takes the global tensors, which every process holds. ``in_specs`` (a prefix of the
    positional arguments: one ``PartitionSpec`` for all, or one per argument) says how each splits over the
    mesh's named axes; each process slices out its block and runs ``f`` on it, where the named collectives
    (``psum``, ``all_gather``, ...) reach the processes along an axis. ``out_specs`` (a prefix of ``f``'s
    outputs) says how the blocks ``f`` returns assemble: a dimension naming axes concatenates their blocks, and
    an axis no dimension names means the blocks along it are equal and one is the result. Every process
    returns the global outputs. Runs only while tracing, inside ``lumen.compile``."""
    if f is None:
        return functools.partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    if not mesh.mesh_dim_names:
        raise ValueError("shard_map(): the mesh needs mesh_dim_names to name its axes")

    def shard(t, spec, where):
        """``t``'s block on this process."""
        dims = _spec_dims(spec, t.ndim, mesh, where)
        return t if mesh.size() == 1 else _Shard.apply(t, (mesh, dims))

    def unshard(t, spec, where):
        """The global output whose block ``t`` is."""
        dims = _spec_dims(spec, t.ndim, mesh, where)
        return t if mesh.size() == 1 else _Unshard.apply(t, (mesh, dims))

    @functools.wraps(f)
    def mapped(*args):
        """``f`` run on this process's blocks of ``args``, its outputs assembled into the global ones."""
        tracer.current_graph()
        if mesh.get_coordinate() is None:
            raise ValueError(f"shard_map(): process {dist.get_rank()} is not in the mesh {mesh!r}")
        blocks = _match(in_specs, args, shard, "in_specs")
        _MESHES.append(mesh)
        try:
            out = f(*blocks)
        finally:
            _MESHES.pop()
        return _match(out_specs, out, unshard, "out_specs")

    return mapped
