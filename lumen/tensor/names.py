"""Axis names, the optional metadata of a ``lumen.Tensor`` and of a ``TracedTensor``: checking them and resolving
the names an op is given into positions, shared by both tensor types."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# An axis an op is given: a position (negative counts from the end) or the name of one of a tensor's axes.
AxisType = str | int
# One axis or several, where an op takes several (a reduction's).
Axes = AxisType | Sequence[AxisType]


def check(names: Any, ndim: int, op: str) -> tuple[str, ...]:
    """``names`` as a tuple: strings, each once, one per dim of an ``ndim``-d tensor; a TypeError or
    ValueError naming ``op`` otherwise."""
    if isinstance(names, str):
        raise TypeError(f"{op}: names are a sequence of strings, got the string {names!r}: write ({names!r},)")
    names = tuple(names)
    for n in names:
        if not isinstance(n, str):
            raise TypeError(f"{op}: an axis name is a string, got {n!r}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"{op}: axis names must be unique, got {', '.join(map(repr, repeated))} more than once")
    if len(names) != ndim:
        raise ValueError(f"{op}: {len(names)} names {names} for a tensor of {ndim} dims")
    return names


def position(names: tuple[str, ...] | None, name: str, op: str) -> int:
    """The position of axis ``name`` among ``names`` (a tensor's); a KeyError naming ``op`` if it has none,
    or no names at all."""
    if names is None:
        raise KeyError(
            f"{op}: dimension {name!r} is given by name, but the tensor has no names: name it first (rename)"
        )
    if name not in names:
        raise KeyError(f"{op}: no axis {name!r} in a tensor of axes {names}")
    return names.index(name)


def axis(axis: AxisType, ndim: int, names: tuple[str, ...] | None, op: str = "dim") -> int:
    """The nonnegative position of ``axis`` in an ``ndim``-d tensor named ``names``: an int in
    ``[-ndim, ndim)`` (a 0-d tensor takes 0 or -1; an IndexError otherwise) or one of its axis names (a
    KeyError naming ``op`` for one it lacks). Every op taking a dimension resolves it here."""
    if isinstance(axis, str):
        return position(names, axis, op)
    n = max(ndim, 1)
    if not -n <= axis < n:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-n}, {n - 1}], but got {axis})")
    return axis % n


def renamed(
    names: tuple[str, ...] | None, ndim: int, new: tuple[Any, ...], mapping: Mapping[str, str]
) -> tuple[str, ...] | None:
    """The names ``rename(*new, **mapping)`` gives a tensor of ``ndim`` dims named ``names`` (None:
    unnamed): ``new``, one name per dim (a lone None: no names), or ``names`` with those ``mapping`` holds
    replaced."""
    if new and mapping:
        raise TypeError("rename takes the new names or a mapping of old names to new ones, not both")
    if new == (None,):
        return None
    if not mapping:
        return check(new, ndim, "rename")
    for old in mapping:
        position(names, old, "rename")
    return check(tuple(mapping.get(n, n) for n in names), ndim, "rename")


def refined(names: tuple[str, ...] | None, ndim: int, new: tuple[Any, ...]) -> tuple[str, ...]:
    """The names ``refine_names(*new)`` gives a tensor of ``ndim`` dims named ``names``: ``new`` for an
    unnamed tensor; a named one's must be ``new`` already (a ValueError otherwise)."""
    new = check(new, ndim, "refine_names")
    if names is not None and names != new:
        raise ValueError(f"refine_names: a tensor of axes {names} cannot be refined to {new}")
    return new


def order(names: tuple[str, ...] | None, order: Sequence[Any], op: str) -> tuple[str, ...]:
    """``order``, a permutation of ``names`` in which one ``...`` stands for the names it leaves out (in
    their order), as a tuple of names."""
    if names is None:
        raise ValueError(f"{op}: the tensor has no names: name it first (rename)")
    order = tuple(order)
    if order.count(Ellipsis) > 1:
        raise ValueError(f"{op}: an order can have one '...', got {order}")
    given = check([n for n in order if n is not Ellipsis], len(order) - order.count(Ellipsis), op)
    for n in given:
        position(names, n, op)
    rest = tuple(n for n in names if n not in given)
    if Ellipsis not in order:
        if rest:
            raise ValueError(f"{op}: the order {given} leaves out axes {rest}: name them or add '...'")
        return given
    at = order.index(Ellipsis)
    return given[:at] + rest + given[at:]
