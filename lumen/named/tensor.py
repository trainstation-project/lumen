"""``NamedTensor``: a tensor whose axes have names, which ``lumen.compile`` and autograd walk as a container."""

from __future__ import annotations

from typing import Any, Mapping

from lumen.graph import tracer

__all__ = ["NamedTensor"]


def _check_names(names: Any, op: str) -> tuple[str, ...]:
    """``names`` (a sequence of strings, each once) as a tuple, or a
    TypeError or ValueError naming ``op``."""
    if isinstance(names, str):
        raise TypeError(f"{op}: names are a sequence of strings, got the string {names!r}: write ({names!r},)")
    names = tuple(names)
    for n in names:
        if not isinstance(n, str):
            raise TypeError(f"{op}: an axis name is a string, got {n!r}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"{op}: axis names must be unique, got {', '.join(map(repr, repeated))} more than once")
    return names


def _order(names: tuple[str, ...], order: tuple[Any, ...], op: str) -> tuple[str, ...]:
    """``order``, a permutation of ``names`` in which one ``...`` stands for
    the names it leaves out (in their order), as a tuple of names."""
    if order.count(Ellipsis) > 1:
        raise ValueError(f"{op}: an order can have one '...', got {order}")
    given = _check_names([n for n in order if n is not Ellipsis], op)
    unknown = [n for n in given if n not in names]
    if unknown:
        raise KeyError(f"{op}: no axis {', '.join(map(repr, unknown))} in a tensor of axes {names}")
    rest = tuple(n for n in names if n not in given)
    if Ellipsis not in order:
        if rest:
            raise ValueError(f"{op}: the order {given} leaves out axes {rest}: name them or add '...'")
        return given
    at = order.index(Ellipsis)
    return given[:at] + rest + given[at:]


class NamedTensor:
    """A tensor (traced inside ``lumen.compile``, or a ``lumen.Tensor``
    passed to or returned from it) whose axes have names, unique per
    tensor: ``NamedTensor(x, ("batch", "embed"))``. Ops on it
    (``lumen.named``) take axis names instead of positions and broadcast by
    name, so the positional order of its axes is an implementation detail;
    they resolve names while tracing into positional ops on the tensor, so
    names cost nothing when the program runs.

    It is a container the tracer walks (``tracer.register_tree``): a
    compiled function takes it as an argument and returns it, and
    ``lumen.grad``, ``vjp`` and ``jvp`` differentiate through it (its
    gradient is named as it is). The names are static, part of a compiled
    function's signature."""

    __slots__ = ("_tensor", "names")

    def __init__(self, tensor: Any, names: Any):
        """``tensor`` with its axes named ``names``, in its positional order (one name each)."""
        names = _check_names(names, "NamedTensor")
        if len(names) != len(tensor.shape):
            raise ValueError(f"NamedTensor: {len(names)} names {names} for a tensor of shape {list(tensor.shape)}")
        self._tensor = tensor
        self.names = names

    @property
    def shape(self) -> dict[str, int]:
        """Each axis' size by name, in positional order."""
        return dict(zip(self.names, self._tensor.shape))

    @property
    def dtype(self) -> str:
        """The tensor's dtype."""
        return self._tensor.dtype

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.names)

    def size(self, axis: str) -> int:
        """The size of axis ``axis``."""
        return self._tensor.shape[self.axis(axis)]

    def axis(self, name: str) -> int:
        """The position of axis ``name`` in the underlying tensor; a KeyError if it has none."""
        if name not in self.names:
            raise KeyError(f"no axis {name!r} in a tensor of axes {self.names}")
        return self.names.index(name)

    def tensor(self, *order: Any) -> Any:
        """The positional tensor, its axes in ``order`` (names; one ``...``
        stands for the rest in their order; none given: as stored, with no
        transpose)."""
        if not order:
            return self._tensor
        order = _order(self.names, order, "tensor")
        if order == self.names:
            return self._tensor
        return self._tensor.permute([self.names.index(n) for n in order])

    def align_to(self, *order: Any) -> NamedTensor:
        """It with its axes stored in ``order`` (as :meth:`tensor` takes):
        the same named tensor to every op, laid out for a consumer that
        reads positions (the last axis contiguous)."""
        order = _order(self.names, order, "align_to")
        return NamedTensor(self.tensor(*order), order)

    def rename(self, mapping: Mapping[str, str] | None = None, **names: str) -> NamedTensor:
        """It with axes renamed: ``mapping`` and ``names`` map old names to new ones."""
        return ops.rename(self, mapping, **names)

    def to(self, dtype: str) -> NamedTensor:
        """It converted to ``dtype``."""
        return ops.cast(self, dtype)

    astype = to

    def __getitem__(self, selection: Mapping[str, Any]) -> NamedTensor:
        """It indexed by name: ``x[{"seq": slice(0, 4), "batch": 0}]`` (see ``lumen.named.index``)."""
        return ops.index(self, selection)

    def sum(self, axis: Any = None) -> NamedTensor:
        """Sum over ``axis`` (a name or names; all if None), see ``lumen.named.sum``."""
        return ops.sum(self, axis)

    def mean(self, axis: Any = None) -> NamedTensor:
        """Mean over ``axis`` (a name or names; all if None)."""
        return ops.mean(self, axis)

    def max(self, axis: Any = None) -> NamedTensor:
        """Maximum over ``axis`` (a name or names; all if None)."""
        return ops.max(self, axis)

    def min(self, axis: Any = None) -> NamedTensor:
        """Minimum over ``axis`` (a name or names; all if None)."""
        return ops.min(self, axis)

    def split(self, axis: str, /, **sizes: int) -> NamedTensor:
        """Axis ``axis`` split into ``sizes`` (see ``lumen.named.split``)."""
        return ops.split(self, axis, **sizes)

    def merge(self, axes: Any, into: str) -> NamedTensor:
        """Axes ``axes`` merged into one named ``into`` (see ``lumen.named.merge``)."""
        return ops.merge(self, axes, into)

    @property
    def grad(self) -> NamedTensor | None:
        """Its gradient after ``backward`` (named as it is), or None."""
        g = getattr(self._tensor, "grad", None)
        return None if g is None else NamedTensor(g, self.names)

    def backward(self, gradient: NamedTensor | None = None) -> None:
        """``TracedTensor.backward`` of the tensor: ``gradient`` (named; by
        default 1 for a tensor of no axes) is aligned to it by name."""
        if gradient is not None:
            gradient = ops.align(gradient, self, "backward")
        self._tensor.backward(gradient)

    def __repr__(self) -> str:
        """Its axes by name with their sizes, and its dtype."""
        axes = ", ".join(f"{n}={s}" for n, s in self.shape.items())
        return f"NamedTensor({axes}; dtype={self.dtype})"

    def __bool__(self) -> bool:
        """Raises: a traced value has no data to branch on."""
        return bool(self._tensor)

    __hash__ = object.__hash__

    def __add__(self, other: Any) -> NamedTensor:
        """``self + other``, broadcast by name."""
        return ops.add(self, other)

    def __radd__(self, other: Any) -> NamedTensor:
        """``other + self``, broadcast by name."""
        return ops.add(other, self)

    def __sub__(self, other: Any) -> NamedTensor:
        """``self - other``, broadcast by name."""
        return ops.sub(self, other)

    def __rsub__(self, other: Any) -> NamedTensor:
        """``other - self``, broadcast by name."""
        return ops.sub(other, self)

    def __mul__(self, other: Any) -> NamedTensor:
        """``self * other``, broadcast by name."""
        return ops.mul(self, other)

    def __rmul__(self, other: Any) -> NamedTensor:
        """``other * self``, broadcast by name."""
        return ops.mul(other, self)

    def __truediv__(self, other: Any) -> NamedTensor:
        """``self / other``, broadcast by name."""
        return ops.div(self, other)

    def __rtruediv__(self, other: Any) -> NamedTensor:
        """``other / self``, broadcast by name."""
        return ops.div(other, self)

    def __pow__(self, other: Any) -> NamedTensor:
        """``self ** other``, broadcast by name."""
        return ops.pow(self, other)

    def __rpow__(self, other: Any) -> NamedTensor:
        """``other ** self``, broadcast by name."""
        return ops.pow(other, self)

    def __neg__(self) -> NamedTensor:
        """Elementwise negation."""
        return ops.neg(self)

    def __abs__(self) -> NamedTensor:
        """Elementwise absolute value."""
        return ops.abs(self)

    def __eq__(self, other: Any) -> NamedTensor:  # type: ignore[override]
        """Elementwise ``self == other``, broadcast by name."""
        return ops.eq(self, other)

    def __ne__(self, other: Any) -> NamedTensor:  # type: ignore[override]
        """Elementwise ``self != other``, broadcast by name."""
        return ops.ne(self, other)

    def __lt__(self, other: Any) -> NamedTensor:
        """Elementwise ``self < other``, broadcast by name."""
        return ops.lt(self, other)

    def __le__(self, other: Any) -> NamedTensor:
        """Elementwise ``self <= other``, broadcast by name."""
        return ops.le(self, other)

    def __gt__(self, other: Any) -> NamedTensor:
        """Elementwise ``self > other``, broadcast by name."""
        return ops.gt(self, other)

    def __ge__(self, other: Any) -> NamedTensor:
        """Elementwise ``self >= other``, broadcast by name."""
        return ops.ge(self, other)


def _flatten(x: NamedTensor) -> tuple[tuple[Any], tuple[str, ...]]:
    """Its tensor, the child the tracer walks, and its names, the static part."""
    return (x._tensor,), x.names


def _unflatten(names: tuple[str, ...], children: tuple[Any]) -> NamedTensor:
    """A named tensor of ``names`` back from its child, unchecked: walks
    map the tensor to anything (``_tree_leaves`` to None), as JAX's
    unflatten does."""
    x = object.__new__(NamedTensor)
    x._tensor, x.names = children[0], names
    return x


tracer.register_tree(NamedTensor, _flatten, _unflatten)

# The ops import NamedTensor from here, so they come after it; its methods call them.
from lumen.named import ops  # noqa: E402
