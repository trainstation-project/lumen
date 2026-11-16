"""How metadata on tensors flows through each primitive: one ``Rule`` per primitive (``RULES``), as
``Primitive::infer`` gives each one's output type. A rule infers its output's axis names from its operands'
(PyTorch's named-tensor name inference); ``prims.bind`` applies it to every node with a named operand. A
primitive without a rule drops the names."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

Names = tuple[str, ...] | None


class Rule:
    """A primitive's propagation rule; this base one, the default, drops names. Subclasses override
    ``names``; other metadata (sharding placements) gets a method of its own beside it."""

    def names(self, operands: Sequence[Any], params: Mapping[str, Any], shape: tuple[int, ...]) -> Names:
        """The axis names of the output (of ``shape``) of the primitive on ``operands`` (traced tensors, at
        least one named) with ``params`` (as ``prims.bind`` gets them), or None for none."""
        return None


def _agreed(named: Sequence[tuple[str, ...]], what: str) -> Names:
    """The one name tuple all of ``named`` share, or None if there are none; a ValueError naming ``what``
    if two differ."""
    for names in named[1:]:
        if names != named[0]:
            raise ValueError(
                f"{what}: operands named {named[0]} and {names} do not match: align them first (align_to)"
            )
    return named[0] if named else None


class Elementwise(Rule):
    """An op on operands of its output's shape: their names, which every named operand must share (the
    tracer aligns operands by name before such an op: ``tracer._broadcast``); with an unnamed one, the
    named one's."""

    def names(self, operands, params, shape):
        """The named operands' common names."""
        return _agreed([x.names for x in operands if x.names is not None and x.shape == shape], "elementwise op")


class First(Rule):
    """An op whose output has its first operand's axes, perhaps resized (a slice, a cumulative sum, a write
    into it): that operand's names."""

    def names(self, operands, params, shape):
        """The first operand's names."""
        return operands[0].names


class Reduce(Rule):
    """A reduction over ``params["axes"]``: the operand's names but those."""

    def names(self, operands, params, shape):
        """The operand's names without the reduced axes'."""
        x = operands[0]
        if x.names is None:
            return None
        return tuple(n for d, n in enumerate(x.names) if d not in params["axes"])


class Transpose(Rule):
    """A transpose: the operand's names, permuted as its axes."""

    def names(self, operands, params, shape):
        """The operand's names in ``params["permutation"]``'s order."""
        x = operands[0]
        return None if x.names is None else tuple(x.names[d] for d in params["permutation"])


class SameShape(Rule):
    """A reshape or broadcast: names only when it changes nothing (the output of the operand's shape, and a
    broadcast's dims in order). A named split or merge, or a broadcast aligning operands by name, names
    its output itself."""

    def names(self, operands, params, shape):
        """The operand's names when the op leaves its shape as it is."""
        x = operands[0]
        dims = params.get("broadcast_dimensions", tuple(range(len(shape))))
        return x.names if x.shape == shape and tuple(dims) == tuple(range(len(shape))) else None


class DotGeneral(Rule):
    """``dot_general``: the batch axes' names (the same in both operands), then each operand's free axes'
    (a contracted axis is dropped), as its output's dims are ordered; None unless both are named, or
    when a name would repeat."""

    def names(self, operands, params, shape):
        """The output's names from the two operands'."""
        lhs, rhs = operands
        if lhs.names is None or rhs.names is None:
            return None
        batch = tuple(lhs.names[d] for d in params["lhs_batch"])
        if batch != tuple(rhs.names[d] for d in params["rhs_batch"]):
            raise ValueError(
                f"dot_general: batch axes named {batch} and {tuple(rhs.names[d] for d in params['rhs_batch'])} "
                "do not match"
            )
        lhs_used = set(params["lhs_batch"]) | set(params["lhs_contracting"])
        rhs_used = set(params["rhs_batch"]) | set(params["rhs_contracting"])
        names = (
            *batch,
            *(n for d, n in enumerate(lhs.names) if d not in lhs_used),
            *(n for d, n in enumerate(rhs.names) if d not in rhs_used),
        )
        return names if len(set(names)) == len(names) else None


class Gather(Rule):
    """``gather`` along ``params["axis"]``: the operand's names with that axis' replaced by the indices'
    (a 0-d index has none to add); None unless both are named, or when a name would repeat."""

    def names(self, operands, params, shape):
        """The operand's names with the gathered axis' replaced by the indices'."""
        x, indices = operands
        ids = indices.names if indices.names is not None or indices.shape else ()
        if x.names is None or ids is None:
            return None
        d = params["axis"]
        names = x.names[:d] + ids + x.names[d + 1 :]
        return names if len(set(names)) == len(names) else None


class TakeAlongAxis(Rule):
    """``take_along_axis``: the output has the indices' shape and the operand's axes, so the indices'
    names, or the operand's when the indices have none."""

    def names(self, operands, params, shape):
        """The indices' names, else the operand's."""
        x, indices = operands
        return indices.names if indices.names is not None else x.names


class Concatenate(Rule):
    """``concatenate``: the operands' names, which the named ones must share."""

    def names(self, operands, params, shape):
        """The named operands' common names."""
        return _agreed([x.names for x in operands if x.names is not None], "concatenate")


_ELEMENTWISE = Elementwise()
_FIRST = First()
_SAME_SHAPE = SameShape()

# Each primitive's rule, by its name in the graph (``prims.bind``'s); one missing drops names (``DROP``).
RULES: dict[str, Rule] = {
    **{
        name: _ELEMENTWISE
        for name in (
            # Unary: their operand has the output's shape.
            "neg exp log sqrt tanh logistic cast sin cos tan asin acos atan sinh cosh asinh acosh atanh floor "
            "ceil round log1p expm1 exp2 log2 rsqrt cbrt erf_inv is_finite abs sign not population_count clz "
            "reduce_precision to_host to_device host_wait device_wait all_reduce broadcast reduce "
            # Binary and select: operands of one shape.
            "add sub mul div max min eq lt pow rem and or xor shift_left shift_right_logical "
            "shift_right_arithmetic nextafter select"
        ).split()
    },
    "reduce_sum": Reduce(),
    "reduce_max": Reduce(),
    "cumsum": _FIRST,
    "slice": _FIRST,
    "dynamic_slice": _FIRST,
    "dynamic_update_slice": _FIRST,
    "scatter_add": _FIRST,
    "transpose": Transpose(),
    "reshape": _SAME_SHAPE,
    "broadcast_in_dim": _SAME_SHAPE,
    "dot_general": DotGeneral(),
    "gather": Gather(),
    "take_along_axis": TakeAlongAxis(),
    "concatenate": Concatenate(),
}

DROP = Rule()


def rule(name: str) -> Rule:
    """Primitive ``name``'s rule (``DROP`` for one without)."""
    return RULES.get(name, DROP)
