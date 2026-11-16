"""Contraction by axis name: ``dot`` and ``contract`` of named traced tensors, each a ``dot_general`` (or a chain
of them) whose result the primitive's rule names (``Primitive::names``)."""

from __future__ import annotations

from typing import Any, Sequence

from lumen.graph.tracer import TracedTensor, _accum_dtype, _common_dtype, _dot, _lift, _union
from lumen.tensor import names as _names

__all__ = ["dot", "contract"]


def _named(x: Any, op: str) -> TracedTensor:
    """``x`` as a traced tensor with names, or a TypeError naming ``op``."""
    x = _lift(x)
    if not isinstance(x, TracedTensor) or x.names is None:
        raise TypeError(f"{op} contracts by name: it takes named tensors, got {x!r}: name its axes first (rename)")
    return x


def _given(names: Any, op: str) -> tuple[str, ...]:
    """``names`` (a name or a sequence of them, each once) as a tuple."""
    names = (names,) if isinstance(names, str) else tuple(names)
    return _names.check(names, len(names), op)


def _pair(a: TracedTensor, b: TracedTensor, over: Sequence[str], batch: Sequence[str], accum: str, output: str):
    """One ``dot_general`` of ``a`` and ``b``: summing the axes ``over``,
    batched over ``batch`` (both in each), accumulated in ``accum`` and
    rounded to ``output``. Its axes are dot_general's order, so no
    transpose follows: ``batch``, then ``a``'s other axes, then ``b``'s."""
    numbers = (
        ([a.names.index(n) for n in over], [b.names.index(n) for n in over]),
        ([a.names.index(n) for n in batch], [b.names.index(n) for n in batch]),
    )
    return _dot(a, b, numbers, accum, output)


def _dtypes(op: str, operands: Sequence[TracedTensor], accum_dtype: str | None, output_dtype: str | None):
    """The operands' common dtype, and the accumulation and output dtypes:
    by default the matmul accumulation dtype of it (float32 for floats
    narrower) and it."""
    dtype = _common_dtype(op, operands)
    return dtype, accum_dtype or _accum_dtype(dtype), output_dtype or dtype


def dot(
    a: Any,
    b: Any,
    over: Any,
    *,
    accum_dtype: str | None = None,
    output_dtype: str | None = None,
) -> TracedTensor:
    """``a`` times ``b`` (named tensors) summed over the axes ``over`` (a
    name or names, in both); the other axes both have are batch axes,
    matched by name, and each one's own axes are kept: a matmul, a batched
    matmul or a linear layer by name (``dot(x, w, over="embed")``). One
    ``dot_general``, accumulated in ``accum_dtype`` (default: float32 for
    float operands, float64 and integers their own) and rounded to
    ``output_dtype`` (default: the operands' dtype). The result's axis
    order is dot_general's: batch axes, then ``a``'s, then ``b``'s."""
    a, b = _named(a, "dot"), _named(b, "dot")
    over = _given(over, "dot")
    missing = [n for n in over if n not in a.names or n not in b.names]
    if missing:
        raise KeyError(f"dot: summed axes must be in both operands: {missing} not in both of {a.names} and {b.names}")
    _union("dot", (a, b))
    _, accum, output = _dtypes("dot", (a, b), accum_dtype, output_dtype)
    batch = [n for n in a.names if n in b.names and n not in over]
    return _pair(a, b, over, batch, accum, output)


def contract(
    *operands: Any,
    keep: Any = (),
    accum_dtype: str | None = None,
    output_dtype: str | None = None,
) -> TracedTensor:
    """The einsum of named ``operands`` by name: every axis two or more of
    them share is summed, except those in ``keep`` (a name or names), batch
    axes the result keeps with every axis one operand alone has. Two
    operands are one ``dot_general`` (as :func:`dot`); more, a chain of
    them left to right, where an axis a later operand also has stays a
    batch axis until its last operand, and each step but the last is
    rounded to the operands' dtype. Dtypes as :func:`dot`."""
    if len(operands) < 2:
        raise ValueError(f"contract takes two or more named tensors, got {len(operands)}")
    operands = [_named(x, "contract") for x in operands]
    keep = _given(keep, "contract")
    _union("contract", operands)
    dtype, accum, output = _dtypes("contract", operands, accum_dtype, output_dtype)
    out = operands[0]
    for i, nxt in enumerate(operands[1:], 1):
        later = set(keep).union(*(x.names for x in operands[i + 1 :]))
        shared = [n for n in out.names if n in nxt.names]
        last = i == len(operands) - 1
        over = [n for n in shared if n not in later]
        batch = [n for n in shared if n in later]
        out = _pair(out, nxt, over, batch, accum, output if last else dtype)
    return out
