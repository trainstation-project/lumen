"""The rules autodiff is made of (JAX: ``jax/_src/interpreters/ad.py``):
a JVP rule per primitive, a transpose rule per linear one, and what they
are written with."""

import functools

from lumen.graph import prims, tracer
from lumen.graph.tracer import TracedTensor

# Keyed by primitive, its ``prims`` function (JAX: by ``Primitive``).
# primitive -> rule(primals, tangents, out, **params): the tangent of
# `out` (None: zero), given the operands' (None: zero).
primitive_jvps = {}
# primitive -> rule(ct, *operands, **params) for a linear primitive: the
# cotangent of each operand (None for one it is not linear in), given
# its output's, its linear operands UndefinedPrimal.
primitive_transposes = {}


class UndefinedPrimal:
    """A linear operand of a node being transposed: only its type is known
    (JAX's ``ad.UndefinedPrimal``)."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype

    @property
    def ndim(self):
        return len(self.shape)


def _is_float(dtype):
    return tracer._is_float(dtype)


def _full(like, value):
    """A tensor of ``like``'s type filled with ``value``."""
    return prims.full(like.shape, float(value) if _is_float(like.dtype) else int(value), like.dtype)


def _sum(xs):
    """The sum of the tensors among ``xs`` (None: zero), or None."""
    xs = [x for x in xs if x is not None]
    return functools.reduce(prims.add, xs) if xs else None


def defjvp(primitive, *rules):
    """``primitive``'s JVP (``prims.exp``): the sum over its operands with
    a tangent of ``rule(t, out, *primals, **params)`` (JAX's ``defjvp2``)."""

    def jvp(primals, tangents, out, **params):
        return _sum(rule(t, out, *primals, **params) for rule, t in zip(rules, tangents) if rule and t is not None)

    primitive_jvps[primitive] = jvp


def deflinear(primitive, transpose_rule):
    """``primitive`` (``prims.neg``) is linear in all its operands: its JVP
    is itself, on the tangents (zeros for those without);
    ``transpose_rule``, its transpose."""

    def jvp(primals, tangents, out, **params):
        tangents = [t if t is not None else _full(p, 0) for p, t in zip(primals, tangents)]
        return prims.bind(primitive.__name__, *tangents, **params)

    primitive_jvps[primitive] = jvp
    primitive_transposes[primitive] = transpose_rule


def _value(var):
    return TracedTensor(tracer.current_graph(), var)


def _accumulate(cts, v, c):
    """Add cotangent ``c`` to ``v``'s in ``cts``."""
    cts[v] = prims.add(_value(cts[v]), c).var if v in cts else c.var
