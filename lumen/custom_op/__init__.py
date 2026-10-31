"""User-defined custom ops: Python functions that mutate tensor arguments, called from compiled code, and Python
kernels registered for ops that accept them, keyed by op name and device."""

import functools
import inspect

from lumen.custom_op._C import MetalKernel, _register_kernel, _registered_ops, _unregister_kernel
from lumen.graph import prims, tracer
from lumen.graph._C import _register_custom_op
from lumen.graph.tracer import TracedTensor

__all__ = ["CustomOp", "MetalKernel", "custom_op", "register", "registered_ops", "signature", "unregister"]


def custom_op(name, fn=None, /, *, mutates_args, overlappable=None):
    """``fn`` as custom op ``name`` (``"namespace::name"``), mutating its
    arguments named in ``mutates_args`` (at least one; this must be
    accurate: a mutation not named is not respected), each named in
    ``overlappable`` perhaps given the memory of the arguments it maps to
    (see the module). A decorator without ``fn``."""

    def wrap(fn):
        """Wraps `fn` as a `CustomOp`."""
        return CustomOp(name, fn, mutates_args, overlappable)

    return wrap if fn is None else wrap(fn)


class CustomOp:
    """A custom op: call it inside a function ``lumen.compile`` compiles."""

    def __init__(self, name, fn, mutates_args, overlappable=None):
        """Validates `mutates_args` and `overlappable` against `fn`'s signature."""
        mutates = (mutates_args,) if isinstance(mutates_args, str) else tuple(mutates_args)
        overlaps = {
            out: (inputs,) if isinstance(inputs, str) else tuple(inputs)
            for out, inputs in (overlappable or {}).items()
        }
        signature = inspect.signature(fn)
        if not mutates:
            raise ValueError(f"{name}: a custom op returns nothing, so it must mutate an argument (mutates_args)")
        unknown = [a for a in mutates if a not in signature.parameters]
        if unknown:
            raise ValueError(f"{name}: mutates_args names {unknown}, which are not arguments of {fn.__name__}")
        if signature.return_annotation not in (inspect.Signature.empty, None, "None"):
            raise TypeError(f"{name}: a custom op returns nothing (-> None): its results are the arguments it mutates")
        for out, inputs in overlaps.items():
            if out not in mutates:
                raise ValueError(f"{name}: overlappable names {out!r}, which it does not mutate (mutates_args)")
            bad = [i for i in inputs if i not in signature.parameters or i in mutates]
            if bad:
                raise ValueError(f"{name}: {out!r} may overlap arguments it reads, not {bad}")
        self.name, self.mutates_args, self.overlappable = name, mutates, overlaps
        self._fn, self._signature = fn, signature
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        """Records a custom call in the current trace; the mutated arguments take its results.

        Raises unless called while tracing or when a mutated argument isn't a tensor."""
        if not tracer._TRACES:
            raise RuntimeError(f"{self.name}: a custom op runs in a function lumen.compile compiles")
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        names, operands = [], []
        for name, value in bound.arguments.items():
            value = tracer._lift(value)
            if isinstance(value, TracedTensor):
                names.append(name)
                operands.append(value)
            elif isinstance(value, (list, tuple, dict)) and any(
                isinstance(v, TracedTensor) for v in (value.values() if isinstance(value, dict) else value)
            ):
                raise TypeError(f"{self.name}: pass each tensor as an argument of its own, not in {name!r}")
        for name in self.mutates_args:
            if not isinstance(bound.arguments[name], TracedTensor):
                raise TypeError(
                    f"{self.name}: mutated argument {name!r} must be a tensor, got {type(bound.arguments[name]).__name__}"
                )
        arguments, fn, signature = dict(bound.arguments), self._fn, self._signature

        def run(tensors):
            """Calls `fn` with the traced tensor arguments replaced by runtime `tensors`."""
            call = inspect.BoundArguments(signature, {**arguments, **dict(zip(names, tensors))})
            if fn(*call.args, **call.kwargs) is not None:
                raise TypeError("a custom op returns nothing: its results are the arguments it mutates")

        mutated = [names.index(name) for name in self.mutates_args]
        # Overlap pairs as (mutated operand, read operand) positions among the tensor operands.
        overlappable = [
            (names.index(out), names.index(i))
            for out, inputs in self.overlappable.items()
            for i in inputs
            if i in names
        ]
        values = prims._custom_call(operands, self.name, _register_custom_op(run), mutated, overlappable)
        for m, value in zip(mutated, values):
            operands[m].copy_(value)

    def __repr__(self):
        """Names the op and the arguments it mutates."""
        return f"<custom op {self.name} mutating {', '.join(self.mutates_args)}>"


def register(op, device, kernel):
    """Registers `kernel` as the implementation of `op` on `device`; it is called as described by `signature(op)`."""
    _register_kernel(op, device, kernel)


def registered_ops():
    """The ops that accept Python kernels, as ``{name: signature}``."""
    return dict(_registered_ops())


def unregister(op, device):
    """Removes the Python kernel of `op` on `device`, returning whether one was registered."""
    return _unregister_kernel(op, device)


def signature(op):
    """Returns the launcher signature `op`'s Python kernels must follow, or None if `op` takes none."""
    return registered_ops().get(op)
