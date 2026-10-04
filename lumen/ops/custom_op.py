"""Custom ops (PyTorch's ``torch.library.custom_op``): a function
``lumen.compile`` calls but never traces into. The compilers see one opaque
step reading its tensor arguments and writing in place those it names in
``mutates_args``: it is never fused, simplified, merged with another call
or looked into, and runs as written, in order with the other steps.

A custom op returns nothing: its results are the arguments it mutates.
Within a traced function, each mutated argument reads its new value after
the call, as if assigned with ``copy_``: so do its views, and a weight or a
tensor argument so mutated is written back after the call.

An argument it mutates may be declared to overlap tensor arguments it
reads (``overlappable={"out": "x"}``, or several, ``{"out": ("x", "y")}``):
the function may be called with ``out`` in ``x``'s memory, so it must read
each element of ``x`` before writing ``out``'s at that index, and not need
``out``'s old value (it may be ``x``'s). Whether it is is the compiler's
choice: ``out`` is its own memory where it can be (nothing reads its old
value after the call), else ``x``'s where ``x`` dies at the call (of
``out``'s type, read nowhere after it), saving the memory and the copy of
``out`` (XLA: a custom call's output-operand aliasing).

When the step runs, the function is called with its tensor arguments as
lumen tensors on the plan's device (views of the plan's memory, no copy)
and its other arguments as given: it writes the mutated ones in place. On
MPS, nothing waits for the device: its own kernels, encoded into lumen's
stream (``lumen.mps.launch``, ``lumen.mps.command_buffer``), run in order
with the plan's, after the earlier steps and before the later ones; host
reads and writes (``lumen.to_numpy``, ``copy_``, DLPack) wait for the
stream themselves. Host work is done by the time it returns. A mutated
argument is the argument's own memory where nothing reads its old value
after the call, else a copy of it (PyTorch: ``auto_functionalized``,
reinplaced where it can be).

    kernel = lumen.mps.compile(AXPY_METAL_SOURCE)  # kernel void axpy(...)

    @lumen.ops.custom_op("mylib::axpy", mutates_args=("y",))
    def axpy(a: float, x: lumen.Tensor, y: lumen.Tensor) -> None:
        lumen.mps.launch(kernel, [x, y], [np.float32(a)], grid=(y.numel,))

    @lumen.compile
    def step(x, y):
        axpy(2.0, x, y)
        return y * 1.0
"""

import functools
import inspect

from lumen._C import _register_custom_op
from lumen.graph import prims, tracer
from lumen.graph.tracer import TracedTensor

__all__ = ["CustomOp", "custom_op"]


def custom_op(name, fn=None, /, *, mutates_args, overlappable=None):
    """``fn`` as custom op ``name`` (``"namespace::name"``), mutating its
    arguments named in ``mutates_args`` (at least one; this must be
    accurate: a mutation not named is not respected), each named in
    ``overlappable`` perhaps given the memory of the arguments it maps to
    (see the module). A decorator without ``fn``."""

    def wrap(fn):
        return CustomOp(name, fn, mutates_args, overlappable)

    return wrap if fn is None else wrap(fn)


class CustomOp:
    """A custom op: call it inside a function ``lumen.compile`` compiles."""

    def __init__(self, name, fn, mutates_args, overlappable=None):
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
        if not tracer._TRACES:
            raise RuntimeError(f"{self.name}: a custom op runs in a function lumen.compile compiles")
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        # Its tensors, the operands, in argument order; the rest, constants.
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
            call = inspect.BoundArguments(signature, {**arguments, **dict(zip(names, tensors))})
            if fn(*call.args, **call.kwargs) is not None:
                raise TypeError("a custom op returns nothing: its results are the arguments it mutates")

        mutated = [names.index(name) for name in self.mutates_args]
        # Each (mutated, read) operand pair that may overlap; arguments not
        # tensors in this call have no memory to share.
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
        return f"<custom op {self.name} mutating {', '.join(self.mutates_args)}>"
