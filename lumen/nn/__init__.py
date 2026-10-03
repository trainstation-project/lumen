"""Modules: models as trees of weights (``lumen/nn/``), after Equinox's
``eqx.Module``.

A ``Module`` subclass is a frozen dataclass whose fields are its weights
(tensors), its submodules, lists or tuples of those, or static
configuration (anything else: ints, strings, ...). Weights are meta tensors:
``lumen.compile`` places them on its device when a function taking the
module compiles, and every function taking it shares that memory::

    class MLP(lumen.nn.Module):
        w1: lumen.Tensor
        w3: lumen.Tensor
        w2: lumen.Tensor

        def __call__(self, x):
            return ((x @ self.w1).relu() * (x @ self.w3)) @ self.w2

    model = MLP(*(lumen.empty(s, device="meta") for s in shapes))
    f = lumen.compile(lambda model, x: model(x), device="mps")
    f(model, lumen.empty([seq, dim], device="meta"))  # compile: place the weights
    model.load_state_dict(lumen.safetensors.load_file("mlp.safetensors"))
    y = f(model, x)  # the weights read in place; x copied in
"""

import dataclasses

from lumen._C import Tensor

__all__ = ["Module"]


class Module:
    """A tree of weights: subclasses are frozen dataclasses (no decorator
    needed), whose tensor fields, recursively, are the weights."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True, eq=False)(cls)

    def named_parameters(self):
        """``(path, weight)`` for each weight, recursively, in field order
        (``layers.0.w1``); a tensor in several fields is listed once."""
        seen = set()
        for path, t in _leaves(self, ""):
            if t.storage_id not in seen:
                seen.add(t.storage_id)
                yield path, t

    def parameters(self):
        """Each weight, as ``named_parameters``."""
        return [t for _, t in self.named_parameters()]

    def load_state_dict(self, tensors, strict=True):
        """Copy ``tensors`` (``{path: tensor}``, as ``named_parameters``
        names them; a safetensors file's) into the weights' memory, which a
        compiled function taking the module has placed. With ``strict``, every
        weight must be given, and nothing else."""
        params = dict(self.named_parameters())
        if strict:
            missing, unexpected = params.keys() - tensors.keys(), tensors.keys() - params.keys()
            if missing or unexpected:
                raise KeyError(f"load_state_dict: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
        for path, t in tensors.items():
            if path in params:
                params[path].copy_(t)


def _leaves(x, path):
    """``(path, tensor)`` for each tensor in ``x`` (a module, a list or
    tuple, or a tensor), recursively."""
    prefix = f"{path}." if path else ""
    if isinstance(x, Tensor):
        yield path, x
    elif isinstance(x, Module):
        for f in dataclasses.fields(x):
            yield from _leaves(getattr(x, f.name), prefix + f.name)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _leaves(v, prefix + str(i))


def structure(x):
    """``x``'s structure, hashable: its classes, static fields and weights'
    types (what a compiled function's plan depends on)."""
    if isinstance(x, Tensor):
        return ("tensor", x.dtype, tuple(x.shape))
    if isinstance(x, Module):
        return (type(x), tuple((f.name, structure(getattr(x, f.name))) for f in dataclasses.fields(x)))
    if isinstance(x, (list, tuple)):
        return (type(x), tuple(map(structure, x)))
    return ("static", x)


def map_tensors(x, f):
    """``x`` with each tensor ``t`` in it (recursively, through modules,
    lists and tuples) replaced by ``f(t)``."""
    if isinstance(x, Tensor):
        return f(x)
    if isinstance(x, Module):
        return dataclasses.replace(x, **{fl.name: map_tensors(getattr(x, fl.name), f) for fl in dataclasses.fields(x)})
    if isinstance(x, (list, tuple)):
        return type(x)(map_tensors(v, f) for v in x)
    return x
