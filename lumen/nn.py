import dataclasses

from lumen.tensor._C import Tensor

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
            if id(t) not in seen:
                seen.add(id(t))
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
    tuple, or a tensor), recursively: inside a compiled function, its
    traced weights (not its floats, runtime scalars)."""
    from lumen.graph.tracer import TracedTensor

    prefix = f"{path}." if path else ""
    if isinstance(x, Tensor) or (isinstance(x, TracedTensor) and not x.weak):
        yield path, x
    elif isinstance(x, Module):
        for f in dataclasses.fields(x):
            yield from _leaves(getattr(x, f.name), prefix + f.name)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _leaves(v, prefix + str(i))


def _set_float(x, k, value):
    """Set the ``k``-th float of ``x`` (in :func:`_floats`' order) to
    ``value``: a module's field or a list's element (a tuple's cannot be)."""
    slots = list(_float_slots(x))
    holder, key = slots[k]
    if holder is None:
        raise TypeError("a float in a tuple cannot be assigned: hold it in a module field or a list")
    if isinstance(holder, Module):
        object.__setattr__(holder, key, value)
    else:
        holder[key] = value


def _float_slots(x):
    """Where each float of ``x`` is, ``(holder, key)``, in :func:`_floats`'
    order (``(None, None)`` in a tuple)."""
    if isinstance(x, Module):
        items = [(x, f.name, getattr(x, f.name)) for f in dataclasses.fields(x)]
    elif isinstance(x, (list, tuple)):
        items = [(x if isinstance(x, list) else None, i, v) for i, v in enumerate(x)]
    else:
        return
    for holder, key, v in items:
        if isinstance(v, float):
            yield (holder, key) if holder is not None else (None, None)
        else:
            yield from _float_slots(v)


def _floats(x):
    """Each float field of ``x`` (a module, a list or tuple), recursively, in
    field order: runtime scalars of a compiled function (``lumen.compile``),
    not baked into it."""
    if isinstance(x, float):
        yield x
    elif isinstance(x, Module):
        for f in dataclasses.fields(x):
            yield from _floats(getattr(x, f.name))
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _floats(v)


def structure(x):
    """``x``'s structure, hashable: its classes, static fields (not floats,
    which are runtime scalars) and weights' types (what a compiled
    function's plan depends on)."""
    if isinstance(x, Tensor):
        return ("tensor", x.dtype, tuple(x.shape))
    if isinstance(x, Module):
        return (type(x), tuple((f.name, structure(getattr(x, f.name))) for f in dataclasses.fields(x)))
    if isinstance(x, (list, tuple)):
        return (type(x), tuple(map(structure, x)))
    if isinstance(x, float):
        return ("scalar",)
    return ("static", x)


def map_tensors(x, f, scalar=None):
    """``x`` with each tensor ``t`` in it (recursively, through modules,
    lists and tuples) replaced by ``f(t)``, and each float ``v`` by
    ``scalar(v)`` if given."""
    if isinstance(x, Tensor):
        return f(x)
    if isinstance(x, float) and scalar is not None:
        return scalar(x)
    if isinstance(x, Module):
        fields = {fl.name: map_tensors(getattr(x, fl.name), f, scalar) for fl in dataclasses.fields(x)}
        return dataclasses.replace(x, **fields)
    if isinstance(x, (list, tuple)):
        return type(x)(map_tensors(v, f, scalar) for v in x)
    return x
