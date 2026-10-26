"""``Module``, a frozen-dataclass tree of weights, and helpers that walk, map and hash such trees for ``lumen.compile``."""

import dataclasses

from lumen.tensor._C import Tensor

__all__ = ["Module"]


class Module:
    """A tree of weights: subclasses are frozen dataclasses (no decorator
    needed), whose tensor fields, recursively, are the weights. Their
    scalar fields (annotated ``float``, ``int`` or ``bool``: a learning
    rate, a step count, a flag) are assigned as Python values, outside a
    compiled function or inside it (``self.step = self.step + 1``), where
    they are runtime scalars or static (``lumen.compile``)."""

    def __init_subclass__(cls, **kwargs):
        """Turns each subclass into a frozen dataclass whose scalar fields stay assignable."""
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True, eq=False)(cls)
        frozen = cls.__setattr__

        def __setattr__(self, name, value):
            """Assigns a scalar field after checking its kind; any other field raises as a frozen dataclass does."""
            kind = _scalar_fields(type(self)).get(name)
            if kind is None:
                return frozen(self, name, value)
            object.__setattr__(self, name, _assigned(name, kind, value))

        cls.__setattr__ = __setattr__

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


# Annotations (types or their string forms under postponed evaluation) that mark a scalar field, mapped to its kind.
_KINDS = {float: "float", int: "int", bool: "bool", "float": "float", "int": "int", "bool": "bool"}


def _scalar_fields(cls):
    """``{name: kind}`` of ``cls``'s fields annotated ``float``, ``int`` or
    ``bool``."""
    fields = dataclasses.fields(cls)
    return {f.name: _KINDS[f.type] for f in fields if isinstance(f.type, (type, str)) and f.type in _KINDS}


def is_scalar(v):
    """Whether ``v`` is a scalar a compiled function can take at runtime: a
    Python float, int or bool (not a subclass: an enum is static), or a
    numpy real or bool scalar."""
    if type(v) in (float, int, bool):
        return True
    # Imported lazily so numpy is only loaded when a non-Python scalar is checked.
    import numpy as np

    return isinstance(v, (np.bool_, np.integer, np.floating))


def _assigned(name, kind, value):
    """``value`` assigned to the ``kind`` scalar field ``name``: a Python
    value of the kind (an int as a float), or inside a compiled function a
    0-d traced tensor of it."""
    import numpy as np

    from lumen.graph.tracer import TracedTensor

    # Inside a trace, accept a 0-d tensor whose dtype family matches the field kind.
    if isinstance(value, TracedTensor) and len(value.shape) == 0:
        dtype = value.dtype
        if {"float": "float" in dtype, "int": "int" in dtype, "bool": dtype == "bool"}[kind]:
            return value
    elif kind == "bool" and isinstance(value, (bool, np.bool_)):
        return bool(value)
    elif kind == "int" and isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    elif kind == "float" and isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"{name} is a {kind} field: assign it a {kind} or a 0-d {kind} tensor, got {value!r}")


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


def scalar_paths(x, path=()):
    """The path (field names and indices) to each scalar (``is_scalar``)
    of ``x``, a module, a list or tuple, or a scalar, recursively, in field
    order."""
    if is_scalar(x):
        yield path
    elif isinstance(x, Module):
        for f in dataclasses.fields(x):
            yield from scalar_paths(getattr(x, f.name), (*path, f.name))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from scalar_paths(v, (*path, i))


def at(x, path):
    """The value at ``path`` in ``x`` (as ``scalar_paths`` names it)."""
    for key in path:
        x = getattr(x, key) if isinstance(x, Module) else x[key]
    return x


def assign(x, path, value):
    """Set the value at ``path`` in ``x`` (a module's field or a list's
    element; a tuple's cannot be)."""
    holder, key = at(x, path[:-1]), path[-1]
    if isinstance(holder, Module):
        object.__setattr__(holder, key, value)
    elif isinstance(holder, list):
        holder[key] = value
    else:
        raise TypeError("a scalar in a tuple cannot be assigned: hold it in a module field or a list")


def structure(x, runtime, path=()):
    """``x``'s structure, hashable: its classes, static values and weights'
    types (what a compiled function's plan depends on); a scalar at a path
    ``runtime(path, value)`` holds by its type alone."""
    if isinstance(x, Tensor):
        return ("tensor", x.dtype, tuple(x.shape))
    if isinstance(x, Module):
        fields = dataclasses.fields(x)
        return (type(x), tuple((f.name, structure(getattr(x, f.name), runtime, (*path, f.name))) for f in fields))
    if isinstance(x, (list, tuple)):
        return (type(x), tuple(structure(v, runtime, (*path, i)) for i, v in enumerate(x)))
    if is_scalar(x):
        return ("scalar", type(x)) if runtime(path, x) else ("static", type(x), x)
    return ("static", x)


def map_tensors(x, f, scalar=None, path=()):
    """``x`` with each tensor ``t`` in it (recursively, through modules,
    lists and tuples) replaced by ``f(t)``, and each scalar ``v`` at
    ``path`` by ``scalar(path, v)`` if given."""
    if isinstance(x, Tensor):
        return f(x)
    if scalar is not None and is_scalar(x):
        return scalar(path, x)
    if isinstance(x, Module):
        fields = {
            fl.name: map_tensors(getattr(x, fl.name), f, scalar, (*path, fl.name)) for fl in dataclasses.fields(x)
        }
        return dataclasses.replace(x, **fields)
    if isinstance(x, (list, tuple)):
        return type(x)(map_tensors(v, f, scalar, (*path, i)) for i, v in enumerate(x))
    return x
