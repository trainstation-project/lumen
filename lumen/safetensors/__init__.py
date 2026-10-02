"""Tensors to and from safetensors files (``lumen/safetensors/``), with the
API of ``safetensors.torch``::

    from lumen.safetensors import save_file, load_file, safe_open

    save_file({"w": w, "b": b}, "model.safetensors", metadata={"format": "lumen"})
    tensors = load_file("model.safetensors", device="mps")
    with safe_open("model.safetensors", device="mps") as f:
        w = f.get_tensor("w")

Files are byte-for-byte those of the reference implementation (Rust port
in ``lumen/safetensors/mod.rs``). Loading reads each tensor's bytes straight
into its storage (on the CPU, or on MPS, whose memory the host shares).
A file of a dtype lumen does not have (float8, ...) raises ``ValueError``.
"""

import os

from lumen._C import Tensor
from lumen._C import _safetensors_deserialize, _safetensors_save_file, _safetensors_serialize
from lumen._C import safe_open

__all__ = ["save", "save_file", "load", "load_file", "safe_open"]


def _flatten(tensors):
    """``tensors`` as ``(name, tensor)`` pairs, checked as safetensors.torch
    does: a dict of tensors, none sharing storage (they would be written
    twice, and loaded as separate tensors). Non-contiguous tensors are
    rejected when written."""
    if not isinstance(tensors, dict):
        raise ValueError(f"Expected a dict of [str, lumen.Tensor] but received {type(tensors)}")
    by_storage = {}
    for name, t in tensors.items():
        if not isinstance(t, Tensor):
            raise ValueError(f"Key `{name}` is invalid, expected lumen.Tensor but received {type(t)}")
        by_storage.setdefault((str(t.device), t.storage_id), []).append(name)
    shared = [names for names in by_storage.values() if len(names) > 1]
    if shared:
        raise RuntimeError(
            f"Some tensors share memory, which would be written twice and loaded as separate tensors: {shared}."
            " Save one tensor of each storage."
        )
    return list(tensors.items())


def save(tensors, metadata=None):
    """``tensors`` (``{name: lumen.Tensor}``, contiguous, any device) as the
    bytes of a safetensors file, with optional string ``metadata``."""
    return _safetensors_serialize(_flatten(tensors), metadata)


def save_file(tensors, filename, metadata=None):
    """Write ``tensors`` to ``filename`` as a safetensors file (through a
    temporary file renamed over it, so readers never see it half-written)."""
    _safetensors_save_file(_flatten(tensors), os.fspath(filename), metadata)


def load(data, device="cpu"):
    """The tensors in safetensors bytes ``data``, as ``{name: tensor}`` on
    ``device``."""
    return dict(_safetensors_deserialize(data, device))


def load_file(filename, device="cpu"):
    """The tensors in safetensors file ``filename``, as ``{name: tensor}`` on
    ``device``."""
    with safe_open(os.fspath(filename), device=device) as f:
        return dict(f.get_tensors())
