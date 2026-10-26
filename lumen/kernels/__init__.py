"""Registers Python kernels for ops that accept them, keyed by op name and device."""

from lumen.kernels._C import _register_kernel, _registered_ops, _unregister_kernel

__all__ = ["register", "registered_ops", "signature", "unregister"]


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
