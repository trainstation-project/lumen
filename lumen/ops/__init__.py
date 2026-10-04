from lumen._C import _register_kernel, _registered_ops, _unregister_kernel
from lumen.ops.custom_op import CustomOp, custom_op

__all__ = ["CustomOp", "custom_op", "register", "registered_ops", "signature", "unregister"]


def register(op, device, kernel):
    _register_kernel(op, device, kernel)


def registered_ops():
    """The ops that accept Python kernels, as ``{name: signature}``."""
    return dict(_registered_ops())


def unregister(op, device):
    return _unregister_kernel(op, device)


def signature(op):
    return registered_ops().get(op)


from lumen.stream import cuda as _cuda

if _cuda.is_available():
    from lumen.ops.fill import cuda
