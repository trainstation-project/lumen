"""``lumen.ops``: registering kernels written in Python.

The dispatcher picks a kernel per device (see ``lumen/ops/mod.rs``). A
backend kernel is normally Rust, but a device kernel may be supplied from
Python instead — for a CUDA kernel authored with
`CuTe DSL <https://github.com/NVIDIA/cutlass>`_::

    import lumen

    def compile(dtype, shape, strides):
        ...  # compile the CuTe DSL kernel for this layout

        def launch(address, value, stream):
            ...  # launch it on the tensor at `address`

        return launch

    lumen.ops.register("lumen::fill_", "cuda", compile)
    t = lumen.zeros((4,), device="cuda")
    t.fill_(1.0)  # compiles once for this layout, then launches

Only the kernel is Python. A registered kernel is a compile hook: Rust
calls it once per dtype and layout (``shape`` and ``strides`` in elements;
``fill_`` hands over the dimensions in stride order), caches the launcher
it returns, and calls that with the data pointer, the value in the tensor's
dtype, and the CUDA stream (0, the legacy default). A launcher that is a
TVM-FFI function (``apache-tvm-ffi``; the CuTe DSL compiles one with
``--enable-tvm-ffi``) is called from Rust through its C ABI, so a launch
runs no Python; any other callable is called through Python.

The registered kernel is called *before* the built-in one for that device,
so a built-in remains the fallback when no Python kernel is registered.
CUDA's ``fill_`` has no built-in: it is ``lumen/ops/fill/cuda.py``,
registered when lumen is imported on a machine with a CUDA device.
Registering for the CPU is refused: every tensor op would then go through
Python.

Registering by op *name* keeps this frontend general — it does not import
one binding per op. ``lumen.ops.registered_ops()`` lists the names that are
actually consulted by Rust, and the signatures of the hook and launcher;
registering anything else raises ``KeyError``.
"""

from lumen._C import _register_kernel, _registered_ops, _unregister_kernel

__all__ = ["register", "registered_ops", "signature", "unregister"]


def register(op, device, kernel):
    """Use ``kernel`` as op ``op``'s kernel for ``device``.

    ``op`` is a dispatcher op name (see :func:`registered_ops`), ``device``
    is anything a tensor's ``device=`` accepts (``"cuda"``, ``"cuda:1"``,
    ``lumen.device(...)``), and ``kernel`` is a compile hook,
    ``kernel(dtype, shape, strides) -> launch(address, value, stream)``.

    Raises ``KeyError`` for an op that takes no Python kernels,
    ``ValueError`` if ``kernel`` is not callable, and ``RuntimeError`` if
    the device is unavailable or the CPU.
    """
    _register_kernel(op, device, kernel)


def registered_ops():
    """The ops that accept Python kernels, as ``{name: signature}``."""
    return dict(_registered_ops())


def unregister(op, device):
    """Drop the Python kernel for ``op`` on ``device``, so the built-in one
    runs again. Returns whether a kernel was registered. Used by tests,
    which must not leak kernels into each other in a process-wide
    registry."""
    return _unregister_kernel(op, device)


def signature(op):
    """The call signature for ``op``, or ``None`` if it takes none."""
    return registered_ops().get(op)


from lumen.stream import cuda as _cuda

if _cuda.is_available():
    from lumen.ops.fill import cuda
