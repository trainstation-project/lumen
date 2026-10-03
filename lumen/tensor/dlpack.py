"""DLPack interop, after PyTorch: ``Tensor.__dlpack__`` and
``__dlpack_device__`` (``torch/_tensor.py``) and ``from_dlpack``
(``torch/utils/dlpack.py``). The capsules themselves are built and consumed
in Rust (``lumen/tensor/dlpack.rs``).

lumen's CUDA work runs on the legacy default stream, so that is the stream
a CUDA exchange is ordered against.
"""

import enum

from lumen import _C
from lumen._C import LIBRARY_NAME

__all__ = ["DLDeviceType", "from_dlpack"]


class DLDeviceType(enum.IntEnum):
    """The ``DLDeviceType`` values lumen's devices map to (``dlpack.h``)."""

    kDLCPU = 1
    kDLCUDA = 2
    kDLMetal = 8


_DEVICE_TYPES = {
    "cpu": DLDeviceType.kDLCPU,
    "cuda": DLDeviceType.kDLCUDA,
    "mps": DLDeviceType.kDLMetal,
}


def __dlpack_device__(self):
    """``(device_type, device_id)`` of this tensor's device."""
    device = _C.device(self.device)
    return (_DEVICE_TYPES[device.type], device.index or 0)


def __dlpack__(self, *, stream=-1, max_version=None, dl_device=None, copy=None):
    """This tensor as a DLPack capsule, a view of the same memory.

    ``stream`` is the consumer's CUDA stream. ``None`` or ``1`` is the legacy
    default stream, which lumen's work already runs on, and ``-1`` asks for
    no synchronization (the default, as in PyTorch). For any other stream,
    the stream waits for lumen's queued work, without blocking the host.

    ``max_version`` is the newest DLPack version the consumer reads: 1.0 or
    newer gets a versioned capsule, older or ``None`` the legacy one.
    ``dl_device`` asks for the tensor on another device, which copies unless
    ``copy`` is ``False``; lumen cannot copy on the same device, so
    ``copy=True`` without a new device raises ``BufferError``.
    """
    if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
        if copy is False:
            raise ValueError(
                f"cannot export a tensor on {self.device} to DLPack device " f"{tuple(dl_device)} without copying"
            )
        return __dlpack__(self.to(_device_of(dl_device)), stream=stream, max_version=max_version)
    if copy:
        raise BufferError(f"{LIBRARY_NAME} cannot copy a tensor on the same device when exporting it")

    if stream is not None and not isinstance(stream, int):
        raise TypeError("stream must be ``int`` or ``None``")
    device = _C.device(self.device)
    if device.type == "cuda":
        if stream == 0:
            raise ValueError("stream 0 is ambiguous on CUDA; use 1 for the legacy default stream")
        if stream == 2:
            raise BufferError("the per-thread default stream is not supported")
        if stream not in (None, -1, 1):
            _C._cuda_stream_wait(device.index or 0, stream)
    elif stream not in (None, -1):
        raise ValueError(f"stream must be None or -1 for a tensor on {self.device}")

    if max_version is None or max_version[0] < 1:
        return _C._to_dlpack(self)
    return _C._to_dlpack_versioned(self)


def _device_of(dl_device):
    """The lumen device a ``(device_type, device_id)`` pair names."""
    device_type, device_id = dl_device
    names = {v: k for k, v in _DEVICE_TYPES.items()}
    if device_type not in names:
        raise BufferError(f"unsupported DLPack device type {device_type}")
    name = names[device_type]
    return f"cuda:{device_id}" if name == "cuda" else name


def from_dlpack(ext_tensor):
    """A lumen tensor sharing the memory of ``ext_tensor``: any object with
    ``__dlpack__`` (a NumPy array, a PyTorch tensor, a lumen tensor), or a
    raw DLPack capsule, which is consumed and cannot be imported again."""
    if hasattr(ext_tensor, "__dlpack__"):
        kwargs = {"max_version": (1, 0)}
        device_type, _ = ext_tensor.__dlpack_device__()
        if device_type == DLDeviceType.kDLCUDA:
            # lumen's CUDA work runs on the legacy default stream.
            kwargs["stream"] = 1
        try:
            capsule = ext_tensor.__dlpack__(**kwargs)
        except TypeError:
            # A producer predating DLPack 1.0 takes no max_version.
            del kwargs["max_version"]
            capsule = ext_tensor.__dlpack__(**kwargs)
    else:
        capsule = ext_tensor
    return _C._from_dlpack(capsule)
