"""``lumen.cuda``, modeled on ``torch.cuda``."""

from typing import Optional, Union

from lumen import _C
from lumen.device import device as _device

__all__ = ["is_available", "synchronize"]


def is_available() -> bool:
    """Whether a CUDA device can hold tensors in this build
    (``torch.cuda.is_available()``)."""
    return _C._cuda_is_available()


def synchronize(device: Optional[Union[int, str, _device]] = None) -> None:
    """Wait for all work on a CUDA device to finish
    (``torch.cuda.synchronize(device)``): an index, a CUDA device, or
    ``None`` for device 0. Raises ``RuntimeError`` without that device."""
    device = _device(f"cuda:{device or 0}" if device is None or isinstance(device, int) else device)
    if device.type != "cuda":
        raise ValueError(f"Expected a cuda device, but got: {device}")
    _C.Stream(device).synchronize()
