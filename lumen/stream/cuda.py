"""``lumen.cuda``, modeled on ``torch.cuda``."""

from typing import Optional, Union

from lumen import _C

__all__ = ["synchronize"]


def synchronize(device: Optional[Union[int, str, "_C.device"]] = None) -> None:
    """Wait for all work on a CUDA device to finish
    (``torch.cuda.synchronize(device)``): an index, a CUDA device, or
    ``None`` for device 0. Raises ``RuntimeError`` without that device."""
    _C._cuda_synchronize(device)
