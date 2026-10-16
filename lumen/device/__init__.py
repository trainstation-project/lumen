"""Devices: ``lumen.device`` (modeled on ``torch.device``), the native
``lumen._C.device`` (bindings in ``lumen/device/python.rs``).

Note that ``lumen.device`` itself is the class, as in PyTorch; import it
from ``lumen`` (or ``from lumen.device import device``).
"""

from lumen._C import device

__all__ = ["device"]
