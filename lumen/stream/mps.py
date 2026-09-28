"""``lumen.mps``, modeled on ``torch.mps``."""

from lumen import _C

__all__ = ["synchronize"]


def synchronize() -> None:
    """Wait for all work on the MPS stream to finish
    (``torch.mps.synchronize()``). Raises ``RuntimeError`` without MPS."""
    _C._mps_synchronize()
