"""``lumen.cuda``, modeled on ``torch.cuda``."""

from lumen import _C

__all__ = ["is_available"]


def is_available() -> bool:
    """Whether a CUDA device can hold tensors in this build
    (``torch.cuda.is_available()``)."""
    return _C._cuda_is_available()
