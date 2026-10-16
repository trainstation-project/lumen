"""CUDA devices (``torch.cuda``)."""

from lumen._C import Stream, _device_count

__all__ = ["device_count", "is_available", "synchronize"]


def device_count():
    return _device_count("cuda")


def is_available():
    return device_count() > 0


def synchronize(device=None):
    """Wait for all work queued on ``device`` (``"cuda"`` by default)."""
    Stream("cuda" if device is None else device).synchronize()
