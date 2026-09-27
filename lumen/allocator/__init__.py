"""Allocator settings: ``lumen.config``, process-wide runtime options for
the device caching allocators (bindings in ``lumen/allocator/python.rs``).

``lumen.config.memory_caching = False`` bypasses the cache, like PyTorch's
``PYTORCH_NO_CUDA_MEMORY_CACHING=1``.
"""

from lumen._C import config

__all__ = ["config"]
