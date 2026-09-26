"""lumen: a minimal tensor library prototype.

A pure-Python package over the native Rust extension `lumen._C` — the
same relationship `torch` has to `torch._C`.

* ``lumen.device`` — a device, modeled on ``torch.device``.
* ``lumen.config`` — process-wide runtime settings, e.g.
  ``lumen.config.memory_caching = False`` to bypass the caching allocator
  (PyTorch: ``PYTORCH_NO_CUDA_MEMORY_CACHING=1``).
"""

from lumen._C import __version__, config, device

__all__ = ["__version__", "config", "device"]
