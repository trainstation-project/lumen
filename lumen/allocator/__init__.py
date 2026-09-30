"""Allocator settings: ``lumen.config``, process-wide runtime options for
the device allocators (bindings in ``lumen/allocator/python.rs``).

Each GPU device allocates from one allocator, reserved on its first allocation;
``lumen.config.static_allocator_bytes`` sets its size (default 1 GiB).
"""

from lumen._C import config

__all__ = ["config"]
