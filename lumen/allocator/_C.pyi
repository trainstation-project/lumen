"""The part of `lumen._C` that ``lumen/allocator/python.rs`` defines (see
``lumen/_C.pyi``)."""

class Config:
    """Process-wide runtime settings; use the ``lumen.config`` instance."""

    static_allocator_bytes: int

config: Config
