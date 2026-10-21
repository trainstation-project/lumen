from lumen.graph_compiler._C import CompilerConfig

class Config:
    """Process-wide runtime settings; use the ``lumen.config`` instance."""

    static_allocator_bytes: int

    @property
    def compiler(self) -> CompilerConfig: ...

config: Config
