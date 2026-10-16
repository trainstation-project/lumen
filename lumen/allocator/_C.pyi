class CompilerConfig:
    fuse: bool
    merge_dots: bool
    normalization_diamonds: bool
    reduction_epilogues: bool
    contraction_epilogues: bool
    multi_output_fusion: bool
    horizontal_fusion: bool
    online_softmax: bool
    flash_attention: bool
    split_k: bool
    deterministic: bool
    neural_engine: bool
    row_cache: int
    memory_limit: int

    def reset(self) -> None: ...

class Config:
    """Process-wide runtime settings; use the ``lumen.config`` instance."""

    static_allocator_bytes: int

    @property
    def compiler(self) -> CompilerConfig: ...

config: Config
