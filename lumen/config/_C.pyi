from typing import Optional

class CompilerConfig:
    """The compiler's settings (``lumen.config.compiler``): one set every
    device's compiler reads, each backend the flags it has a use for."""

    merge_dots: bool = True
    flash_attention: bool = True
    fused_linear_cross_entropy_chunk_size: Optional[int] = None
    memory_limit: int = 0
    dump_to: Optional[str] = None
    preserve_precision: bool = False
    use_triton_backend: bool = False
    fuse: bool = True
    autotune: bool = True
    safe_kernels: bool = False
    horizontal_fusion: bool = True
    multi_output_fusion: bool = True
    split_k: bool = True
    linear_reordering: bool = False
    normalization_diamonds: bool = True
    fused_linear_cross_entropy: bool = True
    deterministic: bool = False
    cuda_graphs: bool = True
    reduction_epilogues: bool = True
    contraction_epilogues: bool = True
    online_softmax: bool = True
    neural_engine: bool = False
    row_cache: int = 8

    def reset(self) -> None:
        """Every setting back to its default."""

class Config:
    """Process-wide runtime settings; use the ``lumen.config`` instance."""

    static_allocator_bytes: int

    @property
    def compiler(self) -> CompilerConfig: ...

config: Config
