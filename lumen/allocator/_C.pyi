"""The part of `lumen._C` that ``lumen/allocator/python.rs`` defines (see
``lumen/_C.pyi``)."""

class CompilerConfig:
    """The graph compilers' flags; use the ``lumen.config.compiler`` instance.
    Read when a graph is compiled; the defaults run every program exactly as
    traced, but for ``online_softmax``, ``flash_attention`` and ``split_k``
    (on: set them off for softmax, attention and dots as traced); kernels may
    add atomically, in no fixed order, unless ``deterministic``;
    ``neural_engine`` (off) runs large float16 dots of fixed weights on the
    Apple Neural Engine (on MPS), which accumulates them narrower than
    float32."""

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
