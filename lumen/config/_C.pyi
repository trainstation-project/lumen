from typing import Optional

from lumen.device._C import DeviceLike

class CpuConfig:
    """The CPU compiler's settings (``lumen.config.compiler.cpu``)."""

    merge_dots: bool
    flash_attention: bool
    fused_linear_cross_entropy_chunk_size: Optional[int]
    memory_limit: int
    dump_to: Optional[str]

    def reset(self) -> None: ...

class CudaConfig:
    """The CUDA compiler's settings (``lumen.config.compiler.cuda``)."""

    merge_dots: bool
    flash_attention: bool
    fused_linear_cross_entropy_chunk_size: Optional[int]
    memory_limit: int
    dump_to: Optional[str]
    use_triton_backend: bool
    autotune: bool

    def reset(self) -> None: ...

class MpsConfig:
    """The MPS compiler's settings (``lumen.config.compiler.mps``)."""

    merge_dots: bool
    flash_attention: bool
    fused_linear_cross_entropy_chunk_size: Optional[int]
    memory_limit: int
    dump_to: Optional[str]
    fuse: bool
    normalization_diamonds: bool
    reduction_epilogues: bool
    contraction_epilogues: bool
    multi_output_fusion: bool
    horizontal_fusion: bool
    online_softmax: bool
    fused_linear_cross_entropy: bool
    split_k: bool
    deterministic: bool
    neural_engine: bool
    row_cache: int
    safe_kernels: bool
    autotune: bool

    def reset(self) -> None: ...

class TpuConfig:
    """The TPU compiler's settings (``lumen.config.compiler.tpu``)."""

    merge_dots: bool
    flash_attention: bool
    fused_linear_cross_entropy_chunk_size: Optional[int]
    memory_limit: int
    dump_to: Optional[str]
    normalization_diamonds: bool
    contraction_epilogues: bool
    horizontal_fusion: bool
    safe_kernels: bool

    def reset(self) -> None: ...

class CompilerConfigs:
    """Each device's compiler settings (``lumen.config.compiler``)."""

    @property
    def cpu(self) -> CpuConfig: ...
    @property
    def cuda(self) -> CudaConfig: ...
    @property
    def mps(self) -> MpsConfig: ...
    @property
    def tpu(self) -> TpuConfig: ...
    def of(self, device: DeviceLike) -> CpuConfig | CudaConfig | MpsConfig | TpuConfig:
        """The settings of ``device``'s compiler (the meta device's, the CPU's)."""

    def reset(self) -> None:
        """Every device's settings back to their defaults."""

class Config:
    """Process-wide runtime settings; use the ``lumen.config`` instance."""

    static_allocator_bytes: int

    @property
    def compiler(self) -> CompilerConfigs: ...

config: Config
