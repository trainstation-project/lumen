from typing import Optional

from lumen.device._C import DeviceLike

class CompilerConfig:
    """A device compiler's settings: the flags every device's compiler has
    (each device's class adds its own)."""

    merge_dots: bool = True
    flash_attention: bool = True
    fused_linear_cross_entropy_chunk_size: Optional[int] = None
    memory_limit: int = 0
    dump_to: Optional[str] = None

    def reset(self) -> None:
        """This device's settings back to their defaults."""

class CpuConfig(CompilerConfig):
    """The CPU compiler's settings (``lumen.config.compiler.cpu``)."""

class CudaConfig(CompilerConfig):
    """The CUDA compiler's settings (``lumen.config.compiler.cuda``)."""

    use_triton_backend: bool = False
    autotune: bool = True
    safe_kernels: bool = False

class MpsConfig(CompilerConfig):
    """The MPS compiler's settings (``lumen.config.compiler.mps``)."""

    fuse: bool = True
    normalization_diamonds: bool = True
    reduction_epilogues: bool = True
    contraction_epilogues: bool = True
    multi_output_fusion: bool = True
    horizontal_fusion: bool = True
    online_softmax: bool = True
    fused_linear_cross_entropy: bool = True
    split_k: bool = True
    deterministic: bool = False
    neural_engine: bool = False
    row_cache: int = 8
    safe_kernels: bool = False
    autotune: bool = True

class TpuConfig(CompilerConfig):
    """The TPU compiler's settings (``lumen.config.compiler.tpu``)."""

    normalization_diamonds: bool = True
    contraction_epilogues: bool = True
    horizontal_fusion: bool = True
    safe_kernels: bool = False

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
