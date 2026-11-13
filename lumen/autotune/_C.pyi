from typing import Optional

class CudaAttentionConfig:
    """A config of the CUDA backend's SM80 flash-attention forward."""

    def __init__(self, tile_m: int, tile_n: int, stages: int = 1, warps: int = 4, q_in_regs: bool = False) -> None: ...
    @staticmethod
    def candidates() -> list[CudaAttentionConfig]:
        """Every config the autotuner times, in its order."""

    def fits(self, h: int, hv: int) -> bool:
        """Whether the config runs head dims ``h`` (Q and K) and ``hv`` (V)."""
    tile_m: int
    tile_n: int
    stages: int
    warps: int
    q_in_regs: bool

class CudaGemmConfig:
    """A tile of the CUDA backend's SM80 GEMM: ``stages`` 0 for as many as fit."""

    def __init__(self, tile_m: int, tile_n: int, tile_k: int, warps: int, stages: int = 0) -> None: ...
    @staticmethod
    def candidates() -> list[CudaGemmConfig]:
        """Every config the autotuner times, in its order."""
    tile_m: int
    tile_n: int
    tile_k: int
    warps: int
    stages: int

class TritonGemmConfig:
    """A tile of the Triton backend's GEMM."""

    def __init__(
        self, block_m: int, block_n: int, block_k: int, stages: int, warps: int, group_m: int = 8
    ) -> None: ...
    @staticmethod
    def candidates(row: Optional[int] = None) -> list[TritonGemmConfig]:
        """Every config the autotuner times for a plain GEMM, or for one with a row epilogue over rows ``row``
        wide."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    warps: int
    group_m: int

class TritonAttentionConfig:
    """A tile of the Triton backend's flash attention."""

    def __init__(self, block_m: int, block_n: int, stages: int, warps: int) -> None: ...
    @staticmethod
    def candidates() -> list[TritonAttentionConfig]:
        """Every config the autotuner times, in its order."""
    block_m: int
    block_n: int
    stages: int
    warps: int

class TritonLoopConfig:
    """The block and warps of the Triton backend's loop kernels (elementwise, row, reduction and scan)."""

    def __init__(self, block_size: int, warps: int) -> None: ...
    @staticmethod
    def candidates(lanes: Optional[int] = None) -> list[TritonLoopConfig]:
        """Every config the autotuner times, or those it times for a kernel of ``lanes``."""
    block_size: int
    warps: int

class MpsGemmConfig:
    """One Metal matmul kernel variant: ``algorithm`` "tiled", "mpp", "skinny" or "gemv"."""

    def __init__(
        self,
        bm: int,
        bn: int,
        bk: int,
        threads: int,
        cols: int = 1,
        prefetch: bool = False,
        algorithm: str = "tiled",
    ) -> None: ...
    @staticmethod
    def candidates(algorithm: Optional[str] = None) -> list[MpsGemmConfig]:
        """Every config the autotuner can time, or those of ``algorithm``."""
    bm: int
    bn: int
    bk: int
    threads: int
    cols: int
    prefetch: bool
    algorithm: str

class MpsAttentionConfig:
    """How the Metal flash-attention forward runs: ``mpp`` on Metal Performance Primitives, its simdgroups a
    threadgroup and keys a block."""

    def __init__(self, mpp: bool, groups: int, keys: int) -> None: ...
    @staticmethod
    def candidates() -> list[MpsAttentionConfig]:
        """Every config the autotuner can time."""

    def fits(self, h: int, hv: int, dtype: str) -> bool:
        """Whether the config runs head dims ``h`` (Q and K) and ``hv`` (V) of ``dtype``."""
    mpp: bool
    groups: int
    keys: int

class MpsElementwiseConfig:
    """The launch of a Metal elementwise kernel: ``per_thread`` elements a thread in threadgroups of ``group``."""

    def __init__(self, per_thread: int, group: int, contiguous: bool = False) -> None: ...
    @staticmethod
    def candidates(itemsize: int = 4) -> list[MpsElementwiseConfig]:
        """Every launch the autotuner times for an output of ``itemsize``-byte elements."""
    per_thread: int
    group: int
    contiguous: bool

def _push(configs: list[object]) -> None: ...
def _pop() -> None: ...
