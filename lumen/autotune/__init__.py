"""Kernel autotuning: the kernel configs a ``use`` block forces in place of the autotuner's pick."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Union

from lumen.autotune import _C
from lumen.autotune._C import (
    CudaAttentionConfig,
    CudaGemmConfig,
    TritonAttentionConfig,
    TritonGemmConfig,
    TritonLoopConfig,
    _pop,
    _push,
)

# The Metal kernels' configs, in a build with Metal (None in one without).
MpsAttentionConfig = getattr(_C, "MpsAttentionConfig", None)
MpsElementwiseConfig = getattr(_C, "MpsElementwiseConfig", None)
MpsGemmConfig = getattr(_C, "MpsGemmConfig", None)

__all__ = [
    "CudaAttentionConfig",
    "CudaGemmConfig",
    "MpsAttentionConfig",
    "MpsElementwiseConfig",
    "MpsGemmConfig",
    "TritonAttentionConfig",
    "TritonGemmConfig",
    "TritonLoopConfig",
    "use",
]

KernelConfig = Union[
    CudaAttentionConfig, CudaGemmConfig, TritonAttentionConfig, TritonGemmConfig, TritonLoopConfig, object
]


@contextlib.contextmanager
def use(*configs: KernelConfig) -> Iterator[None]:
    """Programs compiled inside the block run every kernel of a family given a config here with that config,
    untimed, whether or not ``lumen.config.compiler.autotune`` is on, at any size (the CUDA GEMM: every matmul).
    An inner block's config of a family replaces an outer one's; plans compiled inside are rebuilt once it
    exits."""
    _push(list(configs))
    try:
        yield
    finally:
        _pop()
