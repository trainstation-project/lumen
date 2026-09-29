import functools
import math
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from lumen.ops import register

_ELEMENT = {
    "bool": cutlass.Uint8,
    "uint8": cutlass.Uint8,
    "int8": cutlass.Int8,
    "uint16": cutlass.Uint16,
    "int16": cutlass.Int16,
    "uint32": cutlass.Uint32,
    "int32": cutlass.Int32,
    "uint64": cutlass.Uint64,
    "int64": cutlass.Int64,
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
    "float64": cutlass.Float64,
}


@functools.cache
def _resident_threads() -> int:
    """How many threads the GPU runs at once: SMs x threads per SM, for the
    current device (lumen's CUDA nodes have one kind of GPU)."""
    err, device = cuda.cuCtxGetDevice()
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, device = cuda.cuDeviceGet(0)
    attribute = cuda.CUdevice_attribute
    _, sms = cuda.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
    _, threads = cuda.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device)
    return sms * threads


class _FillCUDAKernel:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        vector_size: int,
        BLOCK_SIZE: int = 256,
    ) -> None:
        self.dtype = dtype
        self.vector_size = vector_size
        self.BLOCK_SIZE = BLOCK_SIZE
        self.shape = (vector_size, (shape[0] // vector_size, *shape[1:]))
        self.strides = (1, (vector_size * strides[0], *strides[1:]))
        self.num_vectors = math.prod(self.shape[1])
        # Persistent: no more blocks than the GPU runs at once, each looping
        # over the vectors a grid apart.
        self.NUM_BLOCKS = min(math.ceil(self.num_vectors / BLOCK_SIZE), _resident_threads() // BLOCK_SIZE)

    @cute.kernel
    def kernel(
        self, gX: cute.Tensor, value: cutlass.Numeric, copy_atom: cute.CopyAtom
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        rX = cute.make_rmem_tensor(self.vector_size, self.dtype)
        rX.fill(value)

        i = BLOCK_ID * self.BLOCK_SIZE + THREAD_ID
        while i < self.num_vectors:
            cute.copy(copy_atom, rX, gX[(None, i)])
            i += self.NUM_BLOCKS * self.BLOCK_SIZE

    @cute.jit
    def __call__(self, ptr: cute.Pointer, value: cutlass.Numeric, stream: cuda.CUstream) -> None:
        mX = cute.make_tensor(ptr, cute.make_layout(self.shape, stride=self.strides))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=self.vector_size * self.dtype.width
        )

        self.kernel(gX=mX, value=value, copy_atom=copy_atom).launch(
            grid=(self.NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream
        )


def _op(dtype: str, shape: tuple[int, ...], strides: tuple[int, ...], vector_size: int) -> Callable:
    element = _ELEMENT[dtype]

    ptr = cute.runtime.nullptr(element, cute.AddressSpace.gmem, assumed_align=vector_size * element.width // 8)
    kernel = _FillCUDAKernel(element, shape, strides, vector_size)

    return cute.compile(kernel, ptr, element(0), cuda.CUstream(0), options="--enable-tvm-ffi")


register("lumen::fill_", "cuda", _op)
