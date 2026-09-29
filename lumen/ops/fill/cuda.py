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


class FillCUDAKernel:
    def __init__(
        self, shape: tuple[int, ...], strides: tuple[int, ...], vector_size: int, BLOCK_SIZE: int = 256
    ) -> None:
        self.vector_size = vector_size
        self.BLOCK_SIZE = BLOCK_SIZE
        self.shape = (vector_size, (shape[0] // vector_size, *shape[1:]))
        self.strides = (1, (vector_size * strides[0], *strides[1:]))

    @cute.kernel
    def kernel(self, gT: cute.Tensor, value: cutlass.Numeric, copy_atom: cute.CopyAtom) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        i = BLOCK_ID * self.BLOCK_SIZE + THREAD_ID
        if i < cute.size(gT, mode=[1]):
            rT = cute.make_rmem_tensor(self.vector_size, gT.element_type)
            rT.fill(value)
            cute.copy(copy_atom, rT, gT[(None, i)])

    @cute.jit
    def __call__(self, ptr: cute.Pointer, value: cutlass.Numeric, stream: cuda.CUstream) -> None:
        mX = cute.make_tensor(ptr, cute.make_layout(self.shape, stride=self.strides))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=self.vector_size * mX.element_type.width
        )

        NUM_BLOCKS = (cute.size(mX, mode=[1]) + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        self.kernel(gT=gT, value=value, copy_atom=copy_atom).launch(
            grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream
        )


def _op(dtype: str, shape: tuple[int, ...], strides: tuple[int, ...], vector_size: int) -> Callable:
    element = _ELEMENT[dtype]
    itemsize = max(1, element.width // 8)

    ptr = cute.runtime.nullptr(element, cute.AddressSpace.gmem, assumed_align=vector_size * itemsize)
    kernel = FillCUDAKernel(shape, strides, vector_size)

    return cute.compile(kernel, ptr, element(0), cuda.CUstream(0), options="--enable-tvm-ffi")


register("lumen::fill_", "cuda", _op)
