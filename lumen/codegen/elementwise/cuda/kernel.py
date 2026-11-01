"""CUDA elementwise kernel: the fixed grid-strided walk of an elementwise fusion, its work a generated ``BODY``.

``BODY(ins, outs, j, SEGMENT)`` computes and stores every output of segment ``SEGMENT`` (the outputs of one
element count) at flat index ``j``, reading the step's input tensors ``ins`` and writing its outputs ``outs``,
each a flat view indexed at its own strides."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

# Threads per block, and elements each thread computes per step of its walk (independent loads in flight).
THREADS = 256
UNROLL = 4


class ElementwiseKernel:
    """An elementwise fusion: `body` computes its outputs at an index; `segments` are the element counts of
    its output groups, each walked by its own grid-strided loop."""

    def __init__(self, body, segments):
        """`body` is the generated ``BODY``, `segments` a tuple of element counts."""
        self.body = body
        self.segments = segments

    @cute.kernel
    def kernel(self, ins: list, outs: list):
        """Walks each segment's indices grid-strided, `UNROLL` a step, computing the outputs at each."""
        THREAD, _, _ = cute.arch.thread_idx()
        BLOCK, _, _ = cute.arch.block_idx()
        BLOCKS, _, _ = cute.arch.grid_dim()
        STEP = Int64(BLOCKS) * THREADS

        for SEGMENT in cutlass.range_constexpr(len(self.segments)):
            n = self.segments[SEGMENT]
            i = Int64(BLOCK) * THREADS + Int64(THREAD)

            while i < n:
                for u in cutlass.range_constexpr(UNROLL):
                    j = i + Int64(u) * STEP

                    if j < n:
                        self.body(ins, outs, j, SEGMENT)

                i += STEP * UNROLL

    @cute.jit
    def launch(self, ins: list, outs: list, blocks: Int32, stream: cuda.CUstream):
        """Runs `kernel` on `blocks` blocks of `THREADS` threads on `stream`."""
        self.kernel(ins, outs).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1], stream=stream)
