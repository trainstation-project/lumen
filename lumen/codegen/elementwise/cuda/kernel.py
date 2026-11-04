"""CUDA elementwise kernel: the fixed grid-strided walk of an elementwise fusion, its work a generated ``BODY``.

``BODY(ins, outs, j, SEGMENT)`` computes and stores every output of segment ``SEGMENT`` (the outputs of one
element count) at flat index ``j``, reading the step's input tensors ``ins`` and writing its outputs ``outs``,
each a flat view indexed at its own strides."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

# Threads per block.
THREADS = 256
# Bytes of its narrowest tensor a thread computes a step.
STEP_BYTES = 16


class ElementwiseKernel:
    """An elementwise fusion: `body` computes its outputs at an index; `segments` are the element counts of
    its output groups, each walked by its own grid-strided loop."""

    def __init__(self, body, segments):
        """`body` is the generated ``BODY``, `segments` a tuple of element counts."""
        self.body = body
        self.segments = segments

    @cute.kernel
    def kernel(self, ins: list, outs: list):
        """Walks each segment's indices grid-strided: whole tiles of `THREADS * V` elements with no bounds
        checks, each thread `V` elements `THREADS` apart (`STEP_BYTES` of its narrowest tensor, each of a
        warp's loads reading consecutive elements), then the tail element by element. Indices are 32-bit
        when they reach every element."""
        THREAD_ID, _, _ = cute.arch.thread_idx()
        BLOCK_ID, _, _ = cute.arch.block_idx()
        NUM_BLOCKS, _, _ = cute.arch.grid_dim()

        # Booleans are 1-bit types stored in a byte.
        narrowest = min(max(t.element_type.width, 8) for t in list(ins) + list(outs))
        V = max(STEP_BYTES * 8 // narrowest, 1)
        tile = THREADS * V

        for SEGMENT in cutlass.range_constexpr(len(self.segments)):
            n = self.segments[SEGMENT]
            Index = Int32 if n < 2**31 - tile else Int64
            tiles = n // tile
            step = Index(NUM_BLOCKS)

            t = Index(BLOCK_ID)

            while t < tiles:
                base = t * tile + Index(THREAD_ID)

                for v in cutlass.range_constexpr(V):
                    self.body(ins, outs, base + v * THREADS, SEGMENT)

                t += step

            # The tail past the last whole tile, an element a thread.
            if cutlass.const_expr(tiles * tile < n):
                j = Index(tiles * tile) + Index(BLOCK_ID) * THREADS + Index(THREAD_ID)

                while j < n:
                    self.body(ins, outs, j, SEGMENT)
                    j += step * THREADS

    @cute.jit
    def launch(self, ins: list, outs: list, blocks: Int32, stream: cuda.CUstream):
        """Runs `kernel` on `blocks` blocks of `THREADS` threads on `stream`."""
        self.kernel(ins, outs).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1], stream=stream)
