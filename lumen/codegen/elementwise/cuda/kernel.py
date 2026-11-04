"""CUDA elementwise kernel: the fixed grid-strided walk of an elementwise fusion, its work a generated ``BODY``.

``BODY(ins, outs, fins, fouts, j, v, SEGMENT)`` computes and stores every output of segment ``SEGMENT`` (the
outputs of one element count) at flat index ``j``, reading the step's input tensors ``ins`` and writing its
outputs ``outs``, each a flat view (built from the pointer the kernel takes) indexed at its own strides. The tensors it reads or writes at ``j`` itself
it reads from ``fins`` and writes to ``fouts`` at ``v``: register fragments of a thread's consecutive
elements, moved with vector loads and stores, or the tensors themselves with ``v = j``."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

# Threads per block.
THREADS = 256
# Bytes of its widest tensor a thread moves a step: one 128-bit load or store.
STEP_BYTES = 16


def vectors(t, V):
    """Flat tensor `t` as its whole vectors of `V` consecutive elements, the vector index second."""
    return cute.make_tensor(t.iterator, cute.make_layout((V, cute.size(t) // V), stride=(1, V)))


class ElementwiseKernel:
    """An elementwise fusion: `body` computes its outputs at an index; `segments` are the element counts of
    its output groups, each walked by its own grid-strided loop; `types` and `spans` the CuTe type each
    tensor (inputs, then outputs; a bool's a byte) is stored as and the elements it spans from its pointer; `fragments_in` and `fragments_out` list
    per segment the inputs and outputs `body` reads and writes at the walk's own index; with `vectorized`
    (every tensor 16-byte aligned) those move a thread's vector of consecutive elements at a time in the
    segments `contiguous` marks (reading every input in row-major order)."""

    def __init__(self, body, segments, types, spans, fragments_in, fragments_out, contiguous, vectorized):
        """`body` is the generated ``BODY``, `segments` and `spans` tuples of element counts, `types` of CuTe
        types, `fragments_in` and
        `fragments_out` a tuple per segment of tensor positions, `contiguous` a flag per segment, and
        `vectorized` whether the fragments of contiguous segments move by vectors."""
        self.body = body
        self.segments = segments
        self.types = types
        self.spans = spans
        self.fragments_in = fragments_in
        self.fragments_out = fragments_out
        self.contiguous = contiguous
        self.vectorized = vectorized

    @cute.kernel
    def kernel(self, ins: list, outs: list):
        """Over the tensors at pointers `ins` and `outs`, walks each segment's indices grid-strided: whole tiles of `THREADS * V` elements with no bounds
        checks, then the tail element by element. In a tile each thread takes `V` elements (`STEP_BYTES` of
        its widest tensor): consecutive ones, its fragment tensors moved by one vector load or store
        each, when vectorized; else ones `THREADS` apart, read in place. Indices are 32-bit when they reach
        every element."""
        THREAD_ID, _, _ = cute.arch.thread_idx()
        BLOCK_ID, _, _ = cute.arch.block_idx()
        NUM_BLOCKS, _, _ = cute.arch.grid_dim()

        # Pointers keep only their storage type, so retype each (the body reads unsigned ints, which stay
        # signless, by bitcasts).
        flat = [
            cute.make_tensor(cute.recast_ptr(p, dtype=dtype), cute.make_layout(span))
            for p, dtype, span in zip(list(ins) + list(outs), self.types, self.spans)
        ]
        ins, outs = flat[: len(ins)], flat[len(ins) :]

        widest = max(t.element_type.width for t in list(ins) + list(outs))
        V = max(STEP_BYTES * 8 // widest, 1)
        tile = THREADS * V

        for SEGMENT in cutlass.range_constexpr(len(self.segments)):
            n = self.segments[SEGMENT]
            Index = Int32 if n < 2**31 - tile else Int64
            tiles = n // tile
            step = Index(NUM_BLOCKS)

            t = Index(BLOCK_ID)

            while t < tiles:
                if cutlass.const_expr(self.vectorized and self.contiguous[SEGMENT]):
                    # The thread's vector: elements `q * V` to `q * V + V - 1`.
                    q = t * THREADS + Index(THREAD_ID)
                    fins = list(ins)
                    fouts = list(outs)

                    for k in self.fragments_in[SEGMENT]:
                        fins[k] = cute.make_rmem_tensor((V,), ins[k].element_type)
                        fins[k].store(vectors(ins[k], V)[(None, q)].load())

                    for k in self.fragments_out[SEGMENT]:
                        fouts[k] = cute.make_rmem_tensor((V,), outs[k].element_type)

                    for v in cutlass.range_constexpr(V):
                        self.body(ins, outs, fins, fouts, q * V + v, v, SEGMENT)

                    for k in self.fragments_out[SEGMENT]:
                        vectors(outs[k], V)[(None, q)].store(fouts[k].load())
                else:
                    base = t * tile + Index(THREAD_ID)

                    for v in cutlass.range_constexpr(V):
                        j = base + v * THREADS
                        self.body(ins, outs, ins, outs, j, j, SEGMENT)

                t += step

            # The tail past the last whole tile, an element a thread.
            if cutlass.const_expr(tiles * tile < n):
                j = Index(tiles * tile) + Index(BLOCK_ID) * THREADS + Index(THREAD_ID)

                while j < n:
                    self.body(ins, outs, ins, outs, j, j, SEGMENT)
                    j += step * THREADS

    @cute.jit
    def launch(self, ins: list, outs: list, blocks: Int32, stream: cuda.CUstream):
        """Runs `kernel` on the tensors at pointers `ins` and `outs` on `blocks` blocks of `THREADS` threads
        on `stream`."""
        self.kernel(ins, outs).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1], stream=stream)
