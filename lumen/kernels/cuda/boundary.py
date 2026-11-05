"""Bounds predicates for the partial last tile of a CuTe elementwise kernel."""

import cutlass.cute as cute
from cutlass import Boolean, range_constexpr


@cute.jit
def lane_boundary(gC: cute.Tensor, tiled_copy: cute.TiledCopy, block_coord, THREAD_ID: int, shape: cute.Shape):
    """This thread's copy slice of tile `block_coord`, its per-element in-bounds mask and whether the
    whole slice is in bounds (the mask is only filled when it isn't)."""
    thr_copy = tiled_copy.get_slice(THREAD_ID)

    bC = gC[block_coord]
    tC = thr_copy.partition_S(bC)

    rC = cute.make_rmem_tensor(tC.shape, Boolean)
    # Elements are ordered, so the slice is fully in bounds iff its last coordinate is.
    is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

    if not is_within_boundary:
        for i in range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

    return thr_copy, rC, is_within_boundary
