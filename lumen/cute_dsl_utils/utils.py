import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from lumen.cute_dsl_utils.constants import get_cute_dtype


def get_powers_of_2(start: int, end: int) -> list[int]:
    """The powers of two from ``start`` to ``end``, inclusive."""
    assert start > 0 and start & (start - 1) == 0, "start is not a power of 2"
    assert end > 0 and end & (end - 1) == 0, "end is not a power of 2"

    output = []
    n = start
    while n <= end:
        output.append(n)
        n <<= 1

    return output


def get_fake_cute_tensor(dtype: str, shape: tuple[int], divisibility: int = 1, leading_dim: int = -1) -> cute.Tensor:
    """A compile-time stand-in for a real tensor of ``dtype`` and ``shape``.

    The strides are partly symbolic (everything but the leading dimension),
    so one compiled kernel serves every conforming shape.
    """
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim

    element = get_cute_dtype(dtype)
    stride = tuple(1 if i == leading_dim else cute.sym_int64(divisibility=divisibility) for i in range(len(shape)))

    return cute.runtime.make_fake_tensor(
        element,
        shape,
        stride=stride,
        assumed_align=divisibility * (element.width // 8),
    )


def element_bytes(dtype: str) -> int:
    """The size in bytes of one ``dtype`` element."""
    return get_cute_dtype(dtype).width // 8


def get_alignment(t: object) -> int:
    """The largest power of two from 4 to 16 dividing ``t``'s address.

    A kernel that loads ``assumed_align`` bytes at a time needs the base
    address to be a multiple of that; promising less than the tensor really
    has costs throughput, promising more is undefined.
    """
    address = t.data_ptr()

    alignment = 4
    if element_bytes(t.dtype) >= 4:
        for i in get_powers_of_2(4, 16):
            if address % i != 0:
                break
            alignment = i

    return alignment


def tensor_to_cute_tensor(t: object, leading_dim: int) -> cute.Tensor:
    """``t`` as a CuTe tensor over the same buffer, with a dynamic layout.

    The layout is dynamic because a fill kernel is compiled once and run on
    views of any shape and stride; ``leading_dim`` is the one dimension whose
    stride is known at compile time (1, or 0 for a broadcast dimension, which
    is reported as no leading dimension at all).
    """
    if leading_dim < 0:
        leading_dim += t.ndim

    t = from_dlpack(t, assumed_align=get_alignment(t))

    # A stride of 0 is a broadcast dimension: there is no leading dimension.
    if t.stride[leading_dim] == 0:
        leading_dim = None

    return t.mark_layout_dynamic(leading_dim=leading_dim)
