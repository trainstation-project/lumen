from lumen.functional.functions import eq
from lumen.graph import prims


def _one_hot(target, n, dtype):
    """``target``'s classes of ``n`` as ``dtype`` rows of zeros and a one."""
    hit = eq(target.reshape(*target.shape, 1), prims.iota(target.dtype, [n], 0))
    return prims.cast(hit, dtype)
