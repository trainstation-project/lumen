import math

from lumen.autograd.function import Function
from lumen.functional.cross_entropy.utils import _one_hot
from lumen.functional.functions import amax, exp, log, sum
from lumen.graph import prims
from lumen.graph.tracer import _lift, _require_float

__all__ = ["cross_entropy"]


class _CrossEntropy(Function):
    """:func:`cross_entropy`'s rows, differentiated from their logsumexp."""

    @staticmethod
    def forward(ctx, x, target):
        m = amax(x, -1, keepdim=True)
        lse = m.reshape(*target.shape) + log(sum(exp(x - m), -1))
        ctx.save_for_backward(x, target, lse)
        # Each row's logit at its class: element row * n + class of the rows.
        rows, n = math.prod(target.shape), x.shape[-1]
        at = prims.iota(target.dtype, [rows], 0) * n + target.reshape(rows)
        return lse - prims.gather(x.reshape(rows * n), at, 0).reshape(*target.shape)

    @staticmethod
    def backward(ctx, g):
        x, target, lse = ctx.saved_tensors
        p = exp(x - lse.reshape(*target.shape, 1))
        return (p - _one_hot(target, x.shape[-1], x.dtype)) * g.reshape(*target.shape, 1), None


def cross_entropy(input, target):
    """``torch.nn.functional.cross_entropy(input, target, reduction="none")``
    over the last dimension: each row's ``logsumexp(input) - input[target]``
    (``target`` its class, int32 or int64), of ``input``'s shape but that
    dimension, in its dtype; each ``target`` a class, in ``[0, n)``. Traced
    as those primitives, the row's logit read at its class (a gather of the
    rows, one element a row): the MPS compiler runs it as one row kernel (an
    online max and sum of exponentials, a loss a row), writing each row's
    ``logsumexp`` for its gradient, ``(exp(input - logsumexp) -
    one_hot(target)) · g``: a pass over the logits, no reduction (Liger's,
    cut-cross-entropy's); the same kernel's second pass where ``g`` is known
    with the losses."""
    x = _require_float(_lift(input), "cross_entropy")
    target = _lift(target)
    if tuple(target.shape) != tuple(x.shape[:-1]):
        raise ValueError(f"cross_entropy: target of shape {target.shape} for input of shape {x.shape}")
    return _CrossEntropy.apply(x, target)
