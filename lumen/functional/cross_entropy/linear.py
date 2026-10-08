from lumen.functional.cross_entropy.logits import cross_entropy
from lumen.functional.functions import matmul
from lumen.graph.tracer import _lift, _require_float

__all__ = ["linear_cross_entropy"]


def linear_cross_entropy(input, weight, target):
    """``cross_entropy(input @ weight.t(), target)``: each row of ``input``
    ``[B, D]``'s ``logsumexp(logits) - logits[target]``, ``[B]`` in float32,
    the logits ``input @ weight.t()`` (``weight`` ``[V, D]``, of ``input``'s
    dtype) accumulated in float32. Traced as those, which the MPS compiler
    matches (``lumen/compiler/mps/linear_cross_entropy.rs``), however
    written, when nothing else reads the logits (not trained: the gradient
    reads them): never stored (Apple's Cut Cross-Entropy: a matmul writing
    each tile's max and sum of exponentials a row, then their
    combination)."""
    h = _require_float(_lift(input), "linear_cross_entropy")
    w = _lift(weight)
    return cross_entropy(matmul(h, w.t(), "float32", "float32"), target)
