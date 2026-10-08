from lumen.functional.cross_entropy.chunked import _ChunkedLinearCrossEntropy
from lumen.functional.cross_entropy.logits import cross_entropy
from lumen.functional.functions import matmul
from lumen.graph.tracer import _lift, _require_float, config

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
    combination). With ``lumen.config.compiler.fused_linear_cross_entropy_chunk_size``,
    a chunk of that many rows at a time, its gradients computed with its
    losses (XMA's chunked fused linear cross entropy, for training: the
    logits of a chunk at a time in memory)."""
    h = _require_float(_lift(input), "linear_cross_entropy")
    w = _lift(weight)
    chunk = config.compiler.fused_linear_cross_entropy_chunk_size
    if chunk is not None:
        return _ChunkedLinearCrossEntropy.apply(h, w, _lift(target), chunk=chunk)
    return cross_entropy(matmul(h, w.t(), "float32", "float32"), target)
