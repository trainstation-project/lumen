"""Contraction by axis name (``lumen.named``): ``dot`` and ``contract`` of named tensors, resolved while tracing
into ``dot_general`` primitives. Names themselves live on the tensors (``lumen.Tensor.rename``,
``TracedTensor.names``)."""

from lumen.named.ops import contract, dot

__all__ = ["dot", "contract"]
