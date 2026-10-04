"""The ops, as functions: ``import lumen.functional as F`` (PyTorch's
``torch`` and ``torch.nn.functional`` functions, by the same names).

Each records primitives (``lumen.graph.prims``) into the graph being traced
by ``lumen.compile``, so each takes traced tensors (and Python scalars).
Tensors keep only their operators (``+ - * / @ == < []``, ``-x``), their
layout (``reshape``, ``permute``, ``transpose``, ...) and dtype casts
(``to``, ``float``, ...); every other op is here.

No op changes a dtype the program does not ask for: operands share a dtype
(a Python scalar takes its tensor operand's, and must be of its kind),
floating-point functions take floating-point tensors, and sums and means
reduce in their input's dtype. Convert first with ``.to(dtype)``
(``F.sum(x.float())``).
"""

from lumen.functional.attention import flash_attention, naive_attention
from lumen.functional.functions import *

__all__ = [
    # arithmetic and comparison
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "maximum",
    "minimum",
    "where",
    # elementwise functions
    "exp",
    "log",
    "sqrt",
    "tanh",
    "sigmoid",
    "relu",
    # reductions
    "sum",
    "mean",
    "amax",
    "max",
    # scans
    "cumsum",
    # normalizations
    "softmax",
    "log_softmax",
    "rms_norm",
    # contractions
    "matmul",
    "flash_attention",
    "naive_attention",
]
