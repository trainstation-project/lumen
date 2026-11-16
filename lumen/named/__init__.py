"""Named tensors (``lumen.named``): ``NamedTensor`` and ops that take axis names instead of positions,
broadcasting by name, resolved into positional ``lumen.functional`` ops while tracing."""

from lumen.named.tensor import NamedTensor

# isort: off
from lumen.named.ops import *
from lumen.named.ops import __all__ as _ops

# isort: on

__all__ = ["NamedTensor", *_ops]
