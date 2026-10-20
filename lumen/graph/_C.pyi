from typing import Any, Optional, Sequence

from lumen.device._C import DeviceLike
from lumen.tensor._C import Tensor

FUSION_SEPARATOR: str
"""What a fusion's label joins its primitives' names with (``mul → tanh → add``)."""

def _register_custom_op(f: Any) -> int:
    """Keep custom op function ``f`` (called with its operands as tensors
    when its step runs): its handle (``lumen.custom_op``)."""

class Graph:
    """A graph of primitive ops, built by tracing (``lumen/graph/``)."""

    def __init__(self) -> None: ...
    def input(self, dtype: str, shape: Sequence[int]) -> int: ...
    def apply(self, name: str, inputs: Sequence[int], params: Optional[dict[str, Any]] = None) -> int: ...
    def type_of(self, var: int) -> tuple[str, list[int]]: ...
    def inputs(self) -> list[int]: ...
    def outputs(self) -> list[int]: ...
    def attentions(self) -> list[dict[str, Any]]:
        """The attentions traced so far, as the compiler matches them: each
        dot's operands, output, the softmax's max and sum, the score chain,
        the causal offset and the values it computes."""

    def prune(self) -> list[int | None]:
        """Remove the nodes no output depends on; each value's new number,
        or None if removed."""

    def precision_warnings(self) -> list[tuple[int, int, str]]:
        """A warning for each dot or sum rounded to a narrower dtype than it
        accumulates in, then cast back up: its output, the cast's, the
        message."""

    def nodes(self) -> list[dict[str, Any]]:
        """Each node: ``primitive``, ``text`` (with its parameters),
        ``fusion`` (``kernel``, ``body``, Metal ``source``; or None),
        ``inputs`` and ``output`` values, ``scope`` (its ``record_function``
        ranges) and ``sources`` (the ``(file, line)``s that traced it)."""

    def set_outputs(self, outputs: Sequence[int]) -> None: ...
    def run(self, inputs: Sequence[Tensor]) -> list[Tensor]: ...
