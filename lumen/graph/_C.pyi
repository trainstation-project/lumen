"""The part of `lumen._C` that ``lumen/graph/python.rs`` defines (see
``lumen/_C.pyi``)."""

from typing import Any, Optional, Sequence

from lumen._C import DeviceLike
from lumen.tensor._C import Tensor

FUSION_SEPARATOR: str
"""What a fusion's label joins its primitives' names with (``mul → tanh → add``)."""

class Graph:
    """A graph of primitive ops, built by tracing (``lumen/graph/``)."""

    def __init__(self) -> None: ...
    def input(self, dtype: str, shape: Sequence[int]) -> int: ...
    def apply(self, name: str, inputs: Sequence[int], params: Optional[dict[str, Any]] = None) -> int: ...
    def type_of(self, var: int) -> tuple[str, list[int]]: ...
    def inputs(self) -> list[int]: ...
    def outputs(self) -> list[int]: ...
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
        ``inputs`` and ``output`` values."""

    def set_outputs(self, outputs: Sequence[int]) -> None: ...
    def run(self, inputs: Sequence[Tensor]) -> list[Tensor]: ...

class Plan:
    """A graph compiled for execution by ``device``'s graph compiler (on
    MPS, with its elementwise ops fused), with its memory planned."""

    def __init__(
        self,
        graph: Graph,
        device: Optional[DeviceLike] = None,
        fuse: bool = True,
        donate: Sequence[int] = (),
        parameters: Optional[Sequence[int]] = None,
        packable: Sequence[int] = (),
    ) -> None:
        """Fused where ``device`` fuses (unless ``fuse`` is false), with outputs
        written into the inputs at positions ``donate`` where they fit. With
        ``parameters`` (input positions), an executable that owns its memory
        (``run_in``), those inputs its parameters, the ``packable`` ones of
        them packed into blocks where dots merge (``packed``)."""

    @property
    def workspace_bytes(self) -> int: ...
    @property
    def packed(self) -> list[tuple[list[int], int]]:
        """The inputs the compiler added after the graph's: each the block
        of these parameter inputs side by side along a dimension."""

    def run_in(self, workspace: Tensor, inputs: Sequence[Tensor]) -> list[Tensor]:
        """Run in ``workspace`` (uint8, at least ``workspace_bytes``): inputs
        that are not parameters copied in, parameters read in place; the
        outputs are views valid until the next run in it."""

    def steps(self) -> list[dict[str, Any]]:
        """Each step, as ``Graph.nodes``, with its ``label`` (what the
        profiler calls it: the primitive's name, or ``3x dot_general`` for a
        merged dot), ``inputs`` and ``output`` as ``(buffer, dtype, shape)``,
        ``extra_outputs`` likewise (a multi-output fusion's other outputs), ``views`` as each input's ``(element
        offset, strides)`` in its buffer if it reads it as a strided view (a
        slice; else None), and its kernel's ``scratch`` in the workspace as
        ``(offset, bytes)`` (or None)."""

    def run(self, inputs: Sequence[Tensor], device: Optional[DeviceLike] = None) -> list[Tensor]:
        """Run on ``device`` (where the inputs must be), or else the inputs'
        device (the CPU without inputs)."""
