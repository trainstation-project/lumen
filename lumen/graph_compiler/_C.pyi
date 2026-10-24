from typing import Any, Optional, Sequence

from lumen.device._C import DeviceLike
from lumen.graph._C import Graph
from lumen.tensor._C import Tensor

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
        scalars: Sequence[int] = (),
        donate_into: Sequence[tuple[int, int]] = (),
    ) -> None:
        """Fused where ``device`` fuses (unless ``fuse`` is false), with outputs
        written into the inputs at positions ``donate`` where they fit, and
        output ``k`` alone into input ``i`` for each ``(i, k)`` of
        ``donate_into`` where it fits. With
        ``parameters`` (input positions), an executable that owns its memory
        (``run_in``), those inputs its parameters, the ``packable`` ones of
        them packed into blocks where dots merge (``packed``)."""

    @property
    def workspace_bytes(self) -> int: ...
    @property
    def workspace_alignment(self) -> int: ...
    @property
    def neural_engine(self) -> bool:
        """Whether it runs steps on the Apple Neural Engine (Core ML
        programs, ``lumen.config.compiler.mps.neural_engine``)."""

    def _invalidate_neural_engine(self) -> None:
        """Compile its Neural Engine steps' programs again on their next
        run, from their weights' values then."""

    @property
    def packed(self) -> list[tuple[list[int], int]]:
        """The inputs the compiler added after the graph's: each the block
        of these parameter inputs side by side along a dimension."""

    @property
    def interleaved(self) -> list[list[int]]:
        """Each of ``packed``'s blocks' regions, in order along its
        dimension: K consecutive parameters interleaved (element ``j`` of the
        ``i``-th at ``j K + i``: SwiGLU's gate and up), 1 a parameter as it
        is."""

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
        ``(offset, bytes)`` (or None), and ``sources``: the ``(file, line)``
        of each line of the traced program it computes (a fusion's
        members')."""

    def run(self, inputs: Sequence[Tensor], device: Optional[DeviceLike] = None) -> list[Tensor]:
        """Run on ``device`` (where the inputs must be), or else the inputs'
        device (the CPU without inputs)."""
