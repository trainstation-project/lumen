from __future__ import annotations

from typing import Any, Optional, Sequence, final

from lumen.tensor._C import Tensor

@final
class MetalKernel:
    """A Metal kernel of your own, for a custom op to launch into lumen's
    MPS stream."""

    def __new__(cls, source: str, name: Optional[str] = None) -> MetalKernel: ...
    @property
    def name(self) -> str: ...
    def launch(
        self,
        tensors: Sequence[Tensor],
        args: Sequence[Any] = (),
        *,
        grid: Optional[Sequence[int]] = None,
        threadgroups: Optional[Sequence[int]] = None,
    ) -> None: ...
