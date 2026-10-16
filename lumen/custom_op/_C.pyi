"""The part of `lumen._C` that ``lumen/custom_op/python.rs`` defines (see
``lumen/_C.pyi``)."""

from typing import Any, Optional, Sequence, final

from lumen.tensor._C import Tensor

@final
class Kernel:
    """A Metal kernel of your own, for a custom op to launch into lumen's
    MPS stream."""

    def __new__(cls, source: str, name: Optional[str] = None) -> "Kernel": ...
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
