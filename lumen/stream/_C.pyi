"""The part of `lumen._C` that ``lumen/stream/python.rs`` defines (see
``lumen/_C.pyi``)."""

from typing import Any, Optional, Sequence, Union, final

from lumen.device import device
from lumen.tensor._C import Tensor

@final
class Stream:
    """A device's stream, modeled on ``torch.Stream``: the queue its ops
    submit work to without waiting."""

    def __new__(cls, device: Optional[Union[str, device]] = None) -> "Stream": ...
    @property
    def device(self) -> str: ...
    def synchronize(self) -> None: ...
    def compile(self, source: str, name: Optional[str] = None) -> "Kernel": ...
    def launch(
        self,
        kernel: "Kernel",
        tensors: Sequence[Tensor],
        args: Sequence[Any] = (),
        *,
        grid: Optional[Sequence[int]] = None,
        threadgroups: Optional[Sequence[int]] = None,
    ) -> None: ...
    def command_buffer(self) -> int: ...
    def buffer(self, tensor: Tensor) -> tuple[int, int]: ...

@final
class Kernel:
    """A Metal kernel of your own, compiled by an MPS stream."""

    @property
    def name(self) -> str: ...

def _cuda_stream_wait(index: int, stream: int) -> None: ...
