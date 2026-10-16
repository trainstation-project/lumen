"""The part of `lumen._C` that ``lumen/stream/python.rs`` defines (see
``lumen/_C.pyi``)."""

from typing import Any, Optional, Sequence, Union, final

from lumen.device import device
from lumen.tensor._C import Tensor

@final
class Stream:
    """A device's stream, modeled on ``torch.Stream``: the queue its ops
    submit work to without waiting."""

    def __new__(cls, device: Optional[Union[str, device]] = None, *, stream_id: int = 0) -> "Stream": ...
    @property
    def device(self) -> device: ...
    @property
    def stream_id(self) -> int: ...
    def synchronize(self) -> None: ...
    def wait_stream(self, other: "Stream") -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
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
