from typing import Literal, Optional, Protocol, Union, final

DeviceLike = Union[str, "device"]

@final
class device:
    """A device type plus an optional index, modeled on ``torch.device``."""

    def __new__(cls, device: Union[str, "device"], index: Optional[int] = None) -> "device": ...
    @property
    def type(self) -> Literal["cpu", "mps", "cuda", "tpu", "trainium", "meta"]: ...
    @property
    def index(self) -> Optional[int]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class _DeviceModule(Protocol):
    """A device type's module (``lumen.cuda``, PyTorch's ``torch.cuda``)."""

    def device_count(self) -> int: ...
    def is_available(self) -> bool: ...
    def synchronize(self, device: Optional[DeviceLike] = None) -> None: ...

cpu: _DeviceModule
cuda: _DeviceModule
meta: _DeviceModule
mps: _DeviceModule
tpu: _DeviceModule
trainium: _DeviceModule
