"""Type stub for the native `lumen._C` extension module."""

from typing import Literal, Optional, Union

__version__: str

class device:
    """A device type plus an optional index, modeled on ``torch.device``."""

    def __init__(self, device: Union[str, "device"], index: Optional[int] = None) -> None: ...
    @property
    def type(self) -> Literal["cpu", "mps", "cuda"]: ...
    @property
    def index(self) -> Optional[int]: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Config:
    """Process-wide runtime settings; use the ``lumen.config`` instance."""

    memory_caching: bool

config: Config
