"""Type stub for the native `lumen._C` extension module.

Each folder that defines bindings (a ``python.rs``) describes them in its own
``_C.pyi`` (``lumen/tensor/_C.pyi``, ...), re-exported here; this file has
what ``lumen/python.rs`` defines. The per-folder stubs are for type checkers
only: at runtime every name lives in ``lumen._C``.
"""

from typing import Literal, Optional, Union, final

from lumen.allocator._C import Config as Config, config as config
from lumen.graph._C import Graph as Graph, Plan as Plan
from lumen.ops._C import (
    _dummy_op as _dummy_op,
    _register_kernel as _register_kernel,
    _registered_ops as _registered_ops,
    _unregister_kernel as _unregister_kernel,
)
from lumen.profiler._C import (
    _Profile as _Profile,
    _profiler_cuda_timing as _profiler_cuda_timing,
    _profiler_enabled as _profiler_enabled,
    _profiler_start as _profiler_start,
    _profiler_stop as _profiler_stop,
    _record_function_enter as _record_function_enter,
    _RecordFunction as _RecordFunction,
)
from lumen.safetensors._C import (
    _safetensors_deserialize as _safetensors_deserialize,
    _safetensors_save_file as _safetensors_save_file,
    _safetensors_serialize as _safetensors_serialize,
    safe_open as safe_open,
)
from lumen.stream._C import (
    _cuda_is_available as _cuda_is_available,
    _cuda_stream_wait as _cuda_stream_wait,
    _cuda_synchronize as _cuda_synchronize,
    _mps_synchronize as _mps_synchronize,
)
from lumen.tensor._C import (
    Scalar as Scalar,
    Tensor as Tensor,
    _from_dlpack as _from_dlpack,
    _pack as _pack,
    _to_dlpack as _to_dlpack,
    _to_dlpack_versioned as _to_dlpack_versioned,
)

__version__: str
LIBRARY_NAME: str

# A device argument (stub-only: there is no such name at runtime).
DeviceLike = Union[str, "device"]

@final
class device:
    """A device type plus an optional index, modeled on ``torch.device``."""

    def __new__(cls, device: Union[str, "device"], index: Optional[int] = None) -> "device": ...
    @property
    def type(self) -> Literal["cpu", "mps", "cuda", "meta"]: ...
    @property
    def index(self) -> Optional[int]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
