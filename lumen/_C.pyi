"""Type stub for the native `lumen._C` extension module.

Each folder that defines bindings (a ``python.rs``) describes them in its own
``_C.pyi`` (``lumen/tensor/_C.pyi``, ...), re-exported here; this file has
what ``lumen/python.rs`` defines. The per-folder stubs are for type checkers
only: at runtime every name lives in ``lumen._C``.
"""

from typing import Literal, Optional, Union, final

from lumen.allocator._C import CompilerConfig as CompilerConfig
from lumen.allocator._C import Config as Config
from lumen.allocator._C import config as config
from lumen.graph._C import Graph as Graph
from lumen.graph._C import Plan as Plan
from lumen.ops._C import _dummy_op as _dummy_op
from lumen.ops._C import _register_kernel as _register_kernel
from lumen.ops._C import _registered_ops as _registered_ops
from lumen.ops._C import _unregister_kernel as _unregister_kernel
from lumen.profiler._C import _Profile as _Profile
from lumen.profiler._C import _profiler_cuda_timing as _profiler_cuda_timing
from lumen.profiler._C import _profiler_enabled as _profiler_enabled
from lumen.profiler._C import _profiler_start as _profiler_start
from lumen.profiler._C import _profiler_stop as _profiler_stop
from lumen.profiler._C import _record_function_enter as _record_function_enter
from lumen.profiler._C import _RecordFunction as _RecordFunction
from lumen.safetensors._C import _safetensors_deserialize as _safetensors_deserialize
from lumen.safetensors._C import _safetensors_save_file as _safetensors_save_file
from lumen.safetensors._C import _safetensors_serialize as _safetensors_serialize
from lumen.safetensors._C import safe_open as safe_open
from lumen.stream._C import _cuda_is_available as _cuda_is_available
from lumen.stream._C import _cuda_stream_wait as _cuda_stream_wait
from lumen.stream._C import _cuda_synchronize as _cuda_synchronize
from lumen.stream._C import _mps_synchronize as _mps_synchronize
from lumen.tensor._C import Scalar as Scalar
from lumen.tensor._C import Tensor as Tensor
from lumen.tensor._C import _from_dlpack as _from_dlpack
from lumen.tensor._C import _pack as _pack
from lumen.tensor._C import _to_dlpack as _to_dlpack
from lumen.tensor._C import _to_dlpack_versioned as _to_dlpack_versioned

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
