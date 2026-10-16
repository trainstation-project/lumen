"""Type stub for the native `lumen._C` extension module.

Each folder that defines bindings (a ``python.rs``) describes them in its own
``_C.pyi`` (``lumen/device/_C.pyi``, ``lumen/tensor/_C.pyi``, ...),
re-exported here; this file has what ``lumen/python.rs`` defines. The per-folder stubs are for type checkers
only: at runtime every name lives in ``lumen._C``.
"""

from lumen.allocator._C import CompilerConfig as CompilerConfig
from lumen.allocator._C import Config as Config
from lumen.allocator._C import config as config
from lumen.device._C import DeviceLike as DeviceLike
from lumen.device._C import device as device
from lumen.graph._C import FUSION_SEPARATOR as FUSION_SEPARATOR
from lumen.graph._C import Graph as Graph
from lumen.graph._C import Plan as Plan
from lumen.graph._C import _register_custom_op as _register_custom_op
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
from lumen.stream._C import Stream as Stream
from lumen.stream._C import _cuda_stream_wait as _cuda_stream_wait
from lumen.stream._C import _mps_buffer as _mps_buffer
from lumen.stream._C import _mps_command_buffer as _mps_command_buffer
from lumen.stream._C import _mps_compile as _mps_compile
from lumen.stream._C import _mps_launch as _mps_launch
from lumen.tensor._C import Scalar as Scalar
from lumen.tensor._C import Tensor as Tensor
from lumen.tensor._C import _from_dlpack as _from_dlpack
from lumen.tensor._C import _pack as _pack
from lumen.tensor._C import _to_dlpack as _to_dlpack
from lumen.tensor._C import _to_dlpack_versioned as _to_dlpack_versioned

__version__: str
LIBRARY_NAME: str
