import pathlib
import re

import lumen

KERNELS = pathlib.Path(lumen.__file__).parent / "kernels"

DTYPES = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "f64": "float64",
    "bool": "bool",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}
SHORT = {v: k for k, v in DTYPES.items()}


def type_text(dtype, shape):
    return f"{SHORT.get(dtype, dtype)}[{','.join(map(str, shape))}]"


_SOURCES = [
    (r"^matmul_atomic_", "dot_general/mps/kernels.metal", "#define MATMUL_ATOMIC"),
    (r"^lce_partials", "dot_general/mps/kernels.metal", "#define LCE_PARTIALS"),
    (r"^lce_combine", "dot_general/mps/kernels.metal", "kernel void lce_combine"),
    (r"^dynamic_slice_copy_", "dynamic_slice/mps/kernels.metal", "#define DS_COPY"),
    (r"^dynamic_update_slice_", "dynamic_slice/mps/kernels.metal", "#define DYNAMIC_UPDATE_SLICE"),
    (r"^dynamic_slice_", "dynamic_slice/mps/kernels.metal", "#define DYNAMIC_SLICE"),
    (r"^scatter_add_scan_", "gather/mps/kernels.metal", "#define SCATTER_ADD_SCAN"),
    (r"^scatter_add_", "gather/mps/kernels.metal", "#define SCATTER_ADD("),
    (r"^random_bits", "factory/mps/kernels.metal", "kernel void random_bits"),
    (r"^matmul_(small|mid)_", "dot_general/mps/kernels.metal", "inline void matmul_sg_impl"),
    (r"^matmul_(f16|bf16|f32)(_f32_(f16|bf16|f32))?$", "dot_general/mps/kernels.metal", "inline void matmul_sg_impl"),
    (r"^matmul_(u64|i64)$", "dot_general/mps/kernels.metal", "inline void matmul_wide_impl"),
    (r"^matmul_", "dot_general/mps/kernels.metal", "inline void matmul_impl"),
    (r"^reduce_\w+_rows_", "reduce/mps/kernels.metal", "inline void reduce_rows"),
    (r"^reduce_\w+_cols_", "reduce/mps/kernels.metal", "inline void reduce_cols"),
    (r"^reduce_\w+_grouped_", "reduce/mps/kernels.metal", "inline void reduce_grouped"),
    (r"^reduce_", "reduce/mps/kernels.metal", "inline void reduce("),
    (r"^gather_", "layout/mps/kernels.metal", "#define GATHER"),
    (r"^transpose_", "layout/mps/kernels.metal", "#define TRANSPOSE"),
    (r"^fill_\d$", "factory/mps/kernels.metal", "#define FULL"),
    (r"^iota_", "factory/mps/kernels.metal", "#define IOTA"),
]


def source_of(kernel):
    """``(file, anchor)`` of a primitive kernel's source, or None."""
    for pattern, path, anchor in _SOURCES:
        if re.match(pattern, kernel):
            return path, anchor
    return None


def _traced_at(sources):
    lines = []
    for file, line in sources:
        try:
            code = pathlib.Path(file).read_text().splitlines()[line - 1].strip()
        except (OSError, UnicodeDecodeError, IndexError):
            code = ""
        lines.append(f"{file}:{line}  {code}")
    return lines
