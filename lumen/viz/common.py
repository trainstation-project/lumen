"""Helpers shared by the visualization tools: dtype names and kernel source lookup."""

import pathlib
import re

import lumen

LUMEN = pathlib.Path(lumen.__file__).parent
KERNELS = LUMEN / "kernels"

# Short dtype names used in visualizations, mapped to lumen dtype names.
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
    """Compact type string such as ``f32[64,128]``."""
    return f"{SHORT.get(dtype, dtype)}[{','.join(map(str, shape))}]"


# (kernel name pattern, source file under lumen/, line prefix where its code starts); first match wins.
_SOURCES = [
    (r"^matmul_atomic_", "codegen/dot_general/mps/tiled/matmul_atomic.metal", "#define MATMUL_ATOMIC"),
    (
        r"^lce_partials",
        "codegen/dot_general/mps/linear_cross_entropy/linear_cross_entropy.metal",
        "#define LCE_PARTIALS",
    ),
    (
        r"^lce_combine",
        "codegen/dot_general/mps/linear_cross_entropy/linear_cross_entropy.metal",
        "kernel void lce_combine",
    ),
    (r"^dynamic_slice_copy_", "codegen/dynamic_slice/mps/kernels.metal", "#define DS_COPY"),
    (r"^dynamic_update_slice_", "codegen/dynamic_slice/mps/kernels.metal", "#define DYNAMIC_UPDATE_SLICE"),
    (r"^dynamic_slice_", "codegen/dynamic_slice/mps/kernels.metal", "#define DYNAMIC_SLICE"),
    (r"^scatter_add_scan_", "codegen/indexing/mps/scatter_add.metal", "#define SCATTER_ADD_SCAN"),
    (r"^scatter_add_", "codegen/indexing/mps/scatter_add.metal", "#define SCATTER_ADD("),
    (r"^random_bits", "kernels/factory/mps/kernels.metal", "kernel void random_bits"),
    (r"^matmul_(u64|i64)$", "codegen/dot_general/mps/integer/matmul_wide.metal", "inline void matmul_wide_impl"),
    (r"^matmul_(u8|u16|u32|i8|i16|i32)$", "codegen/dot_general/mps/integer/matmul.metal", "inline void matmul_impl"),
    (r"^reduce_\w+_rows_", "codegen/reduction/mps/kernels.metal", "inline void reduce_rows"),
    (r"^reduce_\w+_cols_", "codegen/reduction/mps/kernels.metal", "inline void reduce_cols"),
    (r"^reduce_\w+_grouped_", "codegen/reduction/mps/kernels.metal", "inline void reduce_grouped"),
    (r"^reduce_", "codegen/reduction/mps/kernels.metal", "inline void reduce("),
    (r"^fill_\d$", "kernels/factory/mps/kernels.metal", "#define FULL"),
    (r"^iota_", "kernels/factory/mps/kernels.metal", "#define IOTA"),
]


def source_of(kernel):
    """``(file, anchor)`` of a primitive kernel's source, or None."""
    for pattern, path, anchor in _SOURCES:
        if re.match(pattern, kernel):
            return path, anchor
    return None


def _traced_at(sources):
    """``file:line  code`` lines for each traced source location in ``sources``."""
    lines = []
    for file, line in sources:
        try:
            code = pathlib.Path(file).read_text().splitlines()[line - 1].strip()
        except (OSError, UnicodeDecodeError, IndexError):
            code = ""
        lines.append(f"{file}:{line}  {code}")
    return lines
