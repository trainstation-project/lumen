# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from cutlass import BFloat16, Float16, Float32, Int32, Int64, Numeric, Uint32

#: lumen dtype names to the CuTe dtype of the same element.
_LUMEN_DTYPE_TO_CUTE_DTYPE_MAPPING = {
    # floating point dtypes
    "float32": Float32,
    "float16": Float16,
    "bfloat16": BFloat16,
    # integer dtypes
    "int32": Int32,
    "int64": Int64,
    "uint32": Uint32,
}


def get_cute_dtype(dtype: str) -> type[Numeric]:
    """The CuTe dtype for a lumen dtype name (``"float32"``, ...)."""
    return _LUMEN_DTYPE_TO_CUTE_DTYPE_MAPPING[dtype]
