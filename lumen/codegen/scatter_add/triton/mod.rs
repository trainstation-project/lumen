//! Triton scatter-adds: the fixed kernels in `kernel.py` (atomic by default; deterministic, a counting sort
//! of the picks by position, then a row's updates added in pick order) and their launch configs. Nothing
//! fuses into a scatter-add, so there is no code to generate.

mod config;
pub(crate) mod launch;

pub(crate) use config::*;

use crate::DType;

/// Whether Triton's `tl.atomic_add` adds elements of `dtype`: floats from 16 to 64 bits and 32- or 64-bit
/// integers, not bool, float8 or narrower integers.
pub(crate) fn atomic_add(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F16
            | DType::BF16
            | DType::F32
            | DType::F64
            | DType::I32
            | DType::I64
            | DType::U32
            | DType::U64
    )
}
