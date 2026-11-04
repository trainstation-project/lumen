//! What the CUDA (CuTe DSL) code generators share: the backend type, element types and constants.

use crate::codegen::utils::triton::bits;
use crate::{DType, Scalar};

/// The CUDA code generator: kernels in NVIDIA's CuTe DSL (`cutlass.cute`), compiled at launch.
pub(crate) struct Cuda;

/// The CuTe numeric type of `dtype`, or `None` for one it has no kernels for.
pub(crate) fn cute_type(dtype: DType) -> Option<&'static str> {
    use DType::*;

    Some(match dtype {
        Bool => "cutlass.Boolean",
        U8 => "cutlass.Uint8",
        U16 => "cutlass.Uint16",
        U32 => "cutlass.Uint32",
        U64 => "cutlass.Uint64",
        I8 => "cutlass.Int8",
        I16 => "cutlass.Int16",
        I32 => "cutlass.Int32",
        I64 => "cutlass.Int64",
        F16 => "cutlass.Float16",
        BF16 => "cutlass.BFloat16",
        F32 => "cutlass.Float32",
        F64 => "cutlass.Float64",
        _ => return None,
    })
}

/// The unsigned CuTe type of `bits` bits.
pub(crate) fn unsigned(bits: usize) -> &'static str {
    match bits {
        8 => "cutlass.Uint8",
        16 => "cutlass.Uint16",
        32 => "cutlass.Uint32",
        _ => "cutlass.Uint64",
    }
}

/// `value` as a CuTe constant of `dtype`, written as its bits so every dtype's value is exact.
pub(crate) fn constant(dtype: DType, value: Scalar) -> String {
    let bits = bits(dtype, value);
    let t = cute_type(dtype).expect("a CuTe dtype");

    match dtype {
        DType::Bool => format!("cutlass.Boolean({})", bits != 0),
        d if d.is_float() => format!("{}({bits:#x}).bitcast({t})", unsigned(d.bits())),
        _ => format!("{t}({})", signed_bits(dtype, bits)),
    }
}

/// The integer `dtype` holds as `bits`, written as a Python literal.
fn signed_bits(dtype: DType, bits: u64) -> String {
    let width = dtype.bits();

    match dtype {
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            let shifted = (bits << (64 - width)) as i64 >> (64 - width);
            shifted.to_string()
        }
        _ => bits.to_string(),
    }
}
