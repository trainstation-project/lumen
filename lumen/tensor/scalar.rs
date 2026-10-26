//! `Scalar`, a dtype-less bool, integer or float value used for fills and constants.
use super::dtype::{DType, bf16, f16};

/// A bool, integer or float value without a fixed dtype.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl Scalar {
    /// The dtype a tensor made from this scalar gets: bool, i64 or the default float dtype.
    pub fn inferred_dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::I64,
            Scalar::Float(_) => super::tensor_options::DEFAULT_DTYPE,
        }
    }

    /// The value as f64, with `true` as 1.
    pub fn to_f64(self) -> f64 {
        match self {
            Scalar::Bool(b) => b as u8 as f64,
            Scalar::Int(i) => i as f64,
            Scalar::Float(f) => f,
        }
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}

/// Implements `From` for integer types as `Scalar::Int`.
macro_rules! from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Scalar {
            fn from(v: $t) -> Self {
                Scalar::Int(v as i64)
            }
        }
    )*};
}
from_int!(i8, i16, i32, i64, u8, u16, u32, u64, usize);

/// Implements `From` for float types as `Scalar::Float`.
macro_rules! from_float {
    ($($t:ty),*) => {$(
        impl From<$t> for Scalar {
            fn from(v: $t) -> Self {
                Scalar::Float(v.into())
            }
        }
    )*};
}
from_float!(f32, f64);

impl From<f16> for Scalar {
    fn from(v: f16) -> Self {
        Scalar::Float(v.to_f64())
    }
}

impl From<bf16> for Scalar {
    fn from(v: bf16) -> Self {
        Scalar::Float(v.to_f64())
    }
}
