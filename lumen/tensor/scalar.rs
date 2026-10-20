use super::dtype::{DType, bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl Scalar {
    pub fn inferred_dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::I64,
            Scalar::Float(_) => super::tensor_options::DEFAULT_DTYPE,
        }
    }

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
