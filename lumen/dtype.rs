//! | torch                 | lumen      | Rust type       |
//! |-----------------------|------------|-----------------|
//! | Bool                  | [`DType::Bool`] | `bool`     |
//! | Byte                  | [`DType::U8`]   | `u8`       |
//! | Char                  | [`DType::I8`]   | `i8`       |
//! | Short                 | [`DType::I16`]  | `i16`      |
//! | Int                   | [`DType::I32`]  | `i32`      |
//! | Long                  | [`DType::I64`]  | `i64`      |
//! | UInt16/32/64          | `U16..U64` | `u16/u32/u64`   |
//! | Half                  | [`DType::F16`]  | [`f16`]    |
//! | BFloat16              | [`DType::BF16`] | [`bf16`]   |
//! | Float / Double        | `F32`/`F64`| `f32`/`f64`     |
//!
//! Not supported (they need storage/kernels beyond plain bytes):
//! Complex*, quantized (QInt8/QUInt8/QInt32/QUInt4x2/QUInt2x4),
//! Bits1x8..Bits16, Float8_*, and the sub-byte UInt1..UInt7.

use std::fmt;

// Re-exported so users of the crate can name the element types directly.
pub use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    /// Size of one element in bytes (PyTorch: `TypeMeta::itemsize()`).
    pub const fn size_of(self) -> usize {
        use DType::*;
        match self {
            Bool | U8 | I8 => 1,
            U16 | I16 | F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }

    pub const fn name(self) -> &'static str {
        use DType::*;
        match self {
            Bool => "bool",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F16 => "f16",
            BF16 => "bf16",
            F32 => "f32",
            F64 => "f64",
        }
    }

    /// Floating-point (real) dtypes.
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Signed or unsigned integer dtypes.
    pub const fn is_int(self) -> bool {
        matches!(
            self,
            DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Marks Rust types that can be stored in a [`crate::Storage`].
pub trait Element: Copy + fmt::Debug + fmt::Display + 'static {
    const DTYPE: DType;
    const ZERO: Self;
    fn from_usize(v: usize) -> Self;
}

macro_rules! impl_element {
    ($t:ty, $dtype:expr) => {
        impl Element for $t {
            const DTYPE: DType = $dtype;
            const ZERO: Self = 0 as $t;
            fn from_usize(v: usize) -> Self {
                v as $t
            }
        }
    };
}

impl_element!(f32, DType::F32);
impl_element!(f64, DType::F64);
impl_element!(i8, DType::I8);
impl_element!(i16, DType::I16);
impl_element!(i32, DType::I32);
impl_element!(i64, DType::I64);
impl_element!(u8, DType::U8);
impl_element!(u16, DType::U16);
impl_element!(u32, DType::U32);
impl_element!(u64, DType::U64);

impl Element for bool {
    const DTYPE: DType = DType::Bool;
    const ZERO: Self = false;
    fn from_usize(v: usize) -> Self {
        v != 0
    }
}

impl Element for f16 {
    const DTYPE: DType = DType::F16;
    const ZERO: Self = f16::ZERO;
    fn from_usize(v: usize) -> Self {
        f16::from_f64(v as f64)
    }
}

impl Element for bf16 {
    const DTYPE: DType = DType::BF16;
    const ZERO: Self = bf16::ZERO;
    fn from_usize(v: usize) -> Self {
        bf16::from_f64(v as f64)
    }
}
