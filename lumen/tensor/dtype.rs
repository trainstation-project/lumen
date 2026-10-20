use std::fmt;

use super::scalar::Scalar;

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
    I2,
    I4,
    U2,
    U4,
    F8E4M3FN,
    F8E5M2,
    F8E4M3B11FNUZ,
}

impl DType {
    pub const fn size_of(self) -> usize {
        use DType::*;
        match self {
            Bool | U8 | I8 | I2 | I4 | U2 | U4 | F8E4M3FN | F8E5M2 | F8E4M3B11FNUZ => 1,
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
            I2 => "i2",
            I4 => "i4",
            U2 => "u2",
            U4 => "u4",
            F8E4M3FN => "f8e4m3fn",
            F8E5M2 => "f8e5m2",
            F8E4M3B11FNUZ => "f8e4m3b11fnuz",
        }
    }

    pub const fn bits(self) -> usize {
        match self {
            DType::I2 | DType::U2 => 2,
            DType::I4 | DType::U4 => 4,
            other => 8 * other.size_of(),
        }
    }

    pub const fn full_name(self) -> &'static str {
        use DType::*;
        match self {
            F32 => "float32",
            F64 => "float64",
            F16 => "float16",
            BF16 => "bfloat16",
            I8 => "int8",
            I16 => "int16",
            I32 => "int32",
            I64 => "int64",
            U8 => "uint8",
            U16 => "uint16",
            U32 => "uint32",
            U64 => "uint64",
            Bool => "bool",
            I2 => "int2",
            I4 => "int4",
            U2 => "uint2",
            U4 => "uint4",
            F8E4M3FN => "float8_e4m3fn",
            F8E5M2 => "float8_e5m2",
            F8E4M3B11FNUZ => "float8_e4m3b11fnuz",
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F16
                | DType::BF16
                | DType::F32
                | DType::F64
                | DType::F8E4M3FN
                | DType::F8E5M2
                | DType::F8E4M3B11FNUZ
        )
    }

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
                | DType::I2
                | DType::I4
                | DType::U2
                | DType::U4
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub trait Element: Copy + fmt::Debug + fmt::Display + 'static {
    const DTYPE: DType;
    const ZERO: Self;
    fn from_scalar(v: Scalar) -> Self;
    fn to_scalar(self) -> Scalar;
}

macro_rules! impl_element {
    ($t:ty, $dtype:expr, $scalar:ident) => {
        impl Element for $t {
            const DTYPE: DType = $dtype;
            const ZERO: Self = 0 as $t;
            fn from_scalar(v: Scalar) -> Self {
                match v {
                    Scalar::Bool(b) => b as u8 as $t,
                    Scalar::Int(i) => i as $t,
                    Scalar::Float(f) => f as $t,
                }
            }
            fn to_scalar(self) -> Scalar {
                Scalar::$scalar(self as _)
            }
        }
    };
}

impl_element!(f32, DType::F32, Float);
impl_element!(f64, DType::F64, Float);
impl_element!(i8, DType::I8, Int);
impl_element!(i16, DType::I16, Int);
impl_element!(i32, DType::I32, Int);
impl_element!(i64, DType::I64, Int);
impl_element!(u8, DType::U8, Int);
impl_element!(u16, DType::U16, Int);
impl_element!(u32, DType::U32, Int);
impl_element!(u64, DType::U64, Int);

impl Element for bool {
    const DTYPE: DType = DType::Bool;
    const ZERO: Self = false;
    fn from_scalar(v: Scalar) -> Self {
        match v {
            Scalar::Bool(b) => b,
            Scalar::Int(i) => i != 0,
            Scalar::Float(f) => f != 0.0,
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Bool(self)
    }
}

fn round_to_odd(x: f64) -> f32 {
    let f = x as f32;
    let bits = f.to_bits();
    if !f.is_finite() || f64::from(f) == x || bits & 1 == 1 {
        return f;
    }
    let outward = (f64::from(f) < x) != f.is_sign_negative();
    f32::from_bits(if outward { bits + 1 } else { bits - 1 })
}

pub(crate) fn to_f16(x: f64) -> f16 {
    f16::from_f32(round_to_odd(x))
}

pub(crate) fn to_bf16(x: f64) -> bf16 {
    bf16::from_f32(round_to_odd(x))
}

impl Element for f16 {
    const DTYPE: DType = DType::F16;
    const ZERO: Self = f16::ZERO;
    fn from_scalar(v: Scalar) -> Self {
        to_f16(v.to_f64())
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Float(self.to_f64())
    }
}

impl Element for bf16 {
    const DTYPE: DType = DType::BF16;
    const ZERO: Self = bf16::ZERO;
    fn from_scalar(v: Scalar) -> Self {
        to_bf16(v.to_f64())
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Float(self.to_f64())
    }
}

#[derive(Clone, Copy)]
struct Float8 {
    exponent: u32,
    mantissa: u32,
    bias: i32,
    kind: Kind,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Ieee,
    Fn,
    Fnuz,
}

fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

impl Float8 {
    fn decode(self, bits: u8) -> f64 {
        let (e_max, m_max) = ((1u32 << self.exponent) - 1, (1u32 << self.mantissa) - 1);
        let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = u32::from(bits >> self.mantissa) & e_max;
        let m = u32::from(bits) & m_max;
        match self.kind {
            Kind::Fnuz if bits == 0x80 => return f64::NAN,
            Kind::Ieee if e == e_max && m == 0 => return sign * f64::INFINITY,
            Kind::Ieee if e == e_max => return f64::NAN,
            Kind::Fn if e == e_max && m == m_max => return f64::NAN,
            _ => {}
        }
        let unit = f64::from(1u32 << self.mantissa);
        let value = match e {
            0 => f64::from(m) / unit * pow2(1 - self.bias),
            e => (1.0 + f64::from(m) / unit) * pow2(e as i32 - self.bias),
        };
        sign * value
    }

    fn encode(self, x: f64) -> u8 {
        let nan = match self.kind {
            Kind::Ieee => 0x7E,
            Kind::Fn => 0x7F,
            Kind::Fnuz => 0x80,
        };
        let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
        if x.is_nan() {
            return nan;
        }
        let a = x.abs();
        let inf = (((1u32 << self.exponent) - 1) << self.mantissa) as u8;
        if a.is_infinite() {
            return if self.kind == Kind::Ieee {
                sign | inf
            } else {
                nan
            };
        }
        let min_exp = 1 - self.bias;
        let m = self.mantissa as i32;
        let bits: u32 = if a < pow2(min_exp) {
            (a / pow2(min_exp - m)).round_ties_even() as u32
        } else {
            let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
            let q = (a / pow2(e - m)).round_ties_even() as u32;
            (((e - min_exp + 1) as u32) << self.mantissa) + q - (1 << self.mantissa)
        };
        if bits == 0 && self.kind == Kind::Fnuz {
            return 0;
        }
        let largest = match self.kind {
            Kind::Ieee => u32::from(inf) - 1,
            Kind::Fn => u32::from(inf) | ((1 << self.mantissa) - 2),
            Kind::Fnuz => 0x7F,
        };
        match bits > largest {
            true if self.kind == Kind::Ieee => sign | inf,
            true => nan,
            false => sign | bits as u8,
        }
    }
}

macro_rules! narrow {
    ($(#[$doc:meta])* $name:ident, $dtype:expr, $from:expr, $to:expr) => {
        $(#[$doc])*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(pub u8);

        impl $name {
            pub fn from_f64(x: f64) -> Self {
                let from: fn(f64) -> u8 = $from;
                $name(from(x))
            }

            pub fn to_f64(self) -> f64 {
                let to: fn(u8) -> f64 = $to;
                to(self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self, f)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_f64())
            }
        }

        impl Element for $name {
            const DTYPE: DType = $dtype;
            const ZERO: Self = $name(0);
            fn from_scalar(v: Scalar) -> Self {
                match v {
                    Scalar::Float(x) if !DTYPE_IS_FLOAT[$dtype as usize] => {
                        $name::from_f64(x.trunc().clamp(i64::MIN as f64, i64::MAX as f64))
                    }
                    v => $name::from_f64(v.to_f64()),
                }
            }
            fn to_scalar(self) -> Scalar {
                match DTYPE_IS_FLOAT[$dtype as usize] {
                    true => Scalar::Float(self.to_f64()),
                    false => Scalar::Int(self.to_f64() as i64),
                }
            }
        }
    };
}

const DTYPE_IS_FLOAT: [bool; 20] = {
    let all = [
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::I2,
        DType::I4,
        DType::U2,
        DType::U4,
        DType::F8E4M3FN,
        DType::F8E5M2,
        DType::F8E4M3B11FNUZ,
    ];
    let mut out = [false; 20];
    let mut i = 0;
    while i < all.len() {
        out[all[i] as usize] = all[i].is_float();
        i += 1;
    }
    out
};

fn wrap_bits(x: f64, bits: u32) -> u8 {
    ((x as i64) & ((1 << bits) - 1)) as u8
}

fn signed_bits(b: u8, bits: u32) -> f64 {
    f64::from(((b << (8 - bits)) as i8) >> (8 - bits))
}

const E4M3FN: Float8 = Float8 {
    exponent: 4,
    mantissa: 3,
    bias: 7,
    kind: Kind::Fn,
};
const E5M2: Float8 = Float8 {
    exponent: 5,
    mantissa: 2,
    bias: 15,
    kind: Kind::Ieee,
};
const E4M3B11FNUZ: Float8 = Float8 {
    exponent: 4,
    mantissa: 3,
    bias: 11,
    kind: Kind::Fnuz,
};

narrow!(i2, DType::I2, |x| wrap_bits(x, 2), |b| signed_bits(b, 2));
narrow!(i4, DType::I4, |x| wrap_bits(x, 4), |b| signed_bits(b, 4));
narrow!(u2, DType::U2, |x| wrap_bits(x, 2), f64::from);
narrow!(u4, DType::U4, |x| wrap_bits(x, 4), f64::from);
narrow!(f8e4m3fn, DType::F8E4M3FN, |x| E4M3FN.encode(x), |b| E4M3FN
    .decode(b));
narrow!(f8e5m2, DType::F8E5M2, |x| E5M2.encode(x), |b| E5M2
    .decode(b));
narrow!(
    f8e4m3b11fnuz,
    DType::F8E4M3B11FNUZ,
    |x| E4M3B11FNUZ.encode(x),
    |b| E4M3B11FNUZ.decode(b)
);

macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::DType::Bool => {
                type $T = bool;
                $body
            }
            $crate::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::DType::U16 => {
                type $T = u16;
                $body
            }
            $crate::DType::U32 => {
                type $T = u32;
                $body
            }
            $crate::DType::U64 => {
                type $T = u64;
                $body
            }
            $crate::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::DType::F16 => {
                type $T = $crate::tensor::dtype::f16;
                $body
            }
            $crate::DType::BF16 => {
                type $T = $crate::tensor::dtype::bf16;
                $body
            }
            $crate::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::DType::F64 => {
                type $T = f64;
                $body
            }
            $crate::DType::I2 => {
                type $T = $crate::tensor::dtype::i2;
                $body
            }
            $crate::DType::I4 => {
                type $T = $crate::tensor::dtype::i4;
                $body
            }
            $crate::DType::U2 => {
                type $T = $crate::tensor::dtype::u2;
                $body
            }
            $crate::DType::U4 => {
                type $T = $crate::tensor::dtype::u4;
                $body
            }
            $crate::DType::F8E4M3FN => {
                type $T = $crate::tensor::dtype::f8e4m3fn;
                $body
            }
            $crate::DType::F8E5M2 => {
                type $T = $crate::tensor::dtype::f8e5m2;
                $body
            }
            $crate::DType::F8E4M3B11FNUZ => {
                type $T = $crate::tensor::dtype::f8e4m3b11fnuz;
                $body
            }
        }
    };
}
pub(crate) use dispatch_dtype;
