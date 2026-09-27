//! Tests for the dtype layer: sizes, names, and per-dtype tensor roundtrips.

use lumen::dtype::{bf16, f16};
use lumen::{DType, Tensor};

#[test]
fn size_of_matches_rust_types() {
    assert_eq!(DType::Bool.size_of(), 1);
    assert_eq!(DType::U8.size_of(), size_of::<u8>());
    assert_eq!(DType::U16.size_of(), size_of::<u16>());
    assert_eq!(DType::U32.size_of(), size_of::<u32>());
    assert_eq!(DType::U64.size_of(), size_of::<u64>());
    assert_eq!(DType::I8.size_of(), size_of::<i8>());
    assert_eq!(DType::I16.size_of(), size_of::<i16>());
    assert_eq!(DType::I32.size_of(), size_of::<i32>());
    assert_eq!(DType::I64.size_of(), size_of::<i64>());
    assert_eq!(DType::F16.size_of(), 2);
    assert_eq!(DType::BF16.size_of(), 2);
    assert_eq!(DType::F32.size_of(), size_of::<f32>());
    assert_eq!(DType::F64.size_of(), size_of::<f64>());
}

#[test]
fn dtype_predicates() {
    assert!(DType::F32.is_float() && DType::F16.is_float() && DType::BF16.is_float());
    assert!(!DType::I32.is_float());
    assert!(DType::I8.is_int() && DType::U64.is_int());
    assert!(!DType::Bool.is_int() && !DType::F32.is_int());
}

#[test]
fn dtype_names() {
    assert_eq!(DType::BF16.to_string(), "bf16");
    assert_eq!(DType::I8.to_string(), "i8");
}

/// Every dtype can roundtrip values through a tensor: build, read, write.
macro_rules! test_roundtrip {
    ($name:ident, $t:ty, $dtype:expr, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            let a: $t = $a;
            let b: $t = $b;
            let t = Tensor::from_slice(&[a, b]);
            assert_eq!(t.dtype(), $dtype);
            assert_eq!(t.numel(), 2);
            assert_eq!(t.get::<$t>(&[0]), a);
            t.set(&[1], a);
            assert_eq!(t.to_vec::<$t>(), vec![a, a]);
            // nbytes = numel * itemsize
            let m = Tensor::zeros::<$t>(&[2, 3]);
            assert_eq!(m.numel() * $dtype.size_of(), 6 * size_of::<$t>());
        }
    };
}

test_roundtrip!(roundtrip_bool, bool, DType::Bool, true, false);
test_roundtrip!(roundtrip_u8, u8, DType::U8, 1, 2);
test_roundtrip!(roundtrip_u16, u16, DType::U16, 1, 2);
test_roundtrip!(roundtrip_u32, u32, DType::U32, 1, 2);
test_roundtrip!(roundtrip_u64, u64, DType::U64, 1, 2);
test_roundtrip!(roundtrip_i8, i8, DType::I8, -1, 2);
test_roundtrip!(roundtrip_i16, i16, DType::I16, -1, 2);
test_roundtrip!(roundtrip_i32, i32, DType::I32, -1, 2);
test_roundtrip!(roundtrip_i64, i64, DType::I64, -1, 2);
test_roundtrip!(roundtrip_f32, f32, DType::F32, 1.5, 2.5);
test_roundtrip!(roundtrip_f64, f64, DType::F64, 1.5, 2.5);
test_roundtrip!(
    roundtrip_f16,
    f16,
    DType::F16,
    f16::from_f32(1.5),
    f16::from_f32(2.5)
);
test_roundtrip!(
    roundtrip_bf16,
    bf16,
    DType::BF16,
    bf16::from_f32(1.5),
    bf16::from_f32(2.5)
);
#[test]
fn f16_arange_and_display() {
    let t = Tensor::arange::<f16>(4);
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.get::<f16>(&[3]), f16::from_f32(3.0));
    let s = format!("{t}");
    assert!(s.contains("dtype=f16"), "{s}");
}

#[test]
fn views_work_for_every_dtype() {
    // Views are dtype-agnostic (they only move metadata), but check one
    // non-trivial element size to be sure offsets are in *elements*.
    let t = Tensor::arange::<f64>(6).reshape(&[2, 3]);
    let row = t.select(0, 1);
    assert_eq!(row.storage_offset(), 3);
    assert_eq!(row.get::<f64>(&[0]), 3.0);
}
