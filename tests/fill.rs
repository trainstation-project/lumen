//! `Tensor::fill_` / `zero_` on the CPU (PyTorch: `Tensor.fill_`, `zero_`).

use lumen::tensor::dtype::{bf16, f16};
use lumen::{DType, Device, Tensor};

#[test]
fn fill_sets_every_element_and_returns_the_tensor() {
    let t = Tensor::zeros(&[2, 3], DType::F32);
    assert_eq!(t.fill_(1.5).to_vec::<f32>(), vec![1.5; 6]);
    assert_eq!(t.zero_().to_vec::<f32>(), vec![0.0; 6]);
}

#[test]
fn fill_converts_the_value_to_the_dtype() {
    let i = Tensor::zeros(&[3], DType::I32);
    assert_eq!(i.fill_(2.9).to_vec::<i32>(), vec![2; 3]);
    assert_eq!(i.fill_(-1).to_vec::<i32>(), vec![-1; 3]); // all-0xFF bytes: memset
    let b = Tensor::zeros(&[2], DType::Bool);
    assert_eq!(b.fill_(true).to_vec::<bool>(), vec![true, true]);
    let h = Tensor::zeros(&[2], DType::F16);
    assert_eq!(h.fill_(0.5).to_vec::<f16>(), vec![f16::from_f32(0.5); 2]);
    let bh = Tensor::zeros(&[2], DType::BF16);
    assert_eq!(bh.fill_(3).to_vec::<bf16>(), vec![bf16::from_f32(3.0); 2]);
    let u = Tensor::zeros(&[4], DType::U8);
    assert_eq!(u.fill_(0xAB).to_vec::<u8>(), vec![0xAB; 4]);
}

#[test]
fn fill_writes_through_views_only() {
    let t = Tensor::arange(12, DType::F32).reshape(&[3, 4]);
    t.select(0, 1).fill_(-1.0); // contiguous row
    t.select(1, 2).zero_(); // strided column
    assert_eq!(
        t.to_vec::<f32>(),
        vec![
            0.0, 1.0, 0.0, 3.0, -1.0, -1.0, 0.0, -1.0, 8.0, 9.0, 0.0, 11.0
        ]
    );
    let tt = Tensor::arange(4, DType::I64)
        .reshape(&[2, 2])
        .transpose(0, 1);
    tt.fill_(7);
    assert_eq!(tt.to_vec::<i64>(), vec![7; 4]);
}

#[test]
fn fill_of_an_empty_tensor_is_a_no_op() {
    let t = Tensor::zeros(&[0, 3], Device::Cpu);
    t.fill_(1);
    assert_eq!(t.numel(), 0);
}

#[test]
fn full_and_ones_fill_fresh_storage() {
    assert_eq!(
        Tensor::full(&[2, 2], 0, DType::F64).to_vec::<f64>(),
        vec![0.0; 4]
    );
    assert_eq!(
        Tensor::full(&[3], -1, DType::I8).to_vec::<i8>(),
        vec![-1; 3]
    );
    assert_eq!(Tensor::ones(&[3], DType::U8).to_vec::<u8>(), vec![1; 3]);
    assert_eq!(Tensor::ones(&[2], DType::F32).to_vec::<f32>(), vec![1.0; 2]);
}
