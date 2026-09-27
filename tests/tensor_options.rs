//! Tensor factories and `TensorOptions`, checked against PyTorch's
//! semantics (`at::zeros(size, options)` and friends).

use lumen::dtype::f16;
use lumen::{DType, Device, Scalar, Tensor, TensorOptions};

#[test]
fn defaults_are_float32_on_cpu() {
    let t = Tensor::zeros(&[2, 3], TensorOptions::new());
    assert_eq!((t.dtype(), t.device()), (DType::F32, Device::Cpu));
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(Tensor::ones(&[2], TensorOptions::new()).dtype(), DType::F32);
}

#[test]
fn a_dtype_or_device_converts_into_options() {
    assert_eq!(Tensor::zeros(&[1], DType::I64).dtype(), DType::I64);
    let t = Tensor::zeros(&[1], Device::Cpu);
    assert_eq!((t.dtype(), t.device()), (DType::F32, Device::Cpu));
    let opts = TensorOptions::new().dtype(DType::U8).device(Device::Cpu);
    assert_eq!(opts.dtype_opt(), Some(DType::U8));
    assert_eq!(opts.device_opt(), Some(Device::Cpu));
    assert_eq!(TensorOptions::new().dtype_opt(), None);
}

#[test]
fn full_infers_dtype_from_the_fill_value() {
    // torch.full((2,), 7).dtype == int64; True -> bool; 1.5 -> float32
    let i = Tensor::full(&[2], 7, TensorOptions::new());
    assert_eq!(i.dtype(), DType::I64);
    assert_eq!(i.to_vec::<i64>(), vec![7, 7]);
    assert_eq!(
        Tensor::full(&[1], true, TensorOptions::new()).dtype(),
        DType::Bool
    );
    let f = Tensor::full(&[1], 1.5, TensorOptions::new());
    assert_eq!(f.dtype(), DType::F32);
    assert_eq!(f.to_vec::<f32>(), vec![1.5]);
}

#[test]
fn full_converts_the_fill_value_to_the_given_dtype() {
    assert_eq!(Tensor::full(&[1], 2.9, DType::I32).to_vec::<i32>(), vec![2]);
    assert_eq!(
        Tensor::full(&[1], 3, DType::F16).to_vec::<f16>(),
        vec![f16::from_f32(3.0)]
    );
    assert_eq!(
        Tensor::full(&[2], 0, DType::Bool).to_vec::<bool>(),
        vec![false, false]
    );
    let t = Tensor::full(&[2, 2], Scalar::Float(0.5), DType::F64);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.to_vec::<f64>(), vec![0.5; 4]);
}

#[test]
fn ones_uses_the_default_dtype_not_the_int_fill() {
    let t = Tensor::ones(&[3], TensorOptions::new());
    assert_eq!(t.to_vec::<f32>(), vec![1.0; 3]);
    assert_eq!(Tensor::ones(&[1], DType::Bool).to_vec::<bool>(), vec![true]);
}

#[test]
fn arange_infers_dtype_from_end() {
    // torch.arange(4).dtype == int64; torch.arange(2.5) == [0., 1., 2.]
    let i = Tensor::arange(4, TensorOptions::new());
    assert_eq!(i.dtype(), DType::I64);
    assert_eq!(i.to_vec::<i64>(), vec![0, 1, 2, 3]);
    let f = Tensor::arange(2.5, TensorOptions::new());
    assert_eq!(f.dtype(), DType::F32);
    assert_eq!(f.to_vec::<f32>(), vec![0.0, 1.0, 2.0]);
    assert_eq!(
        Tensor::arange(3, DType::F64).to_vec::<f64>(),
        vec![0.0, 1.0, 2.0]
    );
    assert_eq!(Tensor::arange(0, TensorOptions::new()).numel(), 0);
}

#[test]
fn from_slice_keeps_or_converts_the_element_type() {
    let t = Tensor::from_slice(&[1u8, 2, 3], TensorOptions::new());
    assert_eq!(t.dtype(), DType::U8);
    let f = Tensor::from_slice(&[1u8, 2, 3], DType::F32);
    assert_eq!(f.to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
    let b = Tensor::from_slice(&[0.0f64, 2.0], DType::Bool);
    assert_eq!(b.to_vec::<bool>(), vec![false, true]);
}
