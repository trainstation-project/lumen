use crate::graph::TensorType;
use crate::{Device, Tensor, TensorOptions};

pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    let options = TensorOptions::new()
        .dtype(src.dtype())
        .device(Device::Mps(0));
    let staging = unsafe { Tensor::empty(src.shape(), options) };
    unsafe { std::ptr::copy_nonoverlapping(src.data_ptr(), staging.data_ptr(), src.nbytes()) };
    let ty = TensorType::new(dst.dtype(), dst.shape());
    let (from, to) = (staging.data_ptr().cast_const(), dst.data_ptr());
    crate::kernels::dynamic_slice::mps::copy(from, to, &ty, vec![staging, dst.clone()], "copy_h2d")
        .unwrap_or_else(|e| panic!("copy_h2d: {e}"));
}
