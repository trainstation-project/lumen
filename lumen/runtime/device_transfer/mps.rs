//! Metal host-to-device copy.

use crate::graph::TensorType;
use crate::{Device, Tensor, TensorOptions};

/// Copies host tensor `src` into MPS tensor `dst` through a fresh MPS staging buffer and a GPU copy kernel.
pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    // Stage on the CPU, then copy on the GPU so the write is ordered after work already queued on `dst`.
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
