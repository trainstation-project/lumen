//! Copies into MPS memory, in stream order: the host never waits for the
//! GPU (PyTorch: `copy_(non_blocking=True)` from pinned memory).

use crate::graph::TensorType;
use crate::{Device, Tensor, TensorOptions};

/// Copy host `src` into MPS `dst` after the work submitted so far and before
/// what is submitted after, without waiting for either: `src`'s bytes go
/// into new MPS memory now (no GPU work has used it, so the host writes it
/// at once; `src` is the caller's again on return), and a kernel then copies
/// them into `dst` in the stream (`dst` may be in use by work in flight).
pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    let options = TensorOptions::new().dtype(src.dtype()).device(Device::Mps);
    // SAFETY: the copy below writes every element.
    let staging = unsafe { Tensor::empty(src.shape(), options) };
    // SAFETY: distinct contiguous buffers of `nbytes`, both host-accessible
    // (MPS memory is shared), the staging one new.
    unsafe { std::ptr::copy_nonoverlapping(src.data_ptr(), staging.data_ptr(), src.nbytes()) };
    let ty = TensorType::new(dst.dtype(), dst.shape());
    let (from, to) = (staging.data_ptr().cast_const(), dst.data_ptr());
    crate::ops::dynamic_slice::mps::copy(from, to, &ty, vec![staging, dst.clone()], "copy_h2d")
        .unwrap_or_else(|e| panic!("copy_h2d: {e}"));
}
