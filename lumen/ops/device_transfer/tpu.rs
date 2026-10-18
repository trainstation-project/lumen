//! TPU-to-host copies, through the TPU allocator's reads of its region
//! (`allocator::tpu::copy_to_host`). Host-to-TPU copies need a TPU program
//! to write the region: not yet.

use std::ptr::NonNull;

use crate::Tensor;
use crate::device::Device;

pub(super) fn copy_d2h(dst: &Tensor, src: &Tensor) {
    let Device::Tpu(index) = src.device() else {
        unreachable!("the TPU copy kernel runs on TPU tensors")
    };
    let nbytes = dst.nbytes();
    let Some(ptr) = NonNull::new(src.data_ptr()) else {
        return;
    };
    // SAFETY: `dst` is a host tensor of `nbytes` contiguous bytes, which
    // this copy alone writes (`copy_to`'s caller ensures it).
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.data_ptr(), nbytes) };
    crate::allocator::tpu::copy_to_host(index, ptr, dst)
        .unwrap_or_else(|e| panic!("copying {nbytes} bytes from tpu:{index} failed: {e}"));
}
