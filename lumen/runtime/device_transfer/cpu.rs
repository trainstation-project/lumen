//! Host copy kernel for device transfers.

use crate::Tensor;

/// Copies `dst.nbytes()` bytes from `src` to `dst` with a plain memcpy.
pub(super) fn memcpy(dst: &Tensor, src: &Tensor) {
    unsafe { std::ptr::copy_nonoverlapping(src.data_ptr(), dst.data_ptr(), dst.nbytes()) }
}
