use crate::Tensor;

pub(super) fn memcpy(dst: &Tensor, src: &Tensor) {
    unsafe { std::ptr::copy_nonoverlapping(src.data_ptr(), dst.data_ptr(), dst.nbytes()) }
}
