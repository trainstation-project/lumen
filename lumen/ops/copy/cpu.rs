use super::{data_ptr, nbytes};
use crate::Tensor;

pub(super) fn memcpy(dst: &Tensor, src: &Tensor) {
    unsafe { std::ptr::copy_nonoverlapping(data_ptr(src), data_ptr(dst), nbytes(dst)) }
}
