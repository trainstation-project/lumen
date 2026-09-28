//! The CPU copy kernels: an ordinary `memcpy` in both directions. Host
//! memory and CPU storage are the same address space.

use crate::allocator::Allocator;
use crate::device::Device;

/// Copy `nbytes` from the host at `src` into host memory at `dst`.
///
/// # Safety
/// `dst` must be valid for writing and `src` for reading `nbytes`, and the
/// two must not overlap.
pub(super) unsafe fn copy_h2d(
    _alloc: &dyn Allocator,
    _device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    // SAFETY: the caller's contract.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
}

/// Copy `nbytes` from host memory at `src` to the host at `dst`.
///
/// # Safety
/// As [`copy_h2d`].
pub(super) unsafe fn copy_d2h(
    _alloc: &dyn Allocator,
    _device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    // SAFETY: the caller's contract.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
}
