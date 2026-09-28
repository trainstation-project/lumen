//! The MPS copy kernels: an ordinary `memcpy` in both directions.
//!
//! MPS buffers are `MTLBuffer`s in Metal's Shared storage mode. On Apple
//! Silicon's unified memory `buffer.contents` is an ordinary CPU pointer, so
//! an ordinary `memcpy` moves bytes as efficiently as anything Metal offers
//! for a host-side range (PyTorch's MPS `copy_` likewise touches
//! `buffer.contents` directly for host transfers). MPS has no copy API of its
//! own to call here, unlike CUDA's `cudaMemcpy`.
//!
//! So for MPS the device *is* the answer to "can the host reach this
//! memory": [`Device::Mps`] means unified memory, and the kernels below are
//! the CPU ones. The allocator only allocates and frees.

use crate::allocator::Allocator;
use crate::device::Device;

/// Copy `nbytes` from the host at `src` into MPS memory at `dst`.
///
/// # Safety
/// `dst` must be MPS memory valid for writing and `src` valid for reading
/// `nbytes`, and the two must not overlap.
pub(super) unsafe fn copy_h2d(
    _alloc: &dyn Allocator,
    _device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    // SAFETY: the caller's contract; MPS memory is host-addressable.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
}

/// Copy `nbytes` from MPS memory at `src` to the host at `dst`.
///
/// # Safety
/// As [`copy_h2d`], with the roles of source and destination swapped.
pub(super) unsafe fn copy_d2h(
    _alloc: &dyn Allocator,
    _device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    // SAFETY: the caller's contract; MPS memory is host-addressable.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
}
