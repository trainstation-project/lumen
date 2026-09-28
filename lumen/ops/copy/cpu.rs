//! The CPU copy kernels: an ordinary `memcpy` in both directions. Host
//! memory and CPU storage are the same address space, and this kernel is
//! also the fallback where a device's memory is host-addressable (MPS's
//! unified buffers) and has no dedicated kernel of its own.

/// Copy `nbytes` from the host at `src` into host memory at `dst`.
///
/// # Safety
/// `dst` must be valid for writing and `src` for reading `nbytes`, and the
/// two must not overlap.
pub(super) unsafe fn copy_h2d(dst: *mut u8, src: *const u8, nbytes: usize) {
    // SAFETY: the caller's contract.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
}

/// Copy `nbytes` from host memory at `src` to the host at `dst`.
///
/// # Safety
/// As [`copy_h2d`].
pub(super) unsafe fn copy_d2h(dst: *mut u8, src: *const u8, nbytes: usize) {
    // SAFETY: the caller's contract.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
}
