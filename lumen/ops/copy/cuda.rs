//! The CUDA copy kernels: `cudaMemcpy` in both directions, the runtime's
//! own path over the PCIe/NVLink bus, so the host never dereferences device
//! memory.

use std::ffi::c_void;

// `cudaMemcpyKind` values.
const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

/// Copy `nbytes` from the host at `src` into CUDA device memory at `dst`.
///
/// # Safety
/// `dst` must be device memory valid for writing and `src` host memory valid
/// for reading `nbytes`, and the two must not overlap.
pub(super) unsafe fn copy_h2d(dst: *mut u8, src: *const u8, nbytes: usize) {
    copy(dst, src, nbytes, HOST_TO_DEVICE);
}

/// Copy `nbytes` from CUDA device memory at `src` to the host at `dst`.
///
/// # Safety
/// As [`copy_h2d`], with the roles of source and destination swapped.
pub(super) unsafe fn copy_d2h(dst: *mut u8, src: *const u8, nbytes: usize) {
    copy(dst, src, nbytes, DEVICE_TO_HOST);
}

/// `cudaMemcpy` of `nbytes` in direction `kind`, panicking on error.
/// Blocks until the copy is done (the runtime's synchronous path, as
/// PyTorch's blocking `copy_`), so the bytes are there when it returns.
fn copy(dst: *mut u8, src: *const u8, nbytes: usize, kind: i32) {
    // SAFETY: the caller guarantees both ranges are valid for `nbytes`.
    let err = unsafe { cudaMemcpy(dst.cast(), src.cast(), nbytes, kind) };
    assert_eq!(
        err, 0,
        "cudaMemcpy of {nbytes} bytes (kind {kind}) failed (error {err})"
    );
}
