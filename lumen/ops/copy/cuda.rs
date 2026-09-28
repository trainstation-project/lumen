//! The CUDA copy kernels: `cudaMemcpy` in both directions, the runtime's
//! own path over the PCIe/NVLink bus, so the host never dereferences device
//! memory.
//!
//! The copy runs inside [`profiler::cupti::correlated`], which pushes an
//! external correlation id around it: that is what attributes the copy to
//! the op that issued it, so the profiler can name it `Memcpy HtoD`/`DtoH`
//! (`profiler::cupti::name`) and hang it off the right op.

use std::ffi::c_void;

use crate::allocator::Allocator;
use crate::device::Device;

// `cudaMemcpyKind` values.
const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

/// Copy `nbytes` from the host at `src` into CUDA device memory at `dst`.
///
/// # Safety
/// `dst` must be device memory valid for writing and `src` host memory valid
/// for reading `nbytes`, and the two must not overlap.
pub(super) unsafe fn copy_h2d(
    _alloc: &dyn Allocator,
    device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    copy(device, dst, src, nbytes, HOST_TO_DEVICE);
}

/// Copy `nbytes` from CUDA device memory at `src` to the host at `dst`.
///
/// # Safety
/// As [`copy_h2d`], with the roles of source and destination swapped.
pub(super) unsafe fn copy_d2h(
    _alloc: &dyn Allocator,
    device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    copy(device, dst, src, nbytes, DEVICE_TO_HOST);
}

/// `cudaMemcpy` of `nbytes` in direction `kind`, panicking on error and
/// tagged with the current op when the profiler times `device`.
///
/// Blocks until the copy is done (the runtime's synchronous path, as
/// PyTorch's blocking `copy_`), so the bytes are there when it returns.
/// `cudaMemcpy` itself sets the device context from the pointers, so unlike
/// a kernel launch there is no `cudaSetDevice` to match.
fn copy(device: Device, dst: *mut u8, src: *const u8, nbytes: usize, kind: i32) {
    let Device::Cuda(index) = device else {
        unreachable!("the CUDA copy kernel runs on CUDA tensors")
    };
    let mut err = 0;
    crate::profiler::cupti::correlated(Device::Cuda(index), || {
        err = unsafe { cudaMemcpy(dst.cast(), src.cast(), nbytes, kind) };
    });
    assert_eq!(
        err, 0,
        "cudaMemcpy of {nbytes} bytes (kind {kind}) on cuda:{index} failed (error {err})"
    );
}
