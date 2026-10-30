//! CUDA host-to-device and device-to-host copies via `cudaMemcpy`.

use std::ffi::c_void;

use crate::Tensor;
use crate::device::Device;

/// `cudaMemcpyHostToDevice`.
const HOST_TO_DEVICE: i32 = 1;
/// `cudaMemcpyDeviceToHost`.
const DEVICE_TO_HOST: i32 = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    /// Synchronous CUDA runtime copy of `count` bytes in direction `kind`.
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

/// Copies host tensor `src` into CUDA tensor `dst`.
pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    copy(dst.device(), dst, src, HOST_TO_DEVICE);
}

/// Copies CUDA tensor `src` into host tensor `dst`.
pub(super) fn copy_d2h(dst: &Tensor, src: &Tensor) {
    copy(src.device(), dst, src, DEVICE_TO_HOST);
}

/// Runs `cudaMemcpy` on `device` under the profiler's CUPTI correlation; panics on failure.
fn copy(device: Device, dst: &Tensor, src: &Tensor, kind: i32) {
    let Device::Cuda(index) = device else {
        unreachable!("the CUDA copy kernel runs on CUDA tensors")
    };
    let nbytes = dst.nbytes();
    let mut err = 0;
    crate::profiler::cupti::correlated(Device::Cuda(index), || {
        err = unsafe { cudaMemcpy(dst.data_ptr().cast(), src.data_ptr().cast(), nbytes, kind) };
    });
    assert_eq!(
        err, 0,
        "cudaMemcpy of {nbytes} bytes (kind {kind}) on cuda:{index} failed (error {err})"
    );
}
