//! lumen's CUDA stream (PyTorch: `CUDAStream`). CUDA ops launch on the
//! device's default (legacy) stream, as PyTorch's do by default, and return
//! without waiting: work on one stream runs in order, and `cudaMemcpy` (the
//! allocator's host copies) and `cudaFree` wait for it, so the host never
//! needs to wait after a kernel.
//!
//! When the profiler times CUDA, CUPTI records the work (see
//! `profiler/cupti.rs`), as kineto does for PyTorch: nothing is added to the
//! stream.

use std::ffi::c_void;

use crate::allocator::cuda::ffi as cudart;

mod ffi {
    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaGetDevice(device: *mut i32) -> i32;
    }

    #[link(name = "cuda")]
    unsafe extern "C" {
        pub fn cuCtxGetCurrent(ctx: *mut *mut std::ffi::c_void) -> i32;
    }
}

/// Make `device_index` the current device, skipping the call if it already
/// is (PyTorch: `c10::cuda::SetDevice`). On a fresh thread `cudaGetDevice`
/// reports device 0 with no context current, which the driver API calls
/// launched here need, so the call is skipped only once one is.
fn set_device(device_index: usize) {
    let mut current = -1;
    let mut context = std::ptr::null_mut();
    unsafe {
        if ffi::cudaGetDevice(&mut current) != 0
            || current != device_index as i32
            || ffi::cuCtxGetCurrent(&mut context) != 0
            || context.is_null()
        {
            cudart::cudaSetDevice(device_index as i32);
        }
    }
}

/// Wait until all work on CUDA device `device_index` has finished (PyTorch:
/// `torch.cuda.synchronize()`).
pub fn synchronize(device_index: usize) {
    set_device(device_index);
    unsafe { ffi::cudaDeviceSynchronize() };
}

/// Run `work` on CUDA device `device_index`, passing it the stream to
/// launch on, without waiting for it. When the profiler times CUDA, CUPTI
/// records the work, tagged with the current op.
pub(crate) fn launch(device_index: usize, work: impl FnOnce(*mut c_void)) {
    set_device(device_index);
    // The legacy default stream, which cudaMemcpy runs on too.
    let stream = std::ptr::null_mut();
    #[cfg(lumen_cupti_linked)]
    return {
        let device = crate::device::Device::Cuda(device_index);
        crate::profiler::cupti::correlated(device, || work(stream))
    };
    #[cfg(not(lumen_cupti_linked))]
    work(stream)
}
