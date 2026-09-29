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

use crate::device::cuda::set_device;

mod ffi {
    use std::ffi::c_void;

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
        pub fn cudaEventDestroy(event: *mut c_void) -> i32;
    }
}

/// Wait until all work on CUDA device `device_index` has finished (PyTorch:
/// `torch.cuda.synchronize()`).
pub fn synchronize(device_index: usize) {
    set_device(device_index);
    unsafe { ffi::cudaDeviceSynchronize() };
}

/// `cudaEventDisableTiming`: the event only orders work.
const EVENT_DISABLE_TIMING: u32 = 2;

/// Make `stream` (a `cudaStream_t` handle) wait for the work queued so far
/// on lumen's stream on CUDA device `device_index`, without blocking the
/// host (PyTorch: an event recorded on the current stream, then
/// `stream.wait_event(event)`).
pub fn wait(device_index: usize, stream: usize) {
    set_device(device_index);
    let mut event = std::ptr::null_mut();
    let check = |err: i32, call: &str| {
        assert_eq!(err, 0, "{call} on cuda:{device_index} failed (error {err})");
    };
    unsafe {
        check(
            ffi::cudaEventCreateWithFlags(&mut event, EVENT_DISABLE_TIMING),
            "cudaEventCreateWithFlags",
        );
        // lumen's work runs on the legacy default stream.
        check(
            ffi::cudaEventRecord(event, std::ptr::null_mut()),
            "cudaEventRecord",
        );
        check(
            ffi::cudaStreamWaitEvent(stream as *mut c_void, event, 0),
            "cudaStreamWaitEvent",
        );
        // An event destroyed while pending is released once it completes.
        ffi::cudaEventDestroy(event);
    }
}
