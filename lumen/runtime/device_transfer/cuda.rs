//! CUDA host-to-device and device-to-host copies via `cudaMemcpyAsync` on the default stream: ordered after the
//! device work queued before them, without the host waiting for that work unless it reads the result.

use crate::Tensor;
use crate::device::Device;

/// `cudaMemcpyHostToDevice`.
const HOST_TO_DEVICE: i32 = 1;
/// `cudaMemcpyDeviceToHost`.
const DEVICE_TO_HOST: i32 = 2;

/// Copies host tensor `src` into CUDA tensor `dst` without waiting for the device: the copy runs after the work
/// queued before it, and the driver has staged `src` by the time this returns, so `src` may be freed.
pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    copy(dst.device(), dst, src, HOST_TO_DEVICE, false);
}

/// Copies CUDA tensor `src` into host tensor `dst`, waiting for it (and the work queued before it) to finish so
/// the host can read `dst`.
pub(super) fn copy_d2h(dst: &Tensor, src: &Tensor) {
    copy(src.device(), dst, src, DEVICE_TO_HOST, true);
}

/// Queues `cudaMemcpyAsync` on `device`'s default stream under the profiler's CUPTI correlation, then waits for
/// the stream when `wait`; panics on failure.
fn copy(device: Device, dst: &Tensor, src: &Tensor, kind: i32, wait: bool) {
    let Device::Cuda(index) = device else {
        unreachable!("the CUDA copy kernel runs on CUDA tensors")
    };

    let nbytes = dst.nbytes();
    let mut err = 0;

    crate::profiler::cupti::correlated(Device::Cuda(index), || {
        err = unsafe {
            crate::runtime::stream::cuda::ffi::cudaMemcpyAsync(
                dst.data_ptr().cast(),
                src.data_ptr().cast(),
                nbytes,
                kind,
                std::ptr::null_mut(),
            )
        };
    });

    assert_eq!(
        err, 0,
        "cudaMemcpyAsync of {nbytes} bytes (kind {kind}) on cuda:{index} failed (error {err})"
    );

    if wait {
        crate::runtime::stream::cuda::synchronize_external(index, 0);
    }
}

/// Starts copying CUDA tensor `src` into host tensor `dst` and returns the event recorded after the copy on the
/// default stream, to wait on before reading `dst`; when `dst` is not pinned (a copy into it would block anyway),
/// copies synchronously and returns `None`.
pub(super) fn copy_d2h_async(dst: &Tensor, src: &Tensor) -> Option<usize> {
    use crate::runtime::stream::cuda::{EVENT_DISABLE_TIMING, ffi};

    let Device::Cuda(index) = src.device() else {
        unreachable!("the CUDA copy kernel runs on CUDA tensors")
    };

    if !crate::allocator::pinned::is_pinned(dst.data_ptr()) {
        super::copy_d2h(dst, src);
        return None;
    }

    crate::device::cuda::CUDA.set_device(index);
    let _op = super::transfer(&super::COPY_D2H, dst, src, src.device());
    copy(src.device(), dst, src, DEVICE_TO_HOST, false);

    let mut event = std::ptr::null_mut();

    let made = unsafe { ffi::cudaEventCreateWithFlags(&mut event, EVENT_DISABLE_TIMING) };
    assert_eq!(
        made, 0,
        "cudaEventCreateWithFlags on cuda:{index} failed (error {made})"
    );

    let recorded = unsafe { ffi::cudaEventRecord(event, std::ptr::null_mut()) };
    assert_eq!(
        recorded, 0,
        "cudaEventRecord on cuda:{index} failed (error {recorded})"
    );

    Some(event as usize)
}

/// Blocks until `event` (from `copy_d2h_async` on device `index`) has completed, then destroys it.
pub(super) fn wait_event(index: usize, event: usize) {
    use crate::runtime::stream::cuda::ffi;
    crate::device::cuda::CUDA.set_device(index);

    let waited = unsafe { ffi::cudaEventSynchronize(event as *mut std::ffi::c_void) };
    unsafe { ffi::cudaEventDestroy(event as *mut std::ffi::c_void) };

    assert_eq!(
        waited, 0,
        "waiting for a copy to the host on cuda:{index} failed (error {waited})"
    );
}
