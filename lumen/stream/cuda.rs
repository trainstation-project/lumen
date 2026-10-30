//! CUDA streams: device and external-stream synchronization, cross-stream waits and
//! host-side holds, through the CUDA runtime.

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};

use crate::device::cuda::CUDA;

/// Bindings to the CUDA runtime calls used for streams and events.
pub(crate) mod ffi {
    use std::ffi::c_void;

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
        pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
        pub fn cudaEventDestroy(event: *mut c_void) -> i32;
        pub fn cudaEventQuery(event: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(event: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;

        pub fn cudaLaunchHostFunc(
            stream: *mut c_void,
            host: extern "C" fn(*mut c_void),
            data: *mut c_void,
        ) -> i32;
    }
}

/// lumen's stream on CUDA device `index`, the device's default stream.
pub struct CudaStream {
    pub index: usize,
}

impl super::Stream for CudaStream {
    /// Blocks the host until all work on the device finishes.
    fn synchronize(&self) {
        synchronize(self.index)
    }
}

impl CudaStream {
    /// Makes external stream `stream` (a `cudaStream_t` as an integer) wait for the work
    /// queued so far on lumen's stream, without blocking the host.
    pub fn wait(&self, stream: usize) {
        wait(self.index, stream)
    }
}

/// Blocks the host until all work on CUDA device `device_index` finishes.
pub fn synchronize(device_index: usize) {
    CUDA.set_device(device_index);

    unsafe { ffi::cudaDeviceSynchronize() };
}

/// Blocks the host until external stream `stream` (a `cudaStream_t` as an integer) drains.
pub fn synchronize_external(device_index: usize, stream: usize) {
    CUDA.set_device(device_index);

    unsafe { ffi::cudaStreamSynchronize(stream as *mut c_void) };
}

/// `cudaEventDisableTiming`: the event only orders work, so skip recording timestamps.
pub(crate) const EVENT_DISABLE_TIMING: u32 = 2;

/// Makes external stream `stream` on device `device_index` wait for the work queued so far
/// on the default stream, without blocking the host; panics if a CUDA call fails.
pub fn wait(device_index: usize, stream: usize) {
    CUDA.set_device(device_index);

    let mut event = std::ptr::null_mut();

    let check = |err: i32, call: &str| {
        assert_eq!(err, 0, "{call} on cuda:{device_index} failed (error {err})");
    };

    // Record an event on the default stream and have the external stream wait on it; CUDA
    // defers freeing the destroyed event until that wait completes.
    unsafe {
        check(
            ffi::cudaEventCreateWithFlags(&mut event, EVENT_DISABLE_TIMING),
            "cudaEventCreateWithFlags",
        );

        check(
            ffi::cudaEventRecord(event, std::ptr::null_mut()),
            "cudaEventRecord",
        );

        check(
            ffi::cudaStreamWaitEvent(stream as *mut c_void, event, 0),
            "cudaStreamWaitEvent",
        );

        ffi::cudaEventDestroy(event);
    }
}

/// The release flag and its condvar, shared by the parked host function and the releaser.
type Gate = (Mutex<bool>, Condvar);

/// CUDA host function that parks the stream until its gate is released; `data` is an
/// `Arc<Gate>` handed over by `hold`, whose reference it takes ownership of.
extern "C" fn wait_for_release(data: *mut c_void) {
    let gate = unsafe { Arc::from_raw(data.cast_const().cast::<Gate>()) };
    let mut released = gate.0.lock().unwrap_or_else(|e| e.into_inner());

    while !*released {
        released = gate.1.wait(released).unwrap_or_else(|e| e.into_inner());
    }
}

/// Enqueues a host function that stalls device `device_index`'s default stream until the
/// returned closure is called.
pub(crate) fn hold(device_index: usize) -> Result<Box<dyn FnOnce() + Send>, String> {
    CUDA.set_device(device_index);
    let gate: Arc<Gate> = Arc::new((Mutex::new(false), Condvar::new()));
    let data = Arc::into_raw(gate.clone()).cast_mut().cast::<c_void>();
    let err = unsafe { ffi::cudaLaunchHostFunc(std::ptr::null_mut(), wait_for_release, data) };

    if err != 0 {
        // The host function never runs, so reclaim the reference handed to it.
        drop(unsafe { Arc::from_raw(data.cast_const().cast::<Gate>()) });

        return Err(format!(
            "cudaLaunchHostFunc on cuda:{device_index} failed (error {err})"
        ));
    }

    Ok(Box::new(move || {
        *gate.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
        gate.1.notify_all();
    }))
}
