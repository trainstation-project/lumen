//! CUDA streams: device and external-stream synchronization, cross-stream waits and
//! host-side holds, through the CUDA runtime.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
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

        pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
        pub fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
        pub fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;

        pub fn cudaGraphInstantiateWithFlags(
            exec: *mut *mut c_void,
            graph: *mut c_void,
            flags: u64,
        ) -> i32;

        pub fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaGraphDestroy(graph: *mut c_void) -> i32;
        pub fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
        pub fn cudaGetErrorString(err: i32) -> *const std::ffi::c_char;
    }
}

/// The stream lumen's CUDA kernels launch on: the default stream (null), or a graph's capture stream
/// while it captures.
static LAUNCH_STREAM: AtomicUsize = AtomicUsize::new(0);

/// The stream to launch lumen's CUDA kernels on, as a `cudaStream_t`.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn launch_stream() -> *mut c_void {
    LAUNCH_STREAM.load(Ordering::Acquire) as *mut c_void
}

/// `cudaStreamCaptureModeThreadLocal`: only this thread's unsafe calls invalidate the capture.
const CAPTURE_THREAD_LOCAL: i32 = 1;

/// `call`'s error `err` as a message, or `Ok` when it succeeded.
fn checked(err: i32, call: &str) -> Result<(), String> {
    match err {
        0 => Ok(()),
        e => {
            let message = unsafe { std::ffi::CStr::from_ptr(ffi::cudaGetErrorString(e)) };

            Err(format!("{call} failed: {}", message.to_string_lossy()))
        }
    }
}

/// A CUDA graph of lumen's kernel launches on one device: `begin` redirects them to a capture stream
/// (blocking, so ordered with the default stream), `end` instantiates what was launched, and `replay`
/// launches it all again. Memory allocated while capturing stays the graph's until it is dropped.
pub struct CudaGraph {
    device: usize,
    stream: usize,
    exec: usize,
    held: Vec<usize>,
}

impl CudaGraph {
    /// Starts capturing device `device`'s launches into a new graph.
    pub fn begin(device: usize) -> Result<CudaGraph, String> {
        CUDA.set_device(device);

        let mut stream = std::ptr::null_mut();

        checked(
            unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, 0) },
            "cudaStreamCreateWithFlags",
        )?;

        if let Err(e) = checked(
            unsafe { ffi::cudaStreamBeginCapture(stream, CAPTURE_THREAD_LOCAL) },
            "cudaStreamBeginCapture",
        ) {
            unsafe { ffi::cudaStreamDestroy(stream) };

            return Err(e);
        }

        crate::allocator::cuda::hold(device);
        LAUNCH_STREAM.store(stream as usize, Ordering::Release);

        Ok(CudaGraph {
            device,
            stream: stream as usize,
            exec: 0,
            held: Vec::new(),
        })
    }

    /// The capture stream, as a `cudaStream_t` integer (for launchers that take one).
    pub fn stream(&self) -> usize {
        self.stream
    }

    /// Ends the capture and instantiates the graph; errors (and keeps nothing) if a launch while
    /// capturing could not be captured.
    pub fn end(&mut self) -> Result<(), String> {
        LAUNCH_STREAM.store(0, Ordering::Release);
        self.held = crate::allocator::cuda::held(self.device);

        let mut graph = std::ptr::null_mut();

        checked(
            unsafe { ffi::cudaStreamEndCapture(self.stream as *mut c_void, &mut graph) },
            "cudaStreamEndCapture",
        )?;

        let mut exec = std::ptr::null_mut();
        let made = checked(
            unsafe { ffi::cudaGraphInstantiateWithFlags(&mut exec, graph, 0) },
            "cudaGraphInstantiateWithFlags",
        );

        unsafe { ffi::cudaGraphDestroy(graph) };
        made?;
        self.exec = exec as usize;

        Ok(())
    }

    /// Launches the captured graph on its stream.
    pub fn replay(&self) -> Result<(), String> {
        if self.exec == 0 {
            return Err("the CUDA graph was never captured".into());
        }

        CUDA.set_device(self.device);

        checked(
            unsafe { ffi::cudaGraphLaunch(self.exec as *mut c_void, self.stream as *mut c_void) },
            "cudaGraphLaunch",
        )
    }
}

/// Ends a capture still running, destroys the graph and its stream, and gives back the memory it held.
impl Drop for CudaGraph {
    fn drop(&mut self) {
        CUDA.set_device(self.device);

        if LAUNCH_STREAM.load(Ordering::Acquire) == self.stream {
            LAUNCH_STREAM.store(0, Ordering::Release);
            self.held = crate::allocator::cuda::held(self.device);

            let mut graph = std::ptr::null_mut();

            unsafe { ffi::cudaStreamEndCapture(self.stream as *mut c_void, &mut graph) };

            if !graph.is_null() {
                unsafe { ffi::cudaGraphDestroy(graph) };
            }
        }

        // The graph's last replay finishes before it and its memory go.
        unsafe { ffi::cudaStreamSynchronize(self.stream as *mut c_void) };

        if self.exec != 0 {
            unsafe { ffi::cudaGraphExecDestroy(self.exec as *mut c_void) };
        }

        unsafe { ffi::cudaStreamDestroy(self.stream as *mut c_void) };
        crate::allocator::cuda::release(self.device, &self.held);
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
