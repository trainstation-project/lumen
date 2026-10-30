//! The MPS command stream: submission, completion callbacks and synchronization over the
//! Objective-C++ shim.
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};

use crate::Tensor;
use crate::device::Device;
use crate::profiler::GpuContext;
use crate::stream::event::Recorded;

unsafe extern "C" {
    /// Commits the open command buffer and returns the total buffers encoded so far.
    fn lumen_mps_stream_flush() -> u64;

    /// Total command buffers encoded so far.
    fn lumen_mps_stream_encoded_total() -> u64;

    /// Host time in seconds on the clock command buffer GPU times use.
    fn lumen_mps_stream_host_time() -> f64;

    /// The open command buffer, created if needed.
    fn lumen_mps_stream_command_buffer() -> *mut c_void;

    /// Queues a completion callback after the work encoded so far.
    fn lumen_mps_stream_mark(done: Completion, context: *mut c_void);

    /// Makes later encoders wait for earlier ones.
    fn lumen_mps_stream_fence();

    /// Encodes a GPU wait on a new shared event; work encoded after it stalls until the event is released.
    fn lumen_mps_stream_hold() -> *mut c_void;

    /// Signals a gate from `lumen_mps_stream_hold`, letting the stalled work run.
    fn lumen_mps_stream_release(gate: *mut c_void);
}

/// Shared event the stream waits on, released explicitly.
struct Gate(*mut c_void);

unsafe impl Send for Gate {}

impl Gate {
    /// Signals the event so the waiting work runs.
    fn release(self) {
        unsafe { lumen_mps_stream_release(self.0) }
    }
}

/// Stalls GPU work encoded from now on until the returned closure is called.
pub(crate) fn hold() -> Result<Box<dyn FnOnce() + Send>, String> {
    let gate = Gate(unsafe { lumen_mps_stream_hold() });

    if gate.0.is_null() {
        return Err("holding the MPS stream failed: no Metal device".into());
    }

    Ok(Box::new(move || gate.release()))
}

/// Orders later encoders after the work encoded so far.
pub(crate) fn fence() {
    unsafe { lumen_mps_stream_fence() };
}

/// The open Metal command buffer, as an `id<MTLCommandBuffer>` pointer.
pub fn command_buffer() -> *mut c_void {
    unsafe { lumen_mps_stream_command_buffer() }
}

/// Queues a no-op completion that counts toward `synchronize`.
pub(crate) fn mark() {
    let submission = Box::new(Submission {
        _tensors: Vec::new(),
        profile: None,
        event: None,
    });

    unsafe { lumen_mps_stream_mark(completed, Box::into_raw(submission).cast()) };
}

/// Commits the open command buffer.
pub(crate) fn flush() {
    unsafe { lumen_mps_stream_flush() };
}

/// Total command buffers encoded so far.
pub(crate) fn encoded() -> u64 {
    unsafe { lumen_mps_stream_encoded_total() }
}

/// Command buffers whose completion callbacks have run.
static FINISHED: Mutex<u64> = Mutex::new(0);
/// Signalled whenever `FINISHED` grows.
static COMPLETED: Condvar = Condvar::new();

/// Commits pending work and blocks until every encoded command buffer has completed.
pub fn synchronize() {
    let encoded = unsafe { lumen_mps_stream_flush() };
    let mut finished = FINISHED.lock().unwrap_or_else(|e| e.into_inner());

    while *finished < encoded {
        finished = COMPLETED.wait(finished).unwrap_or_else(|e| e.into_inner());
    }
}

/// The MPS device's stream.
pub struct MpsStream;

impl super::Stream for MpsStream {
    fn synchronize(&self) {
        synchronize()
    }
}

/// State kept alive until a command buffer completes.
struct Submission {
    /// Tensors the work reads or writes, held until it finishes.
    _tensors: Vec<Tensor>,
    /// Profiling info to report the work's GPU time.
    profile: Option<Profiled>,
    /// Event to mark complete.
    event: Option<Arc<Recorded>>,
}

/// What the profiler needs to place a completed kernel on its timeline.
struct Profiled {
    context: GpuContext,
    name: &'static str,
    kernel: String,
    /// Profiler clock at submission, in nanoseconds.
    profiler_ns: u64,
    /// Stream host time at submission, in seconds, to map GPU times onto the profiler clock.
    host_seconds: f64,
}

/// Command buffer completion callback: `(context, gpu_start, gpu_end, ok)`.
pub(crate) type Completion = unsafe extern "C" fn(*mut c_void, f64, f64, i32);

/// Prepares a completion context that keeps `tensors` alive and, when profiling, records the
/// kernel's GPU time; returns the context, its callback and whether timing is wanted.
pub(crate) fn submit(
    tensors: Vec<Tensor>,
    name: &'static str,
    kernel: &str,
) -> (*mut c_void, Completion, bool) {
    crate::kernels::launched::launched(kernel, || None);

    let profile = crate::profiler::gpu_context(Device::Mps(0)).map(|context| Profiled {
        context,
        name,
        kernel: kernel.to_owned(),
        profiler_ns: crate::profiler::now_ns(),
        host_seconds: unsafe { lumen_mps_stream_host_time() },
    });

    let timed = profile.is_some();

    let submission = Box::new(Submission {
        _tensors: tensors,
        profile,
        event: None,
    });

    (Box::into_raw(submission).cast(), completed, timed)
}

/// Completion context that sets `recorded` when the command buffer finishes.
pub(crate) fn submit_event(recorded: Arc<Recorded>) -> (*mut c_void, Completion) {
    let submission = Box::new(Submission {
        _tensors: Vec::new(),
        profile: None,
        event: Some(recorded),
    });

    (Box::into_raw(submission).cast(), completed)
}

/// Frees a completion context whose command buffer was never committed.
pub(crate) fn cancel(context: *mut c_void) {
    drop(unsafe { Box::from_raw(context.cast::<Submission>()) });
}

/// Completion callback: sets the event, reports profiling, releases the submission and counts
/// the buffer finished.
unsafe extern "C" fn completed(context: *mut c_void, gpu_start: f64, gpu_end: f64, ok: i32) {
    let submission = unsafe { Box::from_raw(context.cast::<Submission>()) };

    if ok == 0 {
        eprintln!("{}: an MPS command buffer failed", crate::LIBRARY_NAME);
    }

    if let Some(event) = &submission.event {
        event.set(gpu_start, gpu_end, ok != 0);
    }

    if let Some(p) = &submission.profile {
        let to_ns = |t: f64| (p.profiler_ns as f64 + (t - p.host_seconds) * 1e9).max(0.0) as u64;
        let (start, end) = (to_ns(gpu_start), to_ns(gpu_end));
        let kernel = Some(p.kernel.clone());
        let context = p.context.clone();
        crate::profiler::record_kernel_in(context, p.name, kernel, Device::Mps(0), start, end);
    }

    drop(submission);
    *FINISHED.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    COMPLETED.notify_all();
}
