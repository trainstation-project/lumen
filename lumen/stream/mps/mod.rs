//! lumen's MPS stream (PyTorch: `MPSStream`). MPS ops encode GPU work into
//! the stream's open command buffer (`mps.mm`) and return without waiting;
//! it is committed every few ops, and the host waits only in
//! [`synchronize`], which `Storage` calls before it touches MPS memory
//! (PyTorch likewise syncs its stream before host copies).
//!
//! Submitted work keeps its tensors, and so their memory, alive until the GPU
//! is done with it: a block cannot go back to the cache, nor its segment to
//! Metal, while work on it is in flight (PyTorch's allocator checks buffers'
//! Metal retain counts instead). Its profiler event is recorded when it
//! completes, attributed to the op that submitted it.

use std::ffi::c_void;
use std::sync::{Condvar, Mutex};

use crate::Tensor;
use crate::device::Device;
use crate::profiler::GpuContext;

unsafe extern "C" {
    // In mps.mm (which also holds the queue the shims submit to).
    fn lumen_mps_stream_flush() -> u64;
    fn lumen_mps_stream_host_time() -> f64;
    fn lumen_mps_stream_command_buffer() -> *mut c_void;
    fn lumen_mps_stream_mark(done: Completion, context: *mut c_void);
}

/// The stream's open `MTLCommandBuffer` (made if none is), its compute
/// encoder ended, for work of the caller's own: encoded into it, that runs
/// after the work submitted so far and before what is submitted after.
/// Valid until the stream commits it (the next [`synchronize`], or op that
/// fills it). Null without a Metal device.
pub fn command_buffer() -> *mut c_void {
    unsafe { lumen_mps_stream_command_buffer() }
}

/// Count work encoded into the open command buffer outside lumen's ops (a
/// custom op's): [`synchronize`] waits for it from now on.
pub(crate) fn mark() {
    let submission = Box::new(Submission {
        _tensors: Vec::new(),
        profile: None,
    });
    unsafe { lumen_mps_stream_mark(completed, Box::into_raw(submission).cast()) };
}

/// How many encoded ops the GPU has finished.
static FINISHED: Mutex<u64> = Mutex::new(0);
static COMPLETED: Condvar = Condvar::new();

/// Wait until all MPS work submitted so far has finished (PyTorch:
/// `torch.mps.synchronize()`). Waits only for the ops encoded before its
/// flush, which the flush commits: work other threads encode afterwards
/// may sit in the next command buffer until someone flushes again.
pub fn synchronize() {
    let encoded = unsafe { lumen_mps_stream_flush() };
    let mut finished = FINISHED.lock().unwrap_or_else(|e| e.into_inner());
    while *finished < encoded {
        finished = COMPLETED.wait(finished).unwrap_or_else(|e| e.into_inner());
    }
}

/// One submission in flight: handed to Metal as the completion context.
struct Submission {
    /// Keeps the tensors' storage alive until the GPU is done with it.
    _tensors: Vec<Tensor>,
    profile: Option<Profiled>,
}

/// What the profiler needs, captured at submission: who submitted the work,
/// and one reading of both clocks to map GPU times onto the profiler's.
struct Profiled {
    context: GpuContext,
    name: &'static str,
    kernel: String,
    profiler_ns: u64,
    host_seconds: f64,
}

/// The completion callback a shim calls with a [`submit`] context once the
/// GPU finishes: `ok` is 0 if the command buffer failed.
pub(crate) type Completion = unsafe extern "C" fn(*mut c_void, f64, f64, i32);

/// Register work on `tensors` about to be submitted, recorded in the profiler
/// as `name`, running Metal function `kernel` (copied only while profiling).
/// Pass the returned context and [`completed`] to the shim, which
/// must call it exactly once; if the shim does not submit, call [`cancel`].
/// The flag is whether the profiler times it (the shim samples its GPU
/// start and end).
pub(crate) fn submit(
    tensors: Vec<Tensor>,
    name: &'static str,
    kernel: &str,
) -> (*mut c_void, Completion, bool) {
    let profile = crate::profiler::gpu_context(Device::Mps).map(|context| Profiled {
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
    });
    (Box::into_raw(submission).cast(), completed, timed)
}

/// Undo a [`submit`] whose work was not submitted.
pub(crate) fn cancel(context: *mut c_void) {
    drop(unsafe { Box::from_raw(context.cast::<Submission>()) });
}

/// Called on a Metal thread when submitted work completes.
unsafe extern "C" fn completed(context: *mut c_void, gpu_start: f64, gpu_end: f64, ok: i32) {
    let submission = unsafe { Box::from_raw(context.cast::<Submission>()) };
    if ok == 0 {
        eprintln!("{}: an MPS command buffer failed", crate::LIBRARY_NAME);
    }
    if let Some(p) = &submission.profile {
        let to_ns = |t: f64| (p.profiler_ns as f64 + (t - p.host_seconds) * 1e9).max(0.0) as u64;
        let (start, end) = (to_ns(gpu_start), to_ns(gpu_end));
        let kernel = Some(p.kernel.clone());
        let context = p.context.clone();
        crate::profiler::record_kernel_in(context, p.name, kernel, Device::Mps, start, end);
    }
    drop(submission); // may free the tensors' blocks
    *FINISHED.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    COMPLETED.notify_all();
}
