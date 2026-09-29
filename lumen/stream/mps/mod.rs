//! lumen's MPS stream (PyTorch: `MPSStream`). MPS ops encode GPU work into
//! the stream's open command buffer (`mps.mm`) and return without waiting;
//! it is committed every few ops, and the host waits only in
//! [`synchronize`], which `Storage` calls before it touches MPS memory
//! (PyTorch likewise syncs its stream before host copies).
//!
//! Submitted work keeps its tensor, and so its memory, alive until the GPU
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
    fn lumen_mps_stream_flush();
    fn lumen_mps_stream_host_time() -> f64;
}

/// Submitted work whose completion handler has not run yet.
static PENDING: Mutex<usize> = Mutex::new(0);
static COMPLETED: Condvar = Condvar::new();

/// Wait until all submitted MPS work has finished (PyTorch:
/// `torch.mps.synchronize()`).
pub fn synchronize() {
    unsafe { lumen_mps_stream_flush() };
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    while *pending > 0 {
        pending = COMPLETED.wait(pending).unwrap_or_else(|e| e.into_inner());
    }
}

/// One submission in flight: handed to Metal as the completion context.
struct Submission {
    /// Keeps the tensor's storage alive until the GPU is done with it.
    _tensor: Tensor,
    profile: Option<Profiled>,
}

/// What the profiler needs, captured at submission: who submitted the work,
/// and one reading of both clocks to map GPU times onto the profiler's.
struct Profiled {
    context: GpuContext,
    name: &'static str,
    profiler_ns: u64,
    host_seconds: f64,
}

/// The completion callback a shim calls with a [`submit`] context once the
/// GPU finishes: `ok` is 0 if the command buffer failed.
pub(crate) type Completion = unsafe extern "C" fn(*mut c_void, f64, f64, i32);

/// Register work on `tensor` about to be submitted, recorded in the profiler
/// as `name`. Pass the returned context and [`completed`] to the shim, which
/// must call it exactly once; if the shim does not submit, call [`cancel`].
/// The flag is whether the profiler times it (the shim samples its GPU
/// start and end).
pub(crate) fn submit(tensor: &Tensor, name: &'static str) -> (*mut c_void, Completion, bool) {
    let profile = crate::profiler::gpu_context(Device::Mps).map(|context| Profiled {
        context,
        name,
        profiler_ns: crate::profiler::now_ns(),
        host_seconds: unsafe { lumen_mps_stream_host_time() },
    });
    let timed = profile.is_some();
    *PENDING.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    let submission = Box::new(Submission {
        _tensor: tensor.clone(),
        profile,
    });
    (Box::into_raw(submission).cast(), completed, timed)
}

/// Undo a [`submit`] whose work was not submitted.
pub(crate) fn cancel(context: *mut c_void) {
    finish(unsafe { Box::from_raw(context.cast::<Submission>()) });
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
        crate::profiler::record_gpu_in(p.context, p.name, Device::Mps, start, end);
    }
    finish(submission);
}

fn finish(submission: Box<Submission>) {
    drop(submission); // may free the tensor's block
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    *pending -= 1;
    if *pending == 0 {
        COMPLETED.notify_all();
    }
}
