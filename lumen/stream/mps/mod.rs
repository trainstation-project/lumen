use std::ffi::c_void;
use std::sync::{Condvar, Mutex};

use crate::Tensor;
use crate::device::Device;
use crate::profiler::GpuContext;

unsafe extern "C" {
    fn lumen_mps_stream_flush() -> u64;
    fn lumen_mps_stream_encoded_total() -> u64;
    fn lumen_mps_stream_host_time() -> f64;
    fn lumen_mps_stream_command_buffer() -> *mut c_void;
    fn lumen_mps_stream_mark(done: Completion, context: *mut c_void);
}

pub fn command_buffer() -> *mut c_void {
    unsafe { lumen_mps_stream_command_buffer() }
}

pub(crate) fn mark() {
    let submission = Box::new(Submission {
        _tensors: Vec::new(),
        profile: None,
    });
    unsafe { lumen_mps_stream_mark(completed, Box::into_raw(submission).cast()) };
}

pub(crate) fn flush() {
    unsafe { lumen_mps_stream_flush() };
}

pub(crate) fn encoded() -> u64 {
    unsafe { lumen_mps_stream_encoded_total() }
}

static FINISHED: Mutex<u64> = Mutex::new(0);
static COMPLETED: Condvar = Condvar::new();

pub fn synchronize() {
    let encoded = unsafe { lumen_mps_stream_flush() };
    let mut finished = FINISHED.lock().unwrap_or_else(|e| e.into_inner());
    while *finished < encoded {
        finished = COMPLETED.wait(finished).unwrap_or_else(|e| e.into_inner());
    }
}

pub struct MpsStream;

impl super::Stream for MpsStream {
    fn synchronize(&self) {
        synchronize()
    }
}

struct Submission {
    _tensors: Vec<Tensor>,
    profile: Option<Profiled>,
}

struct Profiled {
    context: GpuContext,
    name: &'static str,
    kernel: String,
    profiler_ns: u64,
    host_seconds: f64,
}

pub(crate) type Completion = unsafe extern "C" fn(*mut c_void, f64, f64, i32);

pub(crate) fn submit(
    tensors: Vec<Tensor>,
    name: &'static str,
    kernel: &str,
) -> (*mut c_void, Completion, bool) {
    crate::ops::launched::launched(kernel, || None);
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
    });
    (Box::into_raw(submission).cast(), completed, timed)
}

pub(crate) fn cancel(context: *mut c_void) {
    drop(unsafe { Box::from_raw(context.cast::<Submission>()) });
}

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
        crate::profiler::record_kernel_in(context, p.name, kernel, Device::Mps(0), start, end);
    }
    drop(submission);
    *FINISHED.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    COMPLETED.notify_all();
}
