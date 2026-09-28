//! lumen's CUDA stream (PyTorch: `CUDAStream`). CUDA ops launch on the
//! device's default (legacy) stream, as PyTorch's do by default, and return
//! without waiting: work on one stream runs in order, and `cudaMemcpy` (the
//! allocator's host copies) and `cudaFree` wait for it, so the host never
//! needs to wait after a kernel.
//!
//! When the profiler times CUDA, a launch is bracketed with CUDA events that
//! are read only when the profiler stops ([`flush`]), not after each kernel.

use std::ffi::c_void;
use std::sync::Mutex;

use crate::allocator::cuda::ffi as cudart;
use crate::device::Device;
use crate::profiler::GpuContext;

mod ffi {
    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaDeviceSynchronize() -> i32;
    }
}

/// Wait until all work on CUDA device `device_index` has finished (PyTorch:
/// `torch.cuda.synchronize()`).
pub fn synchronize(device_index: usize) {
    unsafe {
        cudart::cudaSetDevice(device_index as i32);
        ffi::cudaDeviceSynchronize();
    }
}

/// A CUDA event, kept as an address so it can live in a `static`.
#[derive(Clone, Copy)]
struct Event(usize);

impl Event {
    /// A new event on the current device.
    fn new() -> Option<Self> {
        let mut event = std::ptr::null_mut();
        (unsafe { cudart::cudaEventCreate(&mut event) } == 0).then(|| Event(event.addr()))
    }

    fn raw(self) -> *mut c_void {
        std::ptr::without_provenance_mut(self.0)
    }

    fn record(self, stream: *mut c_void) -> bool {
        unsafe { cudart::cudaEventRecord(self.raw(), stream) == 0 }
    }

    fn synchronize(self) -> bool {
        unsafe { cudart::cudaEventSynchronize(self.raw()) == 0 }
    }

    /// Nanoseconds from `self` to `later`, both completed.
    fn ns_until(self, later: Event) -> Option<u64> {
        let mut ms = 0.0f32;
        let ok = unsafe { cudart::cudaEventElapsedTime(&mut ms, self.raw(), later.raw()) } == 0;
        ok.then(|| (f64::from(ms) * 1e6).max(0.0) as u64)
    }

    fn destroy(self) {
        unsafe { cudart::cudaEventDestroy(self.raw()) };
    }
}

/// Timed work whose events have not been read yet.
struct Timed {
    context: GpuContext,
    name: &'static str,
    device_index: usize,
    /// The session's reference event on the device and its profiler time.
    reference: (Event, u64),
    start: Event,
    stop: Event,
}

static PENDING: Mutex<Vec<Timed>> = Mutex::new(Vec::new());

/// Run `work` on CUDA device `device_index`, passing it the stream to
/// launch on, without waiting for it. When the profiler times CUDA, the
/// work is recorded as `name` once [`flush`] reads its events.
pub(crate) fn launch(device_index: usize, name: &'static str, work: impl FnOnce(*mut c_void)) {
    unsafe { cudart::cudaSetDevice(device_index as i32) };
    // The legacy default stream, which cudaMemcpy runs on too.
    let stream = std::ptr::null_mut();
    let context = crate::profiler::gpu_context(Device::Cuda(device_index));
    let events = context
        .zip(reference(device_index))
        .zip(Event::new())
        .zip(Event::new());
    let Some((((context, reference), start), stop)) = events else {
        return work(stream);
    };
    start.record(stream);
    work(stream);
    stop.record(stream);
    let timed = Timed {
        context,
        name,
        device_index,
        reference,
        start,
        stop,
    };
    PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(timed);
}

/// Wait for timed work and record it in the profiler (called when the
/// profiler stops).
pub(crate) fn flush() {
    let pending = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    for t in pending {
        unsafe { cudart::cudaSetDevice(t.device_index as i32) };
        let (reference, reference_ns) = t.reference;
        let times = reference
            .ns_until(t.start)
            .zip(reference.ns_until(t.stop))
            .filter(|_| t.stop.synchronize());
        t.start.destroy();
        t.stop.destroy();
        if let Some((start, stop)) = times {
            let device = Device::Cuda(t.device_index);
            let (start, stop) = (reference_ns + start, reference_ns + stop);
            crate::profiler::record_gpu_in(t.context, t.name, device, start, stop);
        }
    }
}

/// An event recorded once per profiler session on the device, with the
/// profiler time it completed at: GPU times are measured from it, which puts
/// them on the profiler clock. Recording it is the one wait per session.
fn reference(device_index: usize) -> Option<(Event, u64)> {
    /// (device index, session, event, profiler ns).
    static REFERENCES: Mutex<Vec<(usize, u64, Event, u64)>> = Mutex::new(Vec::new());
    let session = crate::profiler::session_id();
    let mut references = REFERENCES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = references.iter().position(|r| r.0 == device_index) {
        let (_, s, event, ns) = references[i];
        if s == session {
            return Some((event, ns));
        }
        event.destroy(); // from an earlier session
        references.swap_remove(i);
    }
    let event = Event::new()?;
    if !(event.record(std::ptr::null_mut()) && event.synchronize()) {
        event.destroy();
        return None;
    }
    let ns = crate::profiler::now_ns();
    references.push((device_index, session, event, ns));
    Some((event, ns))
}
