//! CUDA device timing with CUPTI's activity API, as PyTorch's profiler does
//! through kineto: while a session times CUDA, the driver itself records
//! each memset's and copy's GPU start and end into buffers CUPTI hands back,
//! so nothing is added to the stream and a launch costs only a correlation
//! push and pop. Compiled only when build.rs finds `libcupti` (cfg
//! `lumen_cupti_linked`).
//!
//! Records are tied to the op that launched them like kineto does: each
//! launch pushes an external correlation id, and CUPTI emits a record
//! mapping it to the launch's own correlation id, which its memset or copy
//! record carries. They are read when the session stops ([`stop`]).

use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once};

use super::GpuContext;
use crate::device::Device;

mod ffi {
    use std::ffi::c_void;

    pub type BufferRequested = unsafe extern "C" fn(*mut *mut u8, *mut usize, *mut usize);
    pub type BufferCompleted = unsafe extern "C" fn(*mut c_void, u32, *mut u8, usize, usize);

    // `CUptiResult` (0 is success) and the enums are 32-bit.
    #[link(name = "cupti")]
    unsafe extern "C" {
        pub fn cuptiActivityEnable(kind: u32) -> i32;
        pub fn cuptiActivityDisable(kind: u32) -> i32;
        pub fn cuptiActivityRegisterCallbacks(
            requested: BufferRequested,
            completed: BufferCompleted,
        ) -> i32;
        pub fn cuptiActivityFlushAll(flag: u32) -> i32;
        pub fn cuptiActivityGetNextRecord(
            buffer: *mut u8,
            valid: usize,
            record: *mut *mut u8,
        ) -> i32;
        pub fn cuptiActivityPushExternalCorrelationId(kind: u32, id: u64) -> i32;
        pub fn cuptiActivityPopExternalCorrelationId(kind: u32, last: *mut u64) -> i32;
        pub fn cuptiGetTimestamp(timestamp: *mut u64) -> i32;
    }

    // `CUpti_ActivityKind`.
    pub const KIND_MEMCPY: u32 = 1;
    pub const KIND_MEMSET: u32 = 2;
    pub const KIND_RUNTIME: u32 = 4;
    pub const KIND_DRIVER: u32 = 5;
    pub const KIND_EXTERNAL_CORRELATION: u32 = 39;
    // `CUpti_ExternalCorrelationKind`.
    pub const EXTERNAL_CUSTOM0: u32 = 3;
    // `CUPTI_ACTIVITY_FLAG_FLUSH_FORCED`.
    pub const FLUSH_FORCED: u32 = 1;
}

/// The activities recorded while a session times CUDA. CUPTI emits an
/// external correlation record only for an API call it records, so the
/// runtime (`cudaMemcpy`) and driver (`cuMemset*`) APIs are recorded too
/// (kineto enables them as well); their records are dropped.
const KINDS: [u32; 5] = [
    ffi::KIND_MEMCPY,
    ffi::KIND_MEMSET,
    ffi::KIND_RUNTIME,
    ffi::KIND_DRIVER,
    ffi::KIND_EXTERNAL_CORRELATION,
];

/// What is kept of a CUPTI activity record.
enum Record {
    /// A memset or copy: its CUPTI kind and copy kind, GPU times (CUPTI
    /// ns), device and correlation id.
    Work {
        kind: u32,
        copy_kind: u8,
        start: u64,
        end: u64,
        device: u32,
        correlation: u32,
    },
    /// Which launch (correlation id) ran under one of our external ids.
    External { id: u64, correlation: u32 },
}

impl Record {
    /// Read the fields used from a record CUPTI returned. `CUpti_ActivityMemset4`
    /// and `CUpti_ActivityMemcpy6` (and their earlier versions) share the
    /// offsets of start, end, deviceId and correlationId; the copy kind is
    /// the memcpy record's fifth byte.
    unsafe fn read(record: *const u8) -> Option<Record> {
        let at = |offset: usize| unsafe { record.add(offset) };
        let u32_at = |offset| unsafe { at(offset).cast::<u32>().read_unaligned() };
        let u64_at = |offset| unsafe { at(offset).cast::<u64>().read_unaligned() };
        match u32_at(0) {
            kind @ (ffi::KIND_MEMCPY | ffi::KIND_MEMSET) => Some(Record::Work {
                kind,
                copy_kind: unsafe { at(4).read() },
                start: u64_at(16),
                end: u64_at(24),
                device: u32_at(32),
                correlation: u32_at(44),
            }),
            // `CUpti_ActivityExternalCorrelation`.
            ffi::KIND_EXTERNAL_CORRELATION if u32_at(4) == ffi::EXTERNAL_CUSTOM0 => {
                Some(Record::External {
                    id: u64_at(8),
                    correlation: u32_at(16),
                })
            }
            _ => None,
        }
    }

    /// The profiler's name for a memset or copy (as lumen named them).
    fn name(kind: u32, copy_kind: u8) -> &'static str {
        match (kind, copy_kind) {
            (ffi::KIND_MEMSET, _) => "Memset",
            (_, 1) => "Memcpy HtoD",
            (_, 2) => "Memcpy DtoH",
            (_, 8) => "Memcpy DtoD",
            _ => "Memcpy",
        }
    }
}

/// Records CUPTI has handed back.
static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());
/// External id -> the op that launched under it.
static CONTEXTS: Mutex<Option<HashMap<u64, GpuContext>>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// The running session's clock pair: (CUPTI ns, profiler ns).
static CLOCK: Mutex<Option<(u64, u64)>> = Mutex::new(None);

const BUFFER: Layout = match Layout::from_size_align(1 << 20, 8) {
    Ok(layout) => layout,
    Err(_) => panic!("CUPTI buffer layout"),
};

unsafe extern "C" fn buffer_requested(buffer: *mut *mut u8, size: *mut usize, max: *mut usize) {
    unsafe {
        *buffer = std::alloc::alloc(BUFFER);
        *size = if (*buffer).is_null() {
            0
        } else {
            BUFFER.size()
        };
        *max = 0; // as many records as fit
    }
}

unsafe extern "C" fn buffer_completed(
    _context: *mut c_void,
    _stream: u32,
    buffer: *mut u8,
    _size: usize,
    valid: usize,
) {
    if buffer.is_null() {
        return;
    }
    let mut read = Vec::new();
    let mut record = std::ptr::null_mut();
    while unsafe { ffi::cuptiActivityGetNextRecord(buffer, valid, &mut record) } == 0 {
        read.extend(unsafe { Record::read(record) });
    }
    RECORDS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend(read);
    unsafe { std::alloc::dealloc(buffer, BUFFER) };
}

/// Start recording CUDA activity (a session with the CUDA activity starts).
pub(crate) fn start() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| unsafe {
        ffi::cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    });
    for kind in KINDS {
        unsafe { ffi::cuptiActivityEnable(kind) };
    }
    let mut cupti_ns = 0;
    unsafe { ffi::cuptiGetTimestamp(&mut cupti_ns) };
    *CLOCK.lock().unwrap_or_else(|e| e.into_inner()) = Some((cupti_ns, super::now_ns()));
}

/// Run `work`, which launches CUDA work on `device`, tagged with the
/// current op when the profiler times the device.
pub(crate) fn correlated<R>(device: Device, work: impl FnOnce() -> R) -> R {
    let Some(context) = super::gpu_context(device) else {
        return work();
    };
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(id, context);
    unsafe { ffi::cuptiActivityPushExternalCorrelationId(ffi::EXTERNAL_CUSTOM0, id) };
    let result = work();
    let mut last = 0;
    unsafe { ffi::cuptiActivityPopExternalCorrelationId(ffi::EXTERNAL_CUSTOM0, &mut last) };
    result
}

/// Wait for CUDA work, stop recording, and add what ran to the session (the
/// session stops; PyTorch's profiler also synchronizes CUDA on exit).
pub(crate) fn stop() {
    let Some((cupti_ns, profiler_ns)) = CLOCK.lock().unwrap_or_else(|e| e.into_inner()).take()
    else {
        return;
    };
    for device_index in 0..crate::allocator::cuda::device_count() {
        crate::stream::cuda::synchronize(device_index);
    }
    unsafe { ffi::cuptiActivityFlushAll(ffi::FLUSH_FORCED) };
    for kind in KINDS {
        unsafe { ffi::cuptiActivityDisable(kind) };
    }
    let records = std::mem::take(&mut *RECORDS.lock().unwrap_or_else(|e| e.into_inner()));
    let contexts = CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default();
    let launches: HashMap<u32, u64> = records
        .iter()
        .filter_map(|r| match *r {
            Record::External { id, correlation } => Some((correlation, id)),
            Record::Work { .. } => None,
        })
        .collect();
    let to_ns = |t: u64| (profiler_ns + t).saturating_sub(cupti_ns);
    for record in records {
        let Record::Work {
            kind,
            copy_kind,
            start,
            end,
            device,
            correlation,
        } = record
        else {
            continue;
        };
        let context = launches.get(&correlation).and_then(|id| contexts.get(id));
        if let Some(&context) = context {
            let device = Device::Cuda(device as usize);
            let name = Record::name(kind, copy_kind);
            super::record_gpu_in(context, name, device, to_ns(start), to_ns(end));
        }
    }
}
