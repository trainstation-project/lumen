//! CUDA kernel and copy timing through CUPTI's activity API, tied back to the ops that launched them.
use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once};

use super::GpuContext;
use crate::device::Device;

/// The subset of the CUPTI C API the profiler uses.
mod ffi {
    use std::ffi::c_void;

    /// CUPTI callback asking for an activity buffer: sets its pointer, size and max record count.
    pub type BufferRequested = unsafe extern "C" fn(*mut *mut u8, *mut usize, *mut usize);
    /// CUPTI callback handing back a filled buffer with `valid` bytes of records.
    pub type BufferCompleted = unsafe extern "C" fn(*mut c_void, u32, *mut u8, usize, usize);

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

    /// `CUpti_ActivityKind` values.
    pub const KIND_MEMCPY: u32 = 1;
    pub const KIND_MEMSET: u32 = 2;
    pub const KIND_RUNTIME: u32 = 4;
    pub const KIND_DRIVER: u32 = 5;
    pub const KIND_CONCURRENT_KERNEL: u32 = 10;
    pub const KIND_EXTERNAL_CORRELATION: u32 = 39;
    /// `CUpti_ExternalCorrelationKind` used for lumen's own launch ids.
    pub const EXTERNAL_CUSTOM0: u32 = 3;
    /// `cuptiActivityFlushAll` flag to flush incomplete buffers too.
    pub const FLUSH_FORCED: u32 = 1;
}

/// Activity kinds recorded while profiling.
const KINDS: [u32; 6] = [
    ffi::KIND_CONCURRENT_KERNEL,
    ffi::KIND_MEMCPY,
    ffi::KIND_MEMSET,
    ffi::KIND_RUNTIME,
    ffi::KIND_DRIVER,
    ffi::KIND_EXTERNAL_CORRELATION,
];

/// An activity record the profiler keeps.
enum Record {
    /// A kernel, memcpy or memset with CUPTI timestamps and its CUDA correlation id.
    Work {
        name: String,
        start: u64,
        end: u64,
        device: u32,
        correlation: u32,
    },
    /// Ties CUDA correlation id `correlation` to lumen's external id `id`.
    External { id: u64, correlation: u32 },
}

impl Record {
    /// Decodes one activity record, or `None` for kinds the profiler ignores.
    ///
    /// # Safety
    ///
    /// `record` must point to a valid CUPTI activity record.
    unsafe fn read(record: *const u8) -> Option<Record> {
        let at = |offset: usize| unsafe { record.add(offset) };
        let u32_at = |offset| unsafe { at(offset).cast::<u32>().read_unaligned() };
        let u64_at = |offset| unsafe { at(offset).cast::<u64>().read_unaligned() };

        // Field offsets follow the CUPTI activity record structs.
        match u32_at(0) {
            ffi::KIND_CONCURRENT_KERNEL => {
                let name = unsafe { at(104).cast::<*const c_char>().read_unaligned() };

                Some(Record::Work {
                    name: if name.is_null() {
                        "kernel".to_owned()
                    } else {
                        unsafe { CStr::from_ptr(name) }
                            .to_string_lossy()
                            .into_owned()
                    },
                    start: u64_at(16),
                    end: u64_at(24),
                    device: u32_at(40),
                    correlation: u32_at(92),
                })
            }
            kind @ (ffi::KIND_MEMCPY | ffi::KIND_MEMSET) => Some(Record::Work {
                name: Record::name(kind, unsafe { at(4).read() }).to_owned(),
                start: u64_at(16),
                end: u64_at(24),
                device: u32_at(32),
                correlation: u32_at(44),
            }),
            ffi::KIND_EXTERNAL_CORRELATION if u32_at(4) == ffi::EXTERNAL_CUSTOM0 => {
                Some(Record::External {
                    id: u64_at(8),
                    correlation: u32_at(16),
                })
            }
            _ => None,
        }
    }

    /// Display name of a memset or memcpy record, by copy direction.
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

/// Records CUPTI has delivered since profiling started.
static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());
/// Profiler context of each lumen launch id pushed as an external correlation.
static CONTEXTS: Mutex<Option<HashMap<u64, GpuContext>>> = Mutex::new(None);
/// The step labels of each CUDA graph replay's nodes, in order, by its external correlation id.
static REPLAYS: Mutex<Option<HashMap<u64, Vec<Option<&'static str>>>>> = Mutex::new(None);
/// Next external correlation id.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// `(CUPTI timestamp, profiler timestamp)` taken together at start, to convert CUPTI times.
static CLOCK: Mutex<Option<(u64, u64)>> = Mutex::new(None);

/// Size and alignment of each activity buffer handed to CUPTI.
const BUFFER: Layout = match Layout::from_size_align(1 << 20, 8) {
    Ok(layout) => layout,
    Err(_) => panic!("CUPTI buffer layout"),
};

/// Allocates an activity buffer for CUPTI; size 0 when allocation fails.
///
/// # Safety
///
/// Called by CUPTI with valid out-pointers.
unsafe extern "C" fn buffer_requested(buffer: *mut *mut u8, size: *mut usize, max: *mut usize) {
    unsafe {
        *buffer = std::alloc::alloc(BUFFER);

        *size = if (*buffer).is_null() {
            0
        } else {
            BUFFER.size()
        };

        *max = 0;
    }
}

/// Decodes a filled activity buffer into `RECORDS` and frees it.
///
/// # Safety
///
/// Called by CUPTI with a buffer from `buffer_requested` holding `valid` bytes of records.
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

/// Enables CUPTI activity recording and pairs the CUPTI clock with the profiler's.
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

/// Runs `work`, tagging the CUDA launches it makes with a fresh id tied to the current profiler context.
pub(crate) fn correlated<R>(device: Device, work: impl FnOnce() -> R) -> R {
    tagged(device, None, work)
}

/// Runs `launch`, a CUDA graph's replay, like [`correlated`]: its kernels and copies, which share the
/// replay's correlation, are timed in order under `labels`, the steps that launched them while it captured
/// (`None`: a launch outside a step, which keeps its kernel's name).
pub(crate) fn replayed<R>(
    device: Device,
    labels: &[Option<&'static str>],
    launch: impl FnOnce() -> R,
) -> R {
    tagged(device, Some(labels), launch)
}

/// Body of [`correlated`] and [`replayed`]: `labels`, when given, name the tagged launches' records in order.
fn tagged<R>(
    device: Device,
    labels: Option<&[Option<&'static str>]>,
    work: impl FnOnce() -> R,
) -> R {
    let Some(context) = super::gpu_context(device) else {
        return work();
    };

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

    CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(id, context);

    if let Some(labels) = labels {
        REPLAYS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(HashMap::new)
            .insert(id, labels.to_vec());
    }

    unsafe { ffi::cuptiActivityPushExternalCorrelationId(ffi::EXTERNAL_CUSTOM0, id) };

    let result = work();
    let mut last = 0;

    unsafe { ffi::cuptiActivityPopExternalCorrelationId(ffi::EXTERNAL_CUSTOM0, &mut last) };

    result
}

/// Flushes and disables CUPTI, then records each kernel or copy under the context of the launch that issued it.
pub(crate) fn stop() {
    let Some((cupti_ns, profiler_ns)) = CLOCK.lock().unwrap_or_else(|e| e.into_inner()).take()
    else {
        return;
    };

    for device_index in 0..crate::device::cuda::CUDA.device_count() {
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

    // A graph replay's records, in the order they ran, take its steps' labels; unless their count
    // differs, when they keep the kernels' own names.
    let replays = REPLAYS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default();

    let mut ran: HashMap<u64, Vec<(u64, usize)>> = HashMap::new();

    for (k, record) in records.iter().enumerate() {
        if let Record::Work {
            start, correlation, ..
        } = *record
            && let Some(&id) = launches.get(&correlation)
            && replays.contains_key(&id)
        {
            ran.entry(id).or_default().push((start, k));
        }
    }

    let mut labelled: HashMap<usize, &'static str> = HashMap::new();

    for (id, mut work) in ran {
        let labels = &replays[&id];

        if work.len() != labels.len() {
            eprintln!(
                "{}: a CUDA graph replay ran {} kernels and copies, its capture {} nodes: they keep \
                 their kernels' names",
                crate::LIBRARY_NAME,
                work.len(),
                labels.len()
            );

            continue;
        }

        work.sort_unstable();
        labelled.extend(
            work.iter()
                .zip(labels)
                .filter_map(|(&(_, k), &label)| label.map(|label| (k, label))),
        );
    }

    // CUPTI timestamps shifted onto the profiler clock.
    let to_ns = |t: u64| (profiler_ns + t).saturating_sub(cupti_ns);

    for (k, record) in records.into_iter().enumerate() {
        let Record::Work {
            name,
            start,
            end,
            device,
            correlation,
        } = record
        else {
            continue;
        };

        let context = launches.get(&correlation).and_then(|id| contexts.get(id));
        let context = match labelled.get(&k) {
            Some(label) => context.map(|c| c.launching(label)),
            None => context.cloned(),
        };

        if let Some(context) = context {
            let device = Device::Cuda(device as usize);
            // Under a named launch (a plan step's), the kernel that ran keeps its own name too.
            let kernel = context.launch.is_some().then(|| name.clone());
            super::record_kernel_in(context, &name, kernel, device, to_ns(start), to_ns(end));
        }
    }
}
