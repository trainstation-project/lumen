use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once};

use super::GpuContext;
use crate::device::Device;

mod ffi {
    use std::ffi::c_void;

    pub type BufferRequested = unsafe extern "C" fn(*mut *mut u8, *mut usize, *mut usize);
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

    pub const KIND_MEMCPY: u32 = 1;
    pub const KIND_MEMSET: u32 = 2;
    pub const KIND_RUNTIME: u32 = 4;
    pub const KIND_DRIVER: u32 = 5;
    pub const KIND_CONCURRENT_KERNEL: u32 = 10;
    pub const KIND_EXTERNAL_CORRELATION: u32 = 39;
    pub const EXTERNAL_CUSTOM0: u32 = 3;
    pub const FLUSH_FORCED: u32 = 1;
}

const KINDS: [u32; 6] = [
    ffi::KIND_CONCURRENT_KERNEL,
    ffi::KIND_MEMCPY,
    ffi::KIND_MEMSET,
    ffi::KIND_RUNTIME,
    ffi::KIND_DRIVER,
    ffi::KIND_EXTERNAL_CORRELATION,
];

enum Record {
    Work {
        name: String,
        start: u64,
        end: u64,
        device: u32,
        correlation: u32,
    },
    External {
        id: u64,
        correlation: u32,
    },
}

impl Record {
    unsafe fn read(record: *const u8) -> Option<Record> {
        let at = |offset: usize| unsafe { record.add(offset) };
        let u32_at = |offset| unsafe { at(offset).cast::<u32>().read_unaligned() };
        let u64_at = |offset| unsafe { at(offset).cast::<u64>().read_unaligned() };
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

static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());
static CONTEXTS: Mutex<Option<HashMap<u64, GpuContext>>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
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
        *max = 0;
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
    let to_ns = |t: u64| (profiler_ns + t).saturating_sub(cupti_ns);
    for record in records {
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
        if let Some(context) = context.cloned() {
            let device = Device::Cuda(device as usize);
            super::record_gpu_in(context, &name, device, to_ns(start), to_ns(end));
        }
    }
}
