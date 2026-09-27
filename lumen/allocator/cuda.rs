//! CUDA support for the caching allocator.
//!
//! The pooling logic lives in [`super::caching`]; this module provides
//! c10's size math ([`CudaPolicy`]), `CudaBackend` (an uncached
//! [`Allocator`](crate::Allocator) over `cudaMalloc`/`cudaFree`) and the
//! global per-device allocator registry, mirroring c10's
//! `CUDACachingAllocator::get()`.
//!
//! The backend is only available when built on a machine with the CUDA
//! runtime (see `build.rs`); the policy is always available, so tests
//! exercise it with their own mock backend.

#[cfg(lumen_cuda_linked)]
use super::caching::CachingAllocator;
use super::traits::{CachePolicy, K_MIN_LARGE_ALLOC, K_SMALL_SIZE, dedicated_segment_size};
#[cfg(lumen_cuda_linked)]
use crate::allocator::{Allocator, DataPtr};
#[cfg(lumen_cuda_linked)]
use crate::device::Device;
#[cfg(lumen_cuda_linked)]
use std::alloc::Layout;
#[cfg(lumen_cuda_linked)]
use std::ptr::NonNull;
#[cfg(lumen_cuda_linked)]
use std::sync::{Mutex, OnceLock};

/// All sizes are rounded up to a multiple of this (c10: kMinBlockSize).
pub const K_MIN_BLOCK_SIZE: usize = 512;
/// Segment for small allocations (c10: kSmallBuffer).
pub const K_SMALL_BUFFER: usize = 2 << 20; // 2 MiB
/// Segment for 1–10 MiB allocations (c10: kLargeBuffer).
pub const K_LARGE_BUFFER: usize = 20 << 20; // 20 MiB

/// c10's `CUDACachingAllocator` size math, under the default
/// `PYTORCH_CUDA_ALLOC_CONF` (no `max_split_size_mb`,
/// `roundup_power2_divisions`, or `expandable_segments`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaPolicy;

impl CachePolicy for CudaPolicy {
    fn alignment(&self) -> usize {
        K_MIN_BLOCK_SIZE
    }

    /// c10: `round_size`.
    fn round_size(&self, nbytes: usize) -> usize {
        nbytes.div_ceil(K_MIN_BLOCK_SIZE) * K_MIN_BLOCK_SIZE
    }

    /// c10: `should_split`.
    fn should_split(&self, small: bool, remaining: usize) -> bool {
        if small {
            remaining >= K_MIN_BLOCK_SIZE
        } else {
            remaining > K_SMALL_SIZE
        }
    }

    /// c10: `get_allocation_size`.
    fn segment_size(&self, size: usize, _reserved: usize) -> usize {
        if size <= K_SMALL_SIZE {
            K_SMALL_BUFFER
        } else if size < K_MIN_LARGE_ALLOC {
            K_LARGE_BUFFER
        } else {
            dedicated_segment_size(size)
        }
    }
}

#[cfg(lumen_cuda_linked)]
mod ffi {
    use std::ffi::c_void;

    // Minimal CUDA runtime API surface, linked dynamically against cudart.
    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaSetDevice(device: i32) -> i32;
        pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(devPtr: *mut c_void) -> i32;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        pub fn cudaMemset(dst: *mut c_void, value: i32, count: usize) -> i32;
        pub fn cudaEventCreate(event: *mut *mut c_void) -> i32;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(event: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
        pub fn cudaEventDestroy(event: *mut c_void) -> i32;
    }

    // `cudaMemcpyKind` values.
    pub const HOST_TO_DEVICE: i32 = 1;
    pub const DEVICE_TO_HOST: i32 = 2;
}

/// Number of visible CUDA devices (`torch.cuda.device_count()`); 0 when
/// lumen was built without CUDA or the driver reports an error.
pub fn device_count() -> usize {
    #[cfg(lumen_cuda_linked)]
    {
        let mut count = 0;
        match unsafe { ffi::cudaGetDeviceCount(&mut count) } {
            0 => count.max(0) as usize,
            _ => 0,
        }
    }
    #[cfg(not(lumen_cuda_linked))]
    {
        0
    }
}

/// Whether any CUDA device is usable (`torch.cuda.is_available()`).
pub fn is_available() -> bool {
    device_count() > 0
}

/// An uncached [`Allocator`] over the CUDA runtime: one `cudaMalloc` per
/// `try_allocate`, `cudaFree` when the returned `DataPtr` drops. Only
/// available when the build found a CUDA toolkit to link against
/// (cfg `lumen_cuda_linked`).
#[cfg(lumen_cuda_linked)]
pub struct CudaBackend {
    device_index: i32,
}

#[cfg(lumen_cuda_linked)]
impl CudaBackend {
    /// The backend for CUDA device `device_index`.
    pub fn new(device_index: usize) -> Self {
        CudaBackend {
            device_index: device_index as i32,
        }
    }
}

#[cfg(lumen_cuda_linked)]
impl Allocator for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda(self.device_index as usize)
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        self.try_allocate(nbytes).unwrap_or_else(|| {
            panic!(
                "CUDA out of memory: failed to allocate {nbytes} bytes on cuda:{}",
                self.device_index
            )
        })
    }

    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        // c10 sets the device context before every allocation.
        unsafe { ffi::cudaSetDevice(self.device_index) };
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut ptr, nbytes) };
        if err != 0 {
            return None;
        }
        let device_index = self.device_index;
        Some(DataPtr::with_deleter(
            NonNull::new(ptr.cast())?,
            // cudaMalloc guarantees 256-byte alignment.
            Layout::from_size_align(nbytes, 256).unwrap(),
            move |p| unsafe {
                ffi::cudaSetDevice(device_index);
                ffi::cudaFree(p.as_ptr().cast());
            },
        ))
    }

    // Device memory is not host-addressable: go through cudaMemcpy (which
    // synchronizes with the device, like PyTorch's blocking `copy_`).
    unsafe fn copy_from_host(&self, dst: *mut u8, src: *const u8, nbytes: usize) {
        self.memcpy(dst, src, nbytes, ffi::HOST_TO_DEVICE);
    }

    unsafe fn copy_to_host(&self, dst: *mut u8, src: *const u8, nbytes: usize) {
        self.memcpy(dst, src, nbytes, ffi::DEVICE_TO_HOST);
    }

    unsafe fn memset(&self, dst: *mut u8, value: u8, nbytes: usize) {
        if nbytes == 0 {
            return;
        }

        let mut err = 0;
        self.timed("Memset", || {
            err = unsafe {
                ffi::cudaSetDevice(self.device_index);
                ffi::cudaMemset(dst.cast(), value.into(), nbytes)
            };
        });

        assert_eq!(
            err, 0,
            "cudaMemset of {nbytes} bytes on cuda:{} failed (error {err})",
            self.device_index
        );
    }
}

#[cfg(lumen_cuda_linked)]
impl CudaBackend {
    fn memcpy(&self, dst: *mut u8, src: *const u8, nbytes: usize, kind: i32) {
        if nbytes == 0 {
            return;
        }

        let name = if kind == ffi::HOST_TO_DEVICE {
            "Memcpy HtoD"
        } else {
            "Memcpy DtoH"
        };
        let mut err = 0;
        self.timed(name, || {
            err = unsafe {
                ffi::cudaSetDevice(self.device_index);
                ffi::cudaMemcpy(dst.cast(), src.cast(), nbytes, kind)
            };
        });

        assert_eq!(
            err, 0,
            "cudaMemcpy of {nbytes} bytes on cuda:{} failed (error {err})",
            self.device_index
        );
    }
}

// ---------------- profiler timing ----------------

/// A CUDA event, kept as an address so it can live in a `static`.
#[cfg(lumen_cuda_linked)]
#[derive(Clone, Copy)]
struct CudaEvent(usize);

#[cfg(lumen_cuda_linked)]
impl CudaEvent {
    /// A new event on the current device.
    fn new() -> Option<Self> {
        let mut event = std::ptr::null_mut();
        (unsafe { ffi::cudaEventCreate(&mut event) } == 0).then(|| CudaEvent(event.addr()))
    }

    fn raw(self) -> *mut std::ffi::c_void {
        std::ptr::without_provenance_mut(self.0)
    }

    /// Record on the legacy default stream, which `cudaMemcpy` and
    /// `cudaMemset` run on, so the event orders with them.
    fn record(self) -> bool {
        unsafe { ffi::cudaEventRecord(self.raw(), std::ptr::null_mut()) == 0 }
    }

    fn synchronize(self) -> bool {
        unsafe { ffi::cudaEventSynchronize(self.raw()) == 0 }
    }

    /// Nanoseconds from `self` to `later`, both completed.
    fn ns_until(self, later: CudaEvent) -> Option<u64> {
        let mut ms = 0.0f32;
        let ok = unsafe { ffi::cudaEventElapsedTime(&mut ms, self.raw(), later.raw()) } == 0;
        ok.then(|| (f64::from(ms) * 1e6).max(0.0) as u64)
    }

    fn destroy(self) {
        unsafe { ffi::cudaEventDestroy(self.raw()) };
    }
}

#[cfg(lumen_cuda_linked)]
impl CudaBackend {
    /// Run `work` (device work on the default stream). When the profiler
    /// times CUDA, bracket it with events and record it as `name` (PyTorch's
    /// legacy `use_cuda` profiler timed with CUDA events the same way).
    fn timed(&self, name: &'static str, work: impl FnOnce()) {
        let device = Device::Cuda(self.device_index as usize);
        if !crate::profiler::device_enabled(device) {
            return work();
        }
        unsafe { ffi::cudaSetDevice(self.device_index) };
        let events = self
            .reference_event()
            .zip(CudaEvent::new())
            .zip(CudaEvent::new());
        let Some(((reference, start), stop)) = events else {
            return work();
        };
        start.record();
        work();
        let finished = stop.record() && stop.synchronize();
        let (reference_event, reference_ns) = reference;
        let times = reference_event
            .ns_until(start)
            .zip(reference_event.ns_until(stop))
            .filter(|_| finished);
        start.destroy();
        stop.destroy();
        if let Some((from_reference_to_start, from_reference_to_stop)) = times {
            crate::profiler::record_gpu(
                name,
                device,
                reference_ns + from_reference_to_start,
                reference_ns + from_reference_to_stop,
            );
        }
    }

    /// An event recorded once per profiler session on this device, with the
    /// profiler time it completed at: GPU times are measured from it, which
    /// puts them on the profiler clock.
    fn reference_event(&self) -> Option<(CudaEvent, u64)> {
        use std::collections::HashMap;
        /// Device index -> (session, event, profiler ns).
        type References = HashMap<i32, (u64, CudaEvent, u64)>;
        static REFERENCES: Mutex<Option<References>> = Mutex::new(None);
        let session = crate::profiler::session_id();
        let mut references = REFERENCES.lock().unwrap_or_else(|e| e.into_inner());
        let references = references.get_or_insert_with(HashMap::new);
        if let Some(&(s, event, ns)) = references.get(&self.device_index) {
            if s == session {
                return Some((event, ns));
            }
            event.destroy(); // from an earlier session
            references.remove(&self.device_index);
        }
        let event = CudaEvent::new()?;
        if !(event.record() && event.synchronize()) {
            event.destroy();
            return None;
        }
        let ns = crate::profiler::now_ns();
        references.insert(self.device_index, (session, event, ns));
        Some((event, ns))
    }
}

/// The global CUDA caching allocator type.
#[cfg(lumen_cuda_linked)]
pub type CudaAllocator = CachingAllocator<CudaBackend, CudaPolicy>;

/// Get (creating on first use) the global allocator for a CUDA device,
/// mirroring `c10::cuda::CUDACachingAllocator::get()`.
#[cfg(lumen_cuda_linked)]
pub fn get(device_index: usize) -> CudaAllocator {
    static ALLOCATORS: OnceLock<Mutex<Vec<Option<CudaAllocator>>>> = OnceLock::new();
    let registry = ALLOCATORS.get_or_init(|| Mutex::new(Vec::new()));
    let mut registry = registry.lock().unwrap();
    if registry.len() <= device_index {
        registry.resize_with(device_index + 1, || None);
    }
    registry[device_index]
        .get_or_insert_with(|| CachingAllocator::new(CudaBackend::new(device_index), CudaPolicy))
        .clone()
}
