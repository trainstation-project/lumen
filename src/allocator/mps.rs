use super::caching::CachingAllocator;
use super::traits::{CachePolicy, K_MIN_LARGE_ALLOC, K_SMALL_SIZE, dedicated_segment_size};
#[cfg(lumen_mps_linked)]
use crate::allocator::{Allocator, DataPtr};
#[cfg(lumen_mps_linked)]
use crate::device::Device;
#[cfg(lumen_mps_linked)]
use std::alloc::Layout;
#[cfg(lumen_mps_linked)]
use std::ptr::NonNull;
#[cfg(lumen_mps_linked)]
use std::sync::OnceLock;

/// Heap for small allocations (kSmallHeap).
pub const K_SMALL_HEAP: usize = 8 << 20; // 8 MiB
/// Heap for 1–10 MiB allocations (kLargeHeap).
pub const K_LARGE_HEAP: usize = 32 << 20; // 32 MiB
/// Heap for 10 MiB – 512 MiB allocations absent memory pressure (kXLargeHeap).
pub const K_XLARGE_HEAP: usize = 1 << 30; // 1 GiB

/// aten's `MPSHeapAllocatorImpl` size math for its shared-storage pools,
/// parameterized by the device properties it reads from Metal. Build one
/// for the real device with [`MpsPolicy::from_device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MpsPolicy {
    /// Placement alignment of heap buffers (`heapBufferSizeAndAlignWithLength`'s
    /// `align`; MPS: `BufferPool::alignment`).
    pub alignment: usize,
    /// VM page size: the smallest granule large requests are bucketed to.
    pub page_size: usize,
    /// `MTLDevice.maxBufferLength`; requests must be smaller.
    pub max_buffer_size: usize,
    /// Reserved bytes at which large allocations count as under memory
    /// pressure (MPS: `m_low_watermark_limit`). `usize::MAX` disables it.
    pub low_watermark_limit: usize,
}

/// Heap size class (MPS: `HeapTier` / `getHeapTier`).
#[derive(Debug, PartialEq, Eq)]
enum HeapTier {
    Small,
    Large,
    XLarge,
    Oversize,
}

fn heap_tier(size: usize, memory_pressure: bool) -> HeapTier {
    if size <= K_SMALL_SIZE {
        HeapTier::Small
    } else if size < K_MIN_LARGE_ALLOC {
        HeapTier::Large
    } else if size < K_XLARGE_HEAP / 2 && !memory_pressure {
        HeapTier::XLarge
    } else {
        HeapTier::Oversize
    }
}

impl CachePolicy for MpsPolicy {
    fn alignment(&self) -> usize {
        self.alignment
    }

    /// MPS: `get_allocation_size`.
    fn round_size(&self, nbytes: usize) -> usize {
        let aligned = nbytes.next_multiple_of(self.alignment);
        if aligned <= K_SMALL_SIZE {
            return aligned;
        }
        // Large requests round up into 32 buckets per power of two (at least
        // a page), so a slowly growing allocation reuses its predecessor's
        // freed block.
        let granule = ((1usize << aligned.ilog2()) >> 5).max(self.page_size);
        let bucketed = aligned.next_multiple_of(granule);
        // Never round into a larger heap class or past Metal's limit.
        if bucketed >= self.max_buffer_size
            || heap_tier(bucketed, false) != heap_tier(aligned, false)
        {
            aligned
        } else {
            bucketed
        }
    }

    /// MPS: `remainder_size >= pool.min_split`.
    fn should_split(&self, small: bool, remaining: usize) -> bool {
        remaining >= if small { self.alignment } else { K_SMALL_SIZE }
    }

    /// MPS: `HeapBlock::createHeapBlock`.
    fn segment_size(&self, size: usize, reserved: usize) -> usize {
        // Large requests are under memory pressure once reserved memory is
        // within 1 MiB of the low watermark (`getLowWatermarkValue() <= 0`).
        let pressure =
            size > K_SMALL_SIZE && reserved.saturating_add(K_SMALL_SIZE) > self.low_watermark_limit;
        match heap_tier(size, pressure) {
            HeapTier::Small => K_SMALL_HEAP,
            HeapTier::Large => K_LARGE_HEAP,
            HeapTier::XLarge => K_XLARGE_HEAP,
            HeapTier::Oversize => dedicated_segment_size(size),
        }
    }

    /// MPS: `TORCH_CHECK(size < m_max_buffer_size, "Invalid buffer size")`.
    fn check_request(&self, nbytes: usize) {
        assert!(
            nbytes < self.max_buffer_size,
            "Invalid buffer size: {nbytes} bytes"
        );
    }
}

/// True when lumen was built with Metal support (macOS + `mps` feature) and
/// a default Metal device exists. Safe to call on any build.
pub fn is_available() -> bool {
    #[cfg(lumen_mps_linked)]
    {
        unsafe { ffi::lumen_mps_available() != 0 }
    }
    #[cfg(not(lumen_mps_linked))]
    {
        false
    }
}

#[cfg(lumen_mps_linked)]
mod ffi {
    // C ABI exported by csrc/mps_shim.mm.
    #[repr(C)]
    #[derive(Default)]
    pub struct Limits {
        pub alignment: usize,
        pub page_size: usize,
        pub max_buffer_length: usize,
        pub recommended_max_working_set: usize,
    }

    unsafe extern "C" {
        pub fn lumen_mps_available() -> i32;
        pub fn lumen_mps_limits(out: *mut Limits) -> i32;
        pub fn lumen_mps_alloc(nbytes: usize) -> *mut u8;
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

/// An uncached [`Allocator`] over shared-mode `MTLBuffer`s (via the
/// Objective-C++ shim): one `newBufferWithLength:` per `try_allocate`,
/// released when the returned `DataPtr` drops. Only available on macOS
/// builds where the shim was compiled (cfg `lumen_mps_linked`).
#[cfg(lumen_mps_linked)]
pub struct MpsBackend;

#[cfg(lumen_mps_linked)]
impl DeviceBackend for MpsBackend {
    unsafe fn device_alloc(&self, nbytes: usize) -> *mut u8 {
        unsafe { ffi::lumen_mps_alloc(nbytes) }
    }

    unsafe fn device_free(&self, ptr: *mut u8) {
        unsafe { ffi::lumen_mps_free(ptr) }
    }
}

/// MPS defaults for the watermark ratios (`default_low_watermark_ratio`,
/// `default_high_watermark_ratio`, `default_high_watermark_upper_bound`).
#[cfg(lumen_mps_linked)]
const DEFAULT_LOW_WATERMARK_RATIO: f64 = 1.4;
#[cfg(lumen_mps_linked)]
const DEFAULT_HIGH_WATERMARK_RATIO: f64 = 1.7;
#[cfg(lumen_mps_linked)]
const HIGH_WATERMARK_UPPER_BOUND: f64 = 2.0;

/// A watermark ratio from the environment, falling back to `default` (MPS
/// parses these with `strtod`, which reads garbage as 0).
#[cfg(lumen_mps_linked)]
fn watermark_ratio(var: &str, default: f64) -> f64 {
    std::env::var(var).map_or(default, |v| v.trim().parse().unwrap_or(0.0))
}

impl MpsPolicy {
    /// The policy for the default Metal device.
    ///
    /// The low watermark follows `MPSHeapAllocatorImpl::setLowWatermarkRatio`:
    /// `PYTORCH_MPS_LOW_WATERMARK_RATIO` (default 1.4) times
    /// `recommendedMaxWorkingSetSize`, where 0 disables it. Like PyTorch, the
    /// ratio must not exceed the high-watermark ratio
    /// (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`, default 1.7; 2.0 when that is 0).
    ///
    /// Panics if no Metal device is present or a ratio is out of range.
    #[cfg(lumen_mps_linked)]
    pub fn from_device() -> Self {
        let mut raw = ffi::Limits::default();
        assert!(
            unsafe { ffi::lumen_mps_limits(&mut raw) } != 0,
            "no Metal device available"
        );

        let high = watermark_ratio(
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            DEFAULT_HIGH_WATERMARK_RATIO,
        );
        assert!(
            (0.0..=HIGH_WATERMARK_UPPER_BOUND).contains(&high),
            "invalid high watermark ratio {high}"
        );
        let low = watermark_ratio(
            "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            DEFAULT_LOW_WATERMARK_RATIO,
        );
        let high_limit = if high == 0.0 {
            HIGH_WATERMARK_UPPER_BOUND
        } else {
            high
        };
        assert!(
            (0.0..=high_limit).contains(&low),
            "invalid low watermark ratio {low}"
        );

        MpsPolicy {
            alignment: raw.alignment,
            page_size: raw.page_size,
            max_buffer_size: raw.max_buffer_length,
            low_watermark_limit: if low == 0.0 {
                usize::MAX
            } else {
                (low * raw.recommended_max_working_set as f64) as usize
            },
        }
    }
}

/// The global MPS caching allocator type.
#[cfg(lumen_mps_linked)]
pub type MpsAllocator = CachingAllocator<MpsBackend, MpsPolicy>;

/// Get (creating on first use) the global allocator for the default Metal
/// device, mirroring `at::mps::getMPSAllocator()`.
///
/// Panics if no Metal device is present; check [`is_available`] first.
#[cfg(lumen_mps_linked)]
pub fn get() -> MpsAllocator {
    assert!(is_available(), "no Metal device available");
    static ALLOCATOR: OnceLock<MpsAllocator> = OnceLock::new();
    ALLOCATOR
        .get_or_init(|| CachingAllocator::new(Device::Mps, MpsBackend, MpsPolicy::from_device()))
        .clone()
}
