//! Apple Silicon (Metal) support for the caching allocator, modeled on
//! `aten/src/ATen/mps/MPSAllocator.mm`.
//!
//! What we mirror:
//! - **Shared-mode buffers**: Apple Silicon has unified memory, so an
//!   `MTLBuffer` created with `MTLResourceStorageModeShared` exposes a real
//!   CPU pointer (`buffer.contents`). That lets the Metal backend plug into
//!   the same raw-pointer [`DeviceBackend`] seam as CUDA — no transfer API
//!   needed for host access.
//! - **Tuning**: MPS's heap sizes become our segment sizes
//!   ([`CacheConfig::mps`]): 8 MiB segments for small allocations, 32 MiB
//!   for large, with the same 1 MiB / 10 MiB / 2 MiB thresholds as CUDA.
//!
//! What we do not model (MPS-specific machinery in `MPSHeapAllocatorImpl`):
//! `MTLHeap` suballocation, `PRIVATE` (GPU-only) storage mode, the scalar
//! pool, hazard tracking for in-flight GPU work, and high/low-watermark GC.
//!
//! Metal's API is Objective-C only, so the raw calls live in a small
//! Objective-C++ shim (`csrc/mps_shim.mm`, compiled by `build.rs`) — the
//! same reason PyTorch's allocator is a `.mm` file.

#[cfg(lumen_mps_linked)]
use super::caching::{CacheConfig, CachingAllocator, DeviceBackend};
#[cfg(lumen_mps_linked)]
use crate::device::Device;
#[cfg(lumen_mps_linked)]
use std::sync::OnceLock;

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
    unsafe extern "C" {
        pub fn lumen_mps_available() -> i32;
        pub fn lumen_mps_alloc(nbytes: usize) -> *mut u8;
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

/// Backend that allocates shared-mode `MTLBuffer`s via the shim. Only
/// available on macOS builds where the shim was compiled
/// (cfg `lumen_mps_linked`).
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

/// The global MPS caching allocator type.
#[cfg(lumen_mps_linked)]
pub type MpsAllocator = CachingAllocator<MpsBackend>;

/// Get (creating on first use) the global allocator for the default Metal
/// device, mirroring `at::mps::getMPSAllocator()`.
///
/// Panics if no Metal device is present; check [`is_available`] first.
#[cfg(lumen_mps_linked)]
pub fn get() -> MpsAllocator {
    assert!(is_available(), "no Metal device available");
    static ALLOCATOR: OnceLock<MpsAllocator> = OnceLock::new();
    ALLOCATOR
        .get_or_init(|| CachingAllocator::new(Device::Mps, MpsBackend, CacheConfig::mps()))
        .clone()
}
