//! CUDA support for the caching allocator.
//!
//! The pooling logic lives in [`super::caching`]; this module provides
//! c10's size math ([`CudaPolicy`]), the raw `cudaMalloc`/`cudaFree` seam
//! (`CudaBackend`) and the global per-device allocator registry,
//! mirroring c10's `CUDACachingAllocator::get()`.
//!
//! The backend is only available when built on a machine with the CUDA
//! runtime (see `build.rs`); the policy is always available, so tests
//! exercise it with their own mock backend.

use super::caching::{CachePolicy, K_MIN_LARGE_ALLOC, K_SMALL_SIZE, dedicated_segment_size};
#[cfg(lumen_cuda_linked)]
use super::caching::{CachingAllocator, DeviceBackend};
#[cfg(lumen_cuda_linked)]
use crate::device::Device;
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
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(devPtr: *mut c_void) -> i32;
    }
}

/// Backend that calls the CUDA runtime API. Only available when the build
/// found a CUDA toolkit to link against (cfg `lumen_cuda_linked`).
#[cfg(lumen_cuda_linked)]
pub struct CudaBackend {
    device_index: i32,
}

#[cfg(lumen_cuda_linked)]
impl DeviceBackend for CudaBackend {
    unsafe fn device_alloc(&self, nbytes: usize) -> *mut u8 {
        unsafe {
            // c10 sets the device context before every allocation.
            ffi::cudaSetDevice(self.device_index);
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = ffi::cudaMalloc(&mut ptr, nbytes);
            if err != 0 {
                return std::ptr::null_mut();
            }
            ptr.cast()
        }
    }

    unsafe fn device_free(&self, ptr: *mut u8) {
        unsafe {
            ffi::cudaSetDevice(self.device_index);
            ffi::cudaFree(ptr.cast());
        }
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
        .get_or_insert_with(|| {
            CachingAllocator::new(
                Device::Cuda(device_index),
                CudaBackend {
                    device_index: device_index as i32,
                },
                CudaPolicy,
            )
        })
        .clone()
}
