//! CUDA support for the caching allocator.
//!
//! The pooling logic lives in [`super::caching`]; this module only provides
//! the raw `cudaMalloc`/`cudaFree` seam ([`CudaBackend`]) and the global
//! per-device allocator registry, mirroring c10's
//! `CUDACachingAllocator::get()`.
//!
//! The backend is only available when built on a machine with the CUDA
//! runtime (see `build.rs`); otherwise this module is empty and tests use
//! their own mock backend.

#[cfg(lumen_cuda_linked)]
use super::caching::{CacheConfig, CachingAllocator, DeviceBackend};
#[cfg(lumen_cuda_linked)]
use crate::device::Device;
#[cfg(lumen_cuda_linked)]
use std::sync::{Mutex, OnceLock};

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
pub type CudaAllocator = CachingAllocator<CudaBackend>;

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
                CacheConfig::cuda(),
            )
        })
        .clone()
}
