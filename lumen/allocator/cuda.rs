//! The CUDA static allocator: [`CudaBackend`] (an uncached
//! [`Allocator`](crate::Allocator) over `cudaMalloc`/`cudaFree`) reserves
//! each device's [`StaticAllocator`](super::static_allocator::StaticAllocator), and [`get`] holds one per
//! device.
//!
//! Only available when built on a machine with the CUDA runtime (see
//! `build.rs`).

#[cfg(lumen_cuda_linked)]
use super::static_allocator::StaticAllocator;
#[cfg(lumen_cuda_linked)]
use crate::allocator::Allocator;
#[cfg(lumen_cuda_linked)]
use crate::device::Device;
#[cfg(lumen_cuda_linked)]
use std::ptr::NonNull;
#[cfg(lumen_cuda_linked)]
use std::sync::{Mutex, OnceLock};

/// StaticAllocator offsets are aligned like `cudaMalloc`'s pointers.
pub const ALIGNMENT: usize = 256;

#[cfg(lumen_cuda_linked)]
mod ffi {
    use std::ffi::c_void;

    // Minimal CUDA runtime API surface, linked dynamically against cudart.
    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(devPtr: *mut c_void) -> i32;
    }
}

/// An uncached [`Allocator`] over the CUDA runtime: one `cudaMalloc` per
/// allocation, `cudaFree` in `deallocate`. Only
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
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        // c10 sets the device context before every allocation.
        crate::device::cuda::set_device(self.device_index as usize);
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut ptr, nbytes) };
        // cudaMalloc guarantees 256-byte alignment.
        NonNull::new(ptr.cast())
            .filter(|_| err == 0)
            .unwrap_or_else(|| {
                panic!(
                    "CUDA out of memory: failed to allocate {nbytes} bytes on cuda:{}",
                    self.device_index
                )
            })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _nbytes: usize) {
        crate::device::cuda::set_device(self.device_index as usize);
        unsafe { ffi::cudaFree(ptr.as_ptr().cast()) };
    }
}

/// A CUDA device's static allocator.
#[cfg(lumen_cuda_linked)]
pub type CudaAllocator = StaticAllocator<CudaBackend>;

/// Get (creating on first use, sized by
/// [`static_allocator_bytes`](super::config::static_allocator_bytes)) the static allocator for a CUDA device.
#[cfg(lumen_cuda_linked)]
pub fn get(device_index: usize) -> CudaAllocator {
    static ALLOCATORS: OnceLock<Mutex<Vec<Option<CudaAllocator>>>> = OnceLock::new();
    let registry = ALLOCATORS.get_or_init(|| Mutex::new(Vec::new()));
    let mut registry = registry.lock().unwrap_or_else(|e| e.into_inner());
    if registry.len() <= device_index {
        registry.resize_with(device_index + 1, || None);
    }
    registry[device_index]
        .get_or_insert_with(|| {
            StaticAllocator::new(
                CudaBackend::new(device_index),
                Device::Cuda(device_index),
                ALIGNMENT,
                super::config::static_allocator_bytes(),
            )
        })
        .clone()
}
