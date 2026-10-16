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

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(devPtr: *mut c_void) -> i32;
    }
}

#[cfg(lumen_cuda_linked)]
pub struct CudaBackend {
    device_index: i32,
}

#[cfg(lumen_cuda_linked)]
impl CudaBackend {
    pub fn new(device_index: usize) -> Self {
        CudaBackend {
            device_index: device_index as i32,
        }
    }
}

#[cfg(lumen_cuda_linked)]
impl Allocator for CudaBackend {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        crate::device::cuda::CUDA.set_device(self.device_index as usize);
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut ptr, nbytes) };

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
        crate::device::cuda::CUDA.set_device(self.device_index as usize);
        unsafe { ffi::cudaFree(ptr.as_ptr().cast()) };
    }
}

#[cfg(lumen_cuda_linked)]
pub type CudaAllocator = StaticAllocator<CudaBackend>;

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

impl super::AllocatorBackend for crate::device::cuda::Cuda {
    fn allocator(&self, index: usize) -> std::sync::Arc<dyn super::Allocator> {
        #[cfg(lumen_cuda_linked)]
        return std::sync::Arc::new(get(index));
        #[cfg(not(lumen_cuda_linked))]
        unreachable!("no CUDA device {index} is available without CUDA")
    }
}
