#[cfg(lumen_cuda_linked)]
use std::collections::HashMap;
#[cfg(lumen_cuda_linked)]
use std::ptr::NonNull;
#[cfg(lumen_cuda_linked)]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(lumen_cuda_linked)]
use super::{Allocator, Region};
#[cfg(lumen_cuda_linked)]
use crate::device::Device;

#[cfg(lumen_cuda_linked)]
use crate::device::cuda::CUDA;

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
fn malloc(index: usize, nbytes: usize) -> NonNull<u8> {
    CUDA.set_device(index);
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { ffi::cudaMalloc(&mut ptr, nbytes) };
    NonNull::new(ptr.cast())
        .filter(|_| err == 0)
        .unwrap_or_else(|| {
            panic!("CUDA out of memory: failed to allocate {nbytes} bytes on cuda:{index}")
        })
}

/// # Safety
/// `ptr` is a `malloc` of device `index`'s, not yet freed.
#[cfg(lumen_cuda_linked)]
unsafe fn free(index: usize, ptr: NonNull<u8>) {
    CUDA.set_device(index);
    unsafe { ffi::cudaFree(ptr.as_ptr().cast()) };
}

#[cfg(lumen_cuda_linked)]
pub struct CudaAllocator {
    index: usize,
    region: Region,
}

#[cfg(lumen_cuda_linked)]
impl CudaAllocator {
    pub fn new(index: usize, capacity: usize) -> Self {
        CudaAllocator {
            index,
            region: Region::new(Device::Cuda(index), capacity),
        }
    }
}

#[cfg(lumen_cuda_linked)]
impl Allocator for CudaAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        self.region.take(nbytes, |n| malloc(self.index, n), || {})
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        self.region.give(ptr, nbytes)
    }
}

#[cfg(lumen_cuda_linked)]
impl Drop for CudaAllocator {
    fn drop(&mut self) {
        if let Some(base) = self.region.base() {
            unsafe { free(self.index, base) }
        }
    }
}

#[cfg(lumen_cuda_linked)]
pub(super) fn allocator(index: usize) -> Arc<dyn Allocator> {
    static ALLOCATORS: OnceLock<Mutex<HashMap<usize, Arc<dyn Allocator>>>> = OnceLock::new();
    let mut allocators = ALLOCATORS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let capacity = super::config::static_allocator_bytes();
    Arc::clone(
        allocators
            .entry(index)
            .or_insert_with(|| Arc::new(CudaAllocator::new(index, capacity))),
    )
}
