#[cfg(lumen_cuda_linked)]
use std::ptr::NonNull;

#[cfg(lumen_cuda_linked)]
use crate::device::DeviceBackend;
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
pub(super) fn malloc(index: usize, nbytes: usize) -> NonNull<u8> {
    // c10 sets the device context before every allocation.
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
pub(super) unsafe fn free(index: usize, ptr: NonNull<u8>) {
    CUDA.set_device(index);
    unsafe { ffi::cudaFree(ptr.as_ptr().cast()) };
}
