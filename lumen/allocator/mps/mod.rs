#[cfg(lumen_mps_linked)]
use std::ptr::NonNull;

#[cfg(lumen_mps_linked)]
mod ffi {
    unsafe extern "C" {
        pub fn lumen_mps_alloc(nbytes: usize) -> *mut u8;
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

#[cfg(lumen_mps_linked)]
pub(super) fn malloc(nbytes: usize) -> NonNull<u8> {
    NonNull::new(unsafe { ffi::lumen_mps_alloc(nbytes) })
        .unwrap_or_else(|| panic!("Metal out of memory: failed to allocate {nbytes} bytes"))
}

/// SAFETY: `ptr` is a `malloc` of ours, not yet freed.
#[cfg(lumen_mps_linked)]
pub(super) unsafe fn free(ptr: NonNull<u8>) {
    unsafe { ffi::lumen_mps_free(ptr.as_ptr()) }
}
