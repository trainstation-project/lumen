#[cfg(lumen_mps_linked)]
use std::ptr::NonNull;
#[cfg(lumen_mps_linked)]
use std::sync::{Arc, OnceLock};

#[cfg(lumen_mps_linked)]
use super::{Allocator, Region};
#[cfg(lumen_mps_linked)]
use crate::device::Device;

#[cfg(lumen_mps_linked)]
mod ffi {
    unsafe extern "C" {
        pub fn lumen_mps_alloc(nbytes: usize) -> *mut u8;
        pub fn lumen_mps_alloc_heap(nbytes: usize) -> *mut u8;
        pub fn lumen_mps_place(ptr: *mut u8, nbytes: usize) -> i32;
        pub fn lumen_mps_unplace(ptr: *mut u8);
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

#[cfg(lumen_mps_linked)]
fn heap(nbytes: usize) -> NonNull<u8> {
    NonNull::new(unsafe { ffi::lumen_mps_alloc_heap(nbytes) })
        .unwrap_or_else(|| panic!("Metal out of memory: failed to allocate {nbytes} bytes"))
}

#[cfg(lumen_mps_linked)]
unsafe fn free(ptr: NonNull<u8>) {
    unsafe { ffi::lumen_mps_free(ptr.as_ptr()) }
}

#[cfg(lumen_mps_linked)]
pub(crate) struct Scratch(NonNull<u8>);

#[cfg(lumen_mps_linked)]
impl Scratch {
    pub(crate) fn zeroed(nbytes: usize) -> Option<Self> {
        let data = NonNull::new(unsafe { ffi::lumen_mps_alloc(nbytes) })?;
        unsafe { std::ptr::write_bytes(data.as_ptr(), 0, nbytes) };
        Some(Scratch(data))
    }

    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.0.as_ptr()
    }
}

#[cfg(lumen_mps_linked)]
impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe { free(self.0) }
    }
}

#[cfg(lumen_mps_linked)]
pub struct MpsAllocator {
    region: Region,
}

#[cfg(lumen_mps_linked)]
impl MpsAllocator {
    pub fn new(capacity: usize) -> Self {
        MpsAllocator {
            region: Region::new(Device::Mps(0), capacity),
        }
    }
}

#[cfg(lumen_mps_linked)]
impl Allocator for MpsAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        let data = self
            .region
            .take(nbytes, heap, crate::stream::mps::synchronize);
        if unsafe { ffi::lumen_mps_place(data.as_ptr(), nbytes.max(1)) } != 0 {
            panic!("Metal could not place a buffer of {nbytes} bytes in its heap");
        }
        data
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        unsafe { ffi::lumen_mps_unplace(ptr.as_ptr()) };
        self.region.give(ptr, nbytes)
    }
}

#[cfg(lumen_mps_linked)]
impl Drop for MpsAllocator {
    fn drop(&mut self) {
        if let Some(base) = self.region.base() {
            unsafe { free(base) }
        }
    }
}

#[cfg(lumen_mps_linked)]
pub(super) fn allocator() -> Arc<dyn Allocator> {
    static ALLOCATOR: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
    let capacity = super::config::static_allocator_bytes();
    Arc::clone(ALLOCATOR.get_or_init(|| Arc::new(MpsAllocator::new(capacity))))
}
