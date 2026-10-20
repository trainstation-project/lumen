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
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

#[cfg(lumen_mps_linked)]
fn malloc(nbytes: usize) -> NonNull<u8> {
    NonNull::new(unsafe { ffi::lumen_mps_alloc(nbytes) })
        .unwrap_or_else(|| panic!("Metal out of memory: failed to allocate {nbytes} bytes"))
}

#[cfg(lumen_mps_linked)]
unsafe fn free(ptr: NonNull<u8>) {
    unsafe { ffi::lumen_mps_free(ptr.as_ptr()) }
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
        self.region
            .take(nbytes, malloc, crate::stream::mps::synchronize)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
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
