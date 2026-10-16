pub mod config;
mod cpu;
pub mod cuda;
mod meta;
pub mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod static_allocator;
#[cfg(test)]
mod tests;

pub use cpu::CpuAllocator;
pub use meta::MetaAllocator;

use std::ptr::NonNull;
use std::sync::Arc;

use crate::device::Device;

pub struct DataPtr {
    ptr: NonNull<u8>,
    nbytes: usize,
    allocator: Arc<dyn Allocator>,
}

unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

impl DataPtr {
    pub fn allocate(allocator: Arc<dyn Allocator>, nbytes: usize) -> Self {
        DataPtr {
            ptr: allocator.allocate(nbytes),
            nbytes,
            allocator,
        }
    }

    pub unsafe fn from_raw_parts(
        ptr: NonNull<u8>,
        nbytes: usize,
        allocator: Arc<dyn Allocator>,
    ) -> Self {
        DataPtr {
            ptr,
            nbytes,
            allocator,
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        &self.allocator
    }
}

impl Drop for DataPtr {
    fn drop(&mut self) {
        // SAFETY: an allocation of `allocator`'s, released only here.
        unsafe { self.allocator.deallocate(self.ptr, self.nbytes) }
    }
}

pub trait Allocator: Send + Sync {
    fn allocate(&self, nbytes: usize) -> NonNull<u8>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize);
    fn reclaim(&self) {}
}

pub trait AllocatorBackend: Sync {
    fn allocator(&self, index: usize) -> Arc<dyn Allocator>;
}

pub fn allocator_for(device: Device) -> Result<Arc<dyn Allocator>, String> {
    device.check()?;
    Ok(device.backend().allocator(device.index()))
}
