//! Metal allocator: one lazily reserved Metal heap carved into placed buffers, plus zeroed scratch buffers.
#[cfg(lumen_mps_linked)]
use std::ptr::NonNull;
#[cfg(lumen_mps_linked)]
use std::sync::{Arc, OnceLock};

#[cfg(lumen_mps_linked)]
use super::{Allocator, Region};
#[cfg(lumen_mps_linked)]
use crate::device::Device;

/// Bindings to the Objective-C Metal allocation shims.
#[cfg(lumen_mps_linked)]
mod ffi {
    unsafe extern "C" {
        /// Allocates a standalone shared Metal buffer of `nbytes`, returning its contents pointer or null.
        pub fn lumen_mps_alloc(nbytes: usize) -> *mut u8;
        /// Allocates a Metal heap of `nbytes`, returning its base pointer or null.
        pub fn lumen_mps_alloc_heap(nbytes: usize) -> *mut u8;
        /// Registers a buffer placed at `ptr` inside the heap; returns nonzero on failure.
        pub fn lumen_mps_place(ptr: *mut u8, nbytes: usize) -> i32;
        /// Releases the placed buffer registered at `ptr`.
        pub fn lumen_mps_unplace(ptr: *mut u8);
        /// Frees a buffer or heap returned by one of the allocation calls.
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

/// Reserves the Metal heap backing the allocator's region, panicking when Metal is out of memory.
#[cfg(lumen_mps_linked)]
fn heap(nbytes: usize) -> NonNull<u8> {
    NonNull::new(unsafe { ffi::lumen_mps_alloc_heap(nbytes) })
        .unwrap_or_else(|| panic!("Metal out of memory: failed to allocate {nbytes} bytes"))
}

/// Frees a Metal allocation.
///
/// # Safety
/// `ptr` must come from `lumen_mps_alloc` or `lumen_mps_alloc_heap` and not be freed already.
#[cfg(lumen_mps_linked)]
unsafe fn free(ptr: NonNull<u8>) {
    unsafe { ffi::lumen_mps_free(ptr.as_ptr()) }
}

/// Standalone zero-initialised Metal buffer freed on drop, outside the static region.
#[cfg(lumen_mps_linked)]
pub(crate) struct Scratch(NonNull<u8>);

#[cfg(lumen_mps_linked)]
impl Scratch {
    /// Allocates `nbytes` of zeroed Metal memory, or `None` when Metal cannot allocate it.
    pub(crate) fn zeroed(nbytes: usize) -> Option<Self> {
        let data = NonNull::new(unsafe { ffi::lumen_mps_alloc(nbytes) })?;
        unsafe { std::ptr::write_bytes(data.as_ptr(), 0, nbytes) };
        Some(Scratch(data))
    }

    /// Returns the buffer's CPU-visible base pointer.
    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.0.as_ptr()
    }
}

#[cfg(lumen_mps_linked)]
impl Drop for Scratch {
    /// Frees the buffer.
    fn drop(&mut self) {
        unsafe { free(self.0) }
    }
}

/// Allocator handing out placed buffers from a single Metal heap of fixed capacity.
#[cfg(lumen_mps_linked)]
pub struct MpsAllocator {
    region: Region,
}

#[cfg(lumen_mps_linked)]
impl MpsAllocator {
    /// Creates an allocator whose heap of `capacity` bytes is reserved on first allocation.
    pub fn new(capacity: usize) -> Self {
        MpsAllocator {
            region: Region::new(Device::Mps(0), capacity),
        }
    }
}

#[cfg(lumen_mps_linked)]
impl Allocator for MpsAllocator {
    /// Takes `nbytes` from the heap region, reserving the heap on first use, and places a Metal buffer there.
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        // every allocation is also registered as a placed buffer so kernels can bind it; zero-byte requests still need one
        let data = self
            .region
            .take(nbytes, heap, crate::stream::mps::synchronize);
        if unsafe { ffi::lumen_mps_place(data.as_ptr(), nbytes.max(1)) } != 0 {
            panic!("Metal could not place a buffer of {nbytes} bytes in its heap");
        }
        data
    }

    /// Releases the placed buffer and returns its bytes to the region.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        unsafe { ffi::lumen_mps_unplace(ptr.as_ptr()) };
        self.region.give(ptr, nbytes)
    }
}

#[cfg(lumen_mps_linked)]
impl Drop for MpsAllocator {
    /// Frees the heap if it was ever reserved.
    fn drop(&mut self) {
        if let Some(base) = self.region.base() {
            unsafe { free(base) }
        }
    }
}

/// Returns the process-wide MPS allocator sized by `static_allocator_bytes` at first use.
#[cfg(lumen_mps_linked)]
pub(super) fn allocator() -> Arc<dyn Allocator> {
    static ALLOCATOR: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
    let capacity = super::config::static_allocator_bytes();
    Arc::clone(ALLOCATOR.get_or_init(|| Arc::new(MpsAllocator::new(capacity))))
}
