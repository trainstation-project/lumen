//! `DataPtr`, the owned device buffer every tensor storage holds.

use std::ptr::NonNull;
use std::sync::Arc;

use super::Allocator;

/// Owned device buffer that returns its bytes to its allocator when dropped.
pub struct DataPtr {
    ptr: NonNull<u8>,
    nbytes: usize,
    allocator: Arc<dyn Allocator>,
}

unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

impl DataPtr {
    /// Allocates `nbytes` from `allocator`.
    pub fn allocate(allocator: Arc<dyn Allocator>, nbytes: usize) -> Self {
        DataPtr {
            ptr: allocator.allocate(nbytes),
            nbytes,
            allocator,
        }
    }

    /// Allocates `nbytes` from `allocator` starting at a multiple of `alignment`.
    pub fn allocate_aligned(
        allocator: Arc<dyn Allocator>,
        nbytes: usize,
        alignment: usize,
    ) -> Self {
        DataPtr {
            ptr: allocator.allocate_aligned(nbytes, alignment),
            nbytes,
            allocator,
        }
    }

    /// # Safety
    ///
    /// `ptr` must be `nbytes` bytes `allocator` allocated, owned by the
    /// result alone: it gives them back to `allocator` when dropped.
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

    /// Raw pointer to the buffer's first byte.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Size of the buffer in bytes.
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    /// Allocator the buffer is returned to.
    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        &self.allocator
    }
}

/// Returns the buffer's bytes to its allocator.
impl Drop for DataPtr {
    fn drop(&mut self) {
        unsafe { self.allocator.deallocate(self.ptr, self.nbytes) }
    }
}
