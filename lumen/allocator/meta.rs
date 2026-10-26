//! Allocator for the meta device, which tracks shapes and dtypes without storage.

use std::ptr::NonNull;

use super::Allocator;

/// Hands out dangling pointers; meta tensors never read or write memory.
pub struct MetaAllocator;

impl Allocator for MetaAllocator {
    fn allocate(&self, _nbytes: usize) -> NonNull<u8> {
        NonNull::dangling()
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _nbytes: usize) {}
}
