//! Allocator for the meta device, which tracks shapes and dtypes without storage.

use std::ptr::NonNull;

use super::Allocator;

/// Hands out dangling pointers; meta tensors never read or write memory.
pub struct MetaAllocator;

impl Allocator for MetaAllocator {
    /// A dangling pointer at `alignment`.
    fn allocate(&self, _nbytes: usize, alignment: usize) -> NonNull<u8> {
        NonNull::new(std::ptr::without_provenance_mut(alignment.max(1))).expect("nonzero")
    }

    /// Nothing to give back.
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _nbytes: usize, _alignment: usize) {}
}
