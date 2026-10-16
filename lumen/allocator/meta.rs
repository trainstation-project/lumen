use std::ptr::NonNull;

use super::Allocator;

pub struct MetaAllocator;

impl Allocator for MetaAllocator {
    fn allocate(&self, _nbytes: usize) -> NonNull<u8> {
        NonNull::dangling()
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _nbytes: usize) {}
}
