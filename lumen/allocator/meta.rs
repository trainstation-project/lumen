use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use super::{Allocator, AllocatorBackend};

pub struct MetaAllocator;

impl Allocator for MetaAllocator {
    fn allocate(&self, _nbytes: usize) -> NonNull<u8> {
        NonNull::dangling()
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _nbytes: usize) {}
}

impl AllocatorBackend for crate::device::meta::Meta {
    fn allocator(&self, _index: usize) -> Arc<dyn Allocator> {
        static META: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
        Arc::clone(META.get_or_init(|| Arc::new(MetaAllocator)))
    }
}
