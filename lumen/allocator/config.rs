//! Process-wide runtime settings. Python sets these through `lumen.config`.

use std::sync::atomic::{AtomicUsize, Ordering};

/// The default static allocator size per device: 1 GiB.
pub const DEFAULT_STATIC_ALLOCATOR_BYTES: usize = 1 << 30;

static STATIC_ALLOCATOR_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_STATIC_ALLOCATOR_BYTES);

/// How many bytes each device's static allocator
/// ([`super::static_allocator::StaticAllocator`]) reserves, read when the
/// device's allocator is created, on its first allocation.
pub fn static_allocator_bytes() -> usize {
    STATIC_ALLOCATOR_BYTES.load(Ordering::Relaxed)
}

/// Set the size of static allocators created from now on; a device that
/// has allocated already keeps its allocator.
pub fn set_static_allocator_bytes(nbytes: usize) {
    STATIC_ALLOCATOR_BYTES.store(nbytes, Ordering::Relaxed);
}
