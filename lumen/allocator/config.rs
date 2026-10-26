//! Process-wide size of the static region each accelerator allocator reserves.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Static region size used until `set_static_allocator_bytes` changes it: 1 GiB.
pub const DEFAULT_STATIC_ALLOCATOR_BYTES: usize = 1 << 30;
static STATIC_ALLOCATOR_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_STATIC_ALLOCATOR_BYTES);

/// Bytes each device's static region reserves on its first allocation.
pub fn static_allocator_bytes() -> usize {
    STATIC_ALLOCATOR_BYTES.load(Ordering::Relaxed)
}

/// Sets the static region size; only allocators created afterwards pick it up.
pub fn set_static_allocator_bytes(nbytes: usize) {
    STATIC_ALLOCATOR_BYTES.store(nbytes, Ordering::Relaxed);
}
