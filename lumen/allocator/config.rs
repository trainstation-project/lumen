//! Process-wide allocator settings: the size of the static region each accelerator allocator reserves,
//! the alignment of its blocks, and the size of the CUDA pinned host pool.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Byte alignment and granularity of every `Region` block.
pub const ALIGNMENT: usize = 256;

/// Bytes of page-locked host memory the CUDA pinned pool reserves on first use, for asynchronous transfers
/// between host and device; blocks that don't fit come from the regular host heap.
pub const PINNED_ALLOCATOR_BYTES: usize = 256 << 20;

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
