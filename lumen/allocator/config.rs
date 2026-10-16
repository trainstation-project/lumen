use std::sync::atomic::{AtomicUsize, Ordering};

pub const DEFAULT_STATIC_ALLOCATOR_BYTES: usize = 1 << 30;
static STATIC_ALLOCATOR_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_STATIC_ALLOCATOR_BYTES);

pub fn static_allocator_bytes() -> usize {
    STATIC_ALLOCATOR_BYTES.load(Ordering::Relaxed)
}

pub fn set_static_allocator_bytes(nbytes: usize) {
    STATIC_ALLOCATOR_BYTES.store(nbytes, Ordering::Relaxed);
}
