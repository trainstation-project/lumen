//! Process-wide runtime settings. Python sets these through `lumen.config`;
//! PyTorch reads the equivalents from environment variables (e.g.
//! `PYTORCH_NO_CUDA_MEMORY_CACHING`).

use std::sync::atomic::{AtomicBool, Ordering};

static MEMORY_CACHING: AtomicBool = AtomicBool::new(true);

/// Whether device caching allocators cache freed memory (default: true).
pub fn memory_caching() -> bool {
    MEMORY_CACHING.load(Ordering::Relaxed)
}

/// Turn device memory caching on or off (c10: `forceUncachedAllocator`).
///
/// When off, every allocation goes straight to the device backend and is
/// freed when its `DataPtr` drops — useful for debugging memory errors,
/// which the cache hides from tools like cuda-memcheck. Takes effect for
/// subsequent allocations; memory already cached stays reserved until
/// `empty_cache`. Pointers from either mode can be freed at any time, since
/// each carries its own deleter.
pub fn set_memory_caching(enabled: bool) {
    MEMORY_CACHING.store(enabled, Ordering::Relaxed);
}
