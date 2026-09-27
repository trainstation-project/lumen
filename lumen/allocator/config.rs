//! Process-wide runtime settings. Python sets these through `lumen.config`;
//! PyTorch reads the equivalents from environment variables (e.g.
//! `PYTORCH_NO_CUDA_MEMORY_CACHING`).

use std::sync::atomic::{AtomicBool, Ordering};

static MEMORY_CACHING: AtomicBool = AtomicBool::new(true);

/// Whether device caching allocators cache freed memory (default: true).
pub fn memory_caching() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_OVERRIDE.get() {
        return enabled;
    }
    MEMORY_CACHING.load(Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with [`memory_caching`] returning `enabled` on this thread only.
/// Test-only: unit tests share one process and run in parallel, so flipping
/// the global flag would race the caching tests on other threads.
#[cfg(test)]
pub(crate) fn with_memory_caching<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    TEST_OVERRIDE.set(Some(enabled));
    let result = f();
    TEST_OVERRIDE.set(None);
    result
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
