//! Host allocator: 64-byte aligned heap memory with optional memory profiling.

use std::alloc::{self, Layout};
use std::collections::BTreeSet;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::Allocator;
use crate::device::Device;

/// Allocates host memory from the global Rust allocator.
pub struct CpuAllocator;

static CPU_ALLOCATOR: CpuAllocator = CpuAllocator;
/// Bytes currently allocated while memory profiling was on.
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
/// Addresses allocated while profiling, so frees of unprofiled blocks aren't reported.
static PROFILED: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());
/// Size of `PROFILED`, read without the lock to skip it when nothing was profiled.
static PROFILED_COUNT: AtomicUsize = AtomicUsize::new(0);

impl CpuAllocator {
    /// Shared CPU allocator instance.
    pub fn get() -> &'static dyn Allocator {
        &CPU_ALLOCATOR
    }
}

impl Allocator for CpuAllocator {
    /// Returns 64-byte aligned memory; zero-byte requests get a dangling aligned pointer.
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        let layout = Layout::from_size_align(nbytes, 64).expect("invalid allocation layout");
        if nbytes == 0 {
            return NonNull::new(std::ptr::without_provenance_mut(layout.align())).unwrap();
        }

        let raw = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        // Only blocks allocated with profiling on are tracked and later reported freed.
        if crate::profiler::memory_enabled() {
            let addr = ptr.as_ptr().addr();
            let total = ALLOCATED.fetch_add(nbytes, Ordering::Relaxed) + nbytes;
            crate::profiler::report_memory(Device::Cpu(0), addr, nbytes as i64, total, total);
            PROFILED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(addr);
            PROFILED_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    /// Frees the block, reporting it to the profiler if it was allocated while profiling.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        if nbytes == 0 {
            return;
        }

        let addr = ptr.as_ptr().addr();

        if PROFILED_COUNT.load(Ordering::Relaxed) > 0
            && PROFILED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&addr)
        {
            PROFILED_COUNT.fetch_sub(1, Ordering::Relaxed);
            let total = ALLOCATED.fetch_sub(nbytes, Ordering::Relaxed) - nbytes;
            crate::profiler::report_memory(Device::Cpu(0), addr, -(nbytes as i64), total, total);
        }

        unsafe { alloc::dealloc(ptr.as_ptr(), Layout::from_size_align_unchecked(nbytes, 64)) }
    }
}
