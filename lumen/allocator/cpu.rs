use std::alloc::{self, Layout};
use std::collections::BTreeSet;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::Allocator;
use crate::device::Device;

pub struct CpuAllocator;

static CPU_ALLOCATOR: CpuAllocator = CpuAllocator;

/// Bytes of blocks allocated while memory profiling was on and not yet
/// freed, for the profiler's memory events (PyTorch:
/// `ProfiledCPUMemoryReporter`). Other blocks never touch it.
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

/// The addresses of those blocks, whose frees are reported too, and how
/// many there are (so frees skip the lock while there are none).
static PROFILED: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());
static PROFILED_COUNT: AtomicUsize = AtomicUsize::new(0);

/// The layout of a CPU block: 64-byte aligned, to keep SIMD loads happy.
fn layout(nbytes: usize) -> Layout {
    Layout::from_size_align(nbytes, 64).expect("invalid allocation layout")
}

impl CpuAllocator {
    pub fn get() -> &'static dyn Allocator {
        &CPU_ALLOCATOR
    }
}

impl Allocator for CpuAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        let layout = layout(nbytes);
        if nbytes == 0 {
            // Layout with size 0 is fine; dangling but aligned.
            return NonNull::new(std::ptr::without_provenance_mut(layout.align())).unwrap();
        }
        // Uninitialized memory, like C's `malloc`. Reading uninitialized
        // memory is undefined behaviour in Rust.
        let raw = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        if crate::profiler::memory_enabled() {
            // Profiling memory: report this block now and when it is freed.
            // Blocks allocated before the session are not reported when
            // freed, as in PyTorch.
            let addr = ptr.as_ptr().addr();
            let total = ALLOCATED.fetch_add(nbytes, Ordering::Relaxed) + nbytes;
            crate::profiler::report_memory(Device::Cpu, addr, nbytes as i64, total, total);
            PROFILED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(addr);
            PROFILED_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

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
            crate::profiler::report_memory(Device::Cpu, addr, -(nbytes as i64), total, total);
        }
        // SAFETY: the caller passes a live block of ours, allocated with
        // this layout.
        unsafe { alloc::dealloc(ptr.as_ptr(), layout(nbytes)) }
    }
}
