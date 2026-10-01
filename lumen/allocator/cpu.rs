use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Allocator, DataPtr};
use crate::device::Device;

pub struct CpuAllocator;

static CPU_ALLOCATOR: CpuAllocator = CpuAllocator;

/// Bytes of blocks allocated while memory profiling was on and not yet
/// freed, for the profiler's memory events (PyTorch:
/// `ProfiledCPUMemoryReporter`). Other blocks never touch it.
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

impl CpuAllocator {
    pub fn get() -> &'static dyn Allocator {
        &CPU_ALLOCATOR
    }
}

impl Allocator for CpuAllocator {
    fn allocate(&self, nbytes: usize) -> DataPtr {
        // 64-byte alignment to keep SIMD loads happy.
        let layout = Layout::from_size_align(nbytes, 64).expect("invalid allocation layout");

        let ptr = if nbytes == 0 {
            // Layout with size 0 is fine; dangling but aligned.
            NonNull::new(std::ptr::without_provenance_mut(layout.align())).unwrap()
        } else {
            // Uninitialized memory, like C's `malloc`. Reading
            // uninitialized memory is undefined behaviour in Rust.
            let raw = unsafe { alloc::alloc(layout) };
            NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout))
        };

        if nbytes > 0 && crate::profiler::memory_enabled() {
            // Profiling memory: report this block now and when it is freed.
            // Blocks allocated before the session are not reported when
            // freed, as in PyTorch.
            let addr = ptr.as_ptr().addr();
            let total = ALLOCATED.fetch_add(nbytes, Ordering::Relaxed) + nbytes;
            crate::profiler::report_memory(Device::Cpu, addr, nbytes as i64, total, total);
            return DataPtr::with_deleter(ptr, layout, move |p| {
                let total = ALLOCATED.fetch_sub(nbytes, Ordering::Relaxed) - nbytes;
                crate::profiler::report_memory(Device::Cpu, addr, -(nbytes as i64), total, total);
                unsafe { alloc::dealloc(p.as_ptr(), layout) }
            });
        }

        DataPtr::new(ptr, layout)
    }
}
