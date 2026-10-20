use std::alloc::{self, Layout};
use std::collections::BTreeSet;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::Allocator;
use crate::device::Device;

pub struct CpuAllocator;

static CPU_ALLOCATOR: CpuAllocator = CpuAllocator;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PROFILED: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());
static PROFILED_COUNT: AtomicUsize = AtomicUsize::new(0);

impl CpuAllocator {
    pub fn get() -> &'static dyn Allocator {
        &CPU_ALLOCATOR
    }
}

impl Allocator for CpuAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        let layout = Layout::from_size_align(nbytes, 64).expect("invalid allocation layout");
        if nbytes == 0 {
            return NonNull::new(std::ptr::without_provenance_mut(layout.align())).unwrap();
        }

        let raw = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

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
