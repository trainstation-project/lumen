//! The static allocator: one region per device, reserved once from its
//! backend (one `cudaMalloc`, one `MTLBuffer`), handing out aligned offsets
//! from a bump pointer. An allocation is pointer arithmetic, with no driver
//! call, which is what compiled execution needs: its buffers are laid out
//! up front and reused run after run.
//!
//! Freeing one allocation gives nothing back by itself; when the last live
//! allocation is freed, the bump pointer rewinds to the start, so each run
//! starts from an empty region. Nothing can point into the region then, so
//! the rewind is always safe.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use super::{Allocator, DataPtr};
use crate::device::Device;

/// A bump allocator over one region of `backend`'s memory. Cheap to clone:
/// clones share the region.
pub struct StaticAllocator<B: Allocator> {
    inner: Arc<Inner<B>>,
}

impl<B: Allocator> Clone for StaticAllocator<B> {
    fn clone(&self) -> Self {
        StaticAllocator {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct Inner<B> {
    backend: B,
    device: Device,
    alignment: usize,
    capacity: usize,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// The region, reserved on the first allocation.
    region: Option<DataPtr>,
    /// The bump pointer: bytes from the region's start handed out so far.
    top: usize,
    /// Allocations not yet freed, and their bytes.
    live: usize,
    live_bytes: usize,
}

impl<B: Allocator> StaticAllocator<B> {
    /// A static allocator of `capacity` bytes from `backend`, handing out
    /// offsets aligned to `alignment` (a power of two). Nothing is reserved
    /// until the first allocation.
    pub fn new(backend: B, alignment: usize, capacity: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "alignment must be a power of two"
        );
        StaticAllocator {
            inner: Arc::new(Inner {
                device: backend.device(),
                backend,
                alignment,
                capacity,
                state: Mutex::new(State::default()),
            }),
        }
    }

    /// The region's size in bytes.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Bytes handed out since the allocator was last empty (the bump
    /// pointer).
    pub fn used(&self) -> usize {
        self.inner.lock().top
    }

    /// Bytes of allocations not yet freed.
    pub fn live_bytes(&self) -> usize {
        self.inner.lock().live_bytes
    }
}

impl<B> Inner<B> {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Return an allocation of `nbytes` at `addr`, rewinding the region if it
    /// was the last one.
    fn free(&self, addr: usize, nbytes: usize) {
        let mut state = self.lock();
        state.live -= 1;
        state.live_bytes -= nbytes;
        if state.live == 0 {
            state.top = 0;
        }
        let live_bytes = state.live_bytes;
        drop(state);
        crate::profiler::report_memory(
            self.device,
            addr,
            -(nbytes as i64),
            live_bytes,
            self.capacity,
        );
    }
}

impl<B: Allocator + 'static> Allocator for StaticAllocator<B> {
    fn device(&self) -> Device {
        self.inner.device
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        self.try_allocate(nbytes).unwrap_or_else(|| {
            let state = self.inner.lock();
            panic!(
                "{} static allocator out of memory: {nbytes} bytes requested, {} of {} bytes free; raise \
                 {}.config.static_allocator_bytes",
                self.inner.device,
                self.inner.capacity - state.top.min(self.inner.capacity),
                self.inner.capacity,
                crate::LIBRARY_NAME,
            )
        })
    }

    /// `None` when the region cannot be reserved or has no room left.
    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        let inner = &self.inner;
        let mut state = inner.lock();
        if state.region.is_none() {
            state.region = Some(inner.backend.try_allocate(inner.capacity)?);
        }
        let offset = state.top.next_multiple_of(inner.alignment);
        let end = offset.checked_add(nbytes)?;
        if end > inner.capacity {
            return None;
        }
        let base = state.region.as_ref().expect("reserved above").as_ptr();
        // In the device's address space: the backend's memory is not a Rust
        // allocation, so no `add`.
        let ptr = NonNull::new(base.wrapping_add(offset))?;
        state.top = end;
        state.live += 1;
        state.live_bytes += nbytes;
        let live_bytes = state.live_bytes;
        drop(state);

        let addr = ptr.as_ptr().addr();
        crate::profiler::report_memory(
            inner.device,
            addr,
            nbytes as i64,
            live_bytes,
            inner.capacity,
        );
        let owner = Arc::clone(inner);
        Some(DataPtr::with_deleter(
            ptr,
            Layout::from_size_align(nbytes, inner.alignment).expect("valid layout"),
            move |_| owner.free(addr, nbytes),
        ))
    }
}
