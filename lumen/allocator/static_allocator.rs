use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use super::Allocator;
use crate::device::Device;

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

struct Inner<B: Allocator> {
    backend: B,
    device: Device,
    alignment: usize,
    capacity: usize,
    state: Mutex<State>,
}

struct State {
    /// The region, reserved on the first allocation.
    region: Option<Region>,
    /// Free blocks, offset to length; offsets and lengths are multiples of
    /// the alignment, and no two blocks touch (they are merged).
    free: BTreeMap<usize, usize>,
    /// Allocations not yet freed, offset to requested bytes, and the sum of
    /// their bytes.
    live: BTreeMap<usize, usize>,
    live_bytes: usize,
}

/// The start of the region, in the backend's address space.
struct Region(NonNull<u8>);

// SAFETY: the region is device memory owned by the allocator; the pointer
// is only offset and handed out, never dereferenced here.
unsafe impl Send for Region {}

impl<B: Allocator> StaticAllocator<B> {
    pub fn new(backend: B, device: Device, alignment: usize, capacity: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "alignment must be a power of two"
        );
        StaticAllocator {
            inner: Arc::new(Inner {
                device,
                backend,
                alignment,
                capacity,
                state: Mutex::new(State {
                    region: None,
                    free: BTreeMap::from([(0, capacity - capacity % alignment)]),
                    live: BTreeMap::new(),
                    live_bytes: 0,
                }),
            }),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn used(&self) -> usize {
        self.inner
            .lock()
            .live
            .last_key_value()
            .map_or(0, |(offset, nbytes)| offset + nbytes)
    }

    pub fn live_bytes(&self) -> usize {
        self.inner.lock().live_bytes
    }
}

impl<B: Allocator> Inner<B> {
    fn block(&self, nbytes: usize) -> usize {
        nbytes.next_multiple_of(self.alignment)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Return the allocation of `nbytes` at `offset` (address `addr`),
    /// merging its block with free neighbours.
    fn free(&self, addr: usize, offset: usize, nbytes: usize) {
        let mut state = self.lock();
        if nbytes == 0 {
            return;
        }
        state.live.remove(&offset);
        state.live_bytes -= nbytes;
        let (mut start, mut len) = (offset, self.block(nbytes));
        if let Some((&prev, &prev_len)) = state.free.range(..start).next_back()
            && prev + prev_len == start
        {
            state.free.remove(&prev);
            (start, len) = (prev, prev_len + len);
        }
        if let Some(next_len) = state.free.remove(&(start + len)) {
            len += next_len;
        }
        state.free.insert(start, len);
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

impl<B: Allocator + 'static> StaticAllocator<B> {
    /// An allocation of `nbytes`, or `None` when the region has no room
    /// left. Reserves the region on first use (panicking if the backend
    /// cannot).
    pub(crate) fn try_allocate(&self, nbytes: usize) -> Option<NonNull<u8>> {
        let inner = &self.inner;
        let mut state = inner.lock();
        if state.region.is_none() {
            state.region = Some(Region(inner.backend.allocate(inner.capacity)));
        }
        let block = inner.block(nbytes);
        // An empty allocation takes no block: any aligned offset will do.
        let (&offset, &len) = match nbytes {
            0 => state.free.iter().next().unwrap_or((&0, &0)),
            _ => state.free.iter().find(|&(_, &len)| len >= block)?,
        };
        let base = state.region.as_ref().expect("reserved above").0.as_ptr();
        // In the device's address space: the backend's memory is not a Rust
        // allocation, so no `add`.
        let ptr = NonNull::new(base.wrapping_add(offset))?;
        if nbytes > 0 {
            state.free.remove(&offset);
            if len > block {
                state.free.insert(offset + block, len - block);
            }
            state.live.insert(offset, nbytes);
            state.live_bytes += nbytes;
        }
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
        Some(ptr)
    }
}

impl<B: Allocator + 'static> Allocator for StaticAllocator<B> {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        // Out of room: memory freed but held by device work in flight comes
        // back once that work is done.
        let data = self.try_allocate(nbytes).or_else(|| {
            self.inner.backend.reclaim();
            self.try_allocate(nbytes)
        });
        data.unwrap_or_else(|| {
            let state = self.inner.lock();
            panic!(
                "{} static allocator out of memory: {nbytes} bytes requested, {} of {} bytes free \
                 (largest block {}); raise {}.config.static_allocator_bytes",
                self.inner.device,
                state.free.values().sum::<usize>(),
                self.inner.capacity,
                state.free.values().max().unwrap_or(&0),
                crate::LIBRARY_NAME,
            )
        })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        let base = match &self.inner.lock().region {
            Some(region) => region.0.as_ptr().addr(),
            None => unreachable!("an allocation reserves the region first"),
        };
        let addr = ptr.as_ptr().addr();
        self.inner.free(addr, addr - base, nbytes);
    }
}

impl<B: Allocator> Drop for Inner<B> {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(region) = state.region.take() {
            // SAFETY: the backend allocated the region with this size, and
            // nothing points into it any more.
            unsafe { self.backend.deallocate(region.0, self.capacity) };
        }
    }
}
