//! The static allocator: one region per device, reserved once from its
//! backend (one `cudaMalloc`, one `MTLBuffer`), handing out aligned offsets
//! into it. An allocation is bookkeeping on the host, with no driver call,
//! which is what compiled execution needs: its buffers are laid out up
//! front and reused run after run.
//!
//! Free bytes are kept as a list of blocks, by offset. An allocation takes
//! the lowest block that fits (first fit) and splits off the rest; freeing
//! returns its block and merges it with free neighbours, so memory comes
//! back whatever order allocations are freed in: a loop that keeps its
//! inputs alive reuses the same bytes every iteration.

use std::alloc::Layout;
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use super::{Allocator, DataPtr};
use crate::device::Device;

/// A first-fit allocator over one region of `backend`'s memory. Cheap to
/// clone: clones share the region.
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

struct State {
    /// The region, reserved on the first allocation.
    region: Option<DataPtr>,
    /// Free blocks, offset to length; offsets and lengths are multiples of
    /// the alignment, and no two blocks touch (they are merged).
    free: BTreeMap<usize, usize>,
    /// Allocations not yet freed, offset to requested bytes, and the sum of
    /// their bytes.
    live: BTreeMap<usize, usize>,
    live_bytes: usize,
}

impl<B: Allocator> StaticAllocator<B> {
    /// A static allocator of `capacity` bytes from `backend`, the memory of
    /// `device`, handing out offsets aligned to `alignment` (a power of
    /// two). Nothing is reserved until the first allocation.
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

    /// The region's size in bytes.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Bytes from the region's start to the end of its highest live
    /// allocation.
    pub fn used(&self) -> usize {
        self.inner
            .lock()
            .live
            .last_key_value()
            .map_or(0, |(offset, nbytes)| offset + nbytes)
    }

    /// Bytes of allocations not yet freed.
    pub fn live_bytes(&self) -> usize {
        self.inner.lock().live_bytes
    }
}

impl<B> Inner<B> {
    /// The block an allocation of `nbytes` takes: rounded up to the
    /// alignment, so every block starts aligned.
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
            return; // took no block
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

impl<B: Allocator + 'static> Allocator for StaticAllocator<B> {
    fn allocate(&self, nbytes: usize) -> DataPtr {
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

    /// `None` when the region cannot be reserved or has no room left.
    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        let inner = &self.inner;
        let mut state = inner.lock();
        if state.region.is_none() {
            state.region = Some(inner.backend.try_allocate(inner.capacity)?);
        }
        let block = inner.block(nbytes);
        // An empty allocation takes no block: any aligned offset will do.
        let (&offset, &len) = match nbytes {
            0 => state.free.iter().next().unwrap_or((&0, &0)),
            _ => state.free.iter().find(|&(_, &len)| len >= block)?,
        };
        let base = state.region.as_ref().expect("reserved above").as_ptr();
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
        let owner = Arc::clone(inner);
        Some(DataPtr::with_deleter(
            ptr,
            Layout::from_size_align(nbytes, inner.alignment).expect("valid layout"),
            move |_| owner.free(addr, offset, nbytes),
        ))
    }
}
