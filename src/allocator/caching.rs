pub use super::cache_stats::CacheStats;
use super::traits::{CachePolicy, K_SMALL_SIZE};
use crate::allocator::{Allocator, DataPtr};
use crate::device::Device;
use std::alloc::Layout;
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, PoisonError};

// ---------------- internals ----------------

/// One contiguous slice of device memory tracked by the allocator.
#[derive(Debug)]
struct Block {
    /// Start address (as an integer; the backend treats it as opaque).
    ptr: usize,
    size: usize,
    /// True while handed out to a user.
    allocated: bool,
    /// Base address of the segment this block was carved from
    segment_base: usize,
    segment_size: usize,
    /// Which pool this block belongs to. Membership is a property of the
    /// segment the block was carved from, NOT of its current size: a small
    /// block that coalesces back into a whole 2 MiB segment stays in the
    /// small pool.
    pool_small: bool,
}

/// Key for the pool sets: ordered by size, then address (best-fit with lowest address wins).
type PoolKey = (usize, usize);

#[derive(Default)]
struct State {
    /// All blocks ever carved, keyed by address (enables neighbor lookups).
    blocks: BTreeMap<usize, Block>,
    /// Owning handle for each backend segment, keyed by base address.
    /// Dropping the `DataPtr` frees the segment on the device.
    segments: BTreeMap<usize, DataPtr>,
    /// Free blocks up to `K_SMALL_SIZE`.
    small_pool: BTreeSet<PoolKey>,
    /// Free blocks above `K_SMALL_SIZE`.
    large_pool: BTreeSet<PoolKey>,
    stats: CacheStats,
}

impl State {
    fn pool_for(&mut self, small: bool) -> &mut BTreeSet<PoolKey> {
        if small {
            &mut self.small_pool
        } else {
            &mut self.large_pool
        }
    }
}

struct Inner<B: DeviceBackend, P: CachePolicy> {
    device: Device,
    policy: P,
    backend: B,
    state: Mutex<State>,
}

/// A caching device allocator. Clone it freely: clones share one cache.
///
/// Construct via the device modules (`cuda::get`, `mps::get`) or directly
/// with a custom [`DeviceBackend`].
pub struct CachingAllocator<B: DeviceBackend, P: CachePolicy> {
    inner: Arc<Inner<B, P>>,
}

impl<B: DeviceBackend, P: CachePolicy> Clone for CachingAllocator<B, P> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<B: DeviceBackend, P: CachePolicy> CachingAllocator<B, P> {
    pub fn new(device: Device, backend: B, policy: P) -> Self {
        Self {
            inner: Arc::new(Inner {
                device,
                policy,
                backend,
                state: Mutex::new(State::default()),
            }),
        }
    }

    /// Current statistics snapshot.
    pub fn stats(&self) -> CacheStats {
        self.inner.state.lock().unwrap().stats
    }

    /// Release every cached block whose entire segment is free back to the
    /// device (c10: `emptyCache` / `releaseCachedBlocks`).
    pub fn empty_cache(&self) {
        let mut state = self.inner.state.lock().unwrap();
        self.inner.release_free_segments(&mut state);
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        if nbytes == 0 {
            return DataPtr::with_deleter(
                NonNull::dangling(),
                Layout::from_size_align(0, 1).unwrap(),
                |_| {},
            );
        }

        let policy = &self.inner.policy;
        policy.check_request(nbytes);
        let size = policy.round_size(nbytes);
        let mut state = self.inner.state.lock().unwrap();

        // 1. Best-fit search in the pool.
        if let Some(block) = self.try_alloc(&mut state, size) {
            return self.make_data_ptr(&state, block);
        }

        // 2. Nothing cached: ask the device for a fresh segment.
        let seg_size = policy.segment_size(size, state.stats.reserved_bytes);
        match self.alloc_new_segment(&mut state, size, seg_size) {
            Ok(block) => self.make_data_ptr(&state, block),
            // 3. Device is out of memory: release cached segments and retry
            //    once (c10's OutOfMemoryError path).
            Err(_) => {
                drop(state);
                self.empty_cache();
                let mut state = self.inner.state.lock().unwrap();
                match self.alloc_new_segment(&mut state, size, seg_size) {
                    Ok(block) => self.make_data_ptr(&state, block),
                    Err(e) => {
                        // Release the lock before panicking: panicking while
                        // holding the guard would poison the cache's mutex,
                        // making the allocator unusable for whoever catches
                        // the panic.
                        drop(state);
                        panic!("{e} (after empty_cache retry)");
                    }
                }
            }
        }
    }

    /// Find the smallest free block that fits and hand it out. Returns the
    /// allocated block's address.
    fn try_alloc(&self, state: &mut State, size: usize) -> Option<usize> {
        let pool = if size <= K_SMALL_SIZE {
            &state.small_pool
        } else {
            &state.large_pool
        };
        // First entry >= (size, 0) is the best fit.
        let &(_, ptr) = pool.range((size, 0)..).next()?;
        Some(self.take_block(state, ptr, size))
    }

    /// Hand out the free block at `ptr` for a `size`-byte request, splitting
    /// off the remainder if it is worth keeping
    /// (c10: `alloc_found_block`, MPS: `split_free_block`).
    fn take_block(&self, state: &mut State, ptr: usize, size: usize) -> usize {
        let block = &state.blocks[&ptr];
        let (block_size, small) = (block.size, block.pool_small);
        let (segment, segment_size) = (block.segment, block.segment_size);
        state.pool_for(small).remove(&(block_size, ptr));

        let remaining = block_size - size;
        if self.inner.policy.should_split(small, remaining) {
            state.blocks.insert(
                ptr + size,
                Block {
                    ptr: ptr + size,
                    size: remaining,
                    allocated: false,
                    segment,
                    segment_size,
                    pool_small: small,
                },
            );
            state.pool_for(small).insert((remaining, ptr + size));
            state.blocks.get_mut(&ptr).unwrap().size = size;
        }

        let block = state.blocks.get_mut(&ptr).unwrap();
        block.allocated = true;
        // Count the whole block: it may be bigger than `size` if unsplit.
        state.stats.allocated_bytes += block.size;
        state.stats.allocated_bytes_peak = state
            .stats
            .allocated_bytes_peak
            .max(state.stats.allocated_bytes);
        state.stats.num_alloc += 1;
        ptr
    }

    /// Obtain a new segment from the backend and carve the request out of it
    /// (c10: `alloc_block`, MPS: `alloc_heap`).
    fn alloc_new_segment(
        &self,
        state: &mut State,
        size: usize,
        seg_size: usize,
    ) -> Result<usize, String> {
        let raw = unsafe { self.inner.backend.device_alloc(seg_size) };
        let Some(segment) = NonNull::new(raw) else {
            return Err(format!(
                "failed to allocate {seg_size} bytes on {}",
                self.inner.device
            ));
        };
        state.stats.reserved_bytes += seg_size;
        state.stats.reserved_bytes_peak = state
            .stats
            .reserved_bytes_peak
            .max(state.stats.reserved_bytes);
        state.stats.num_device_alloc += 1;

        // The segment starts out as one free block the request is cut from.
        let ptr = raw.addr();
        let small = size <= K_SMALL_SIZE;
        state.blocks.insert(
            ptr,
            Block {
                ptr,
                size: seg_size,
                allocated: false,
                segment,
                segment_size: seg_size,
                pool_small: small,
            },
        );
        state.pool_for(small).insert((seg_size, ptr));
        Ok(self.take_block(state, ptr, size))
    }

    /// Wrap an allocated block in a `DataPtr` whose deleter returns the block
    /// to the cache.
    fn make_data_ptr(&self, state: &State, ptr: usize) -> DataPtr {
        let block = &state.blocks[&ptr];
        let (block_size, segment) = (block.size, block.segment);
        let inner = Arc::clone(&self.inner);
        DataPtr::with_deleter(
            NonNull::new(segment.as_ptr().with_addr(ptr)).expect("block ptr null"),
            Layout::from_size_align(block_size, self.inner.policy.alignment()).unwrap(),
            move |ptr| inner.free_block(ptr.addr().get(), block_size),
        )
    }
}

impl<B: DeviceBackend, P: CachePolicy> Inner<B, P> {
    /// Return a block to its pool, coalescing with free neighbors
    /// (c10: `free_block` + `coalesce_block`).
    fn free_block(&self, ptr: usize, size: usize) {
        let mut state = self.state.lock().unwrap();
        let block = state.blocks.get_mut(&ptr).expect("free of unknown block");
        debug_assert!(block.allocated);
        block.allocated = false;
        state.stats.allocated_bytes -= size;
        state.stats.num_free += 1;

        let ptr = Self::coalesce_block(&mut state, ptr);
        let block = &state.blocks[&ptr];
        let (size, ptr, small) = (block.size, block.ptr, block.pool_small);
        state.pool_for(small).insert((size, ptr));
    }

    /// Merge the block at `ptr` with adjacent free blocks in its segment.
    /// Returns the (possibly earlier) address of the merged block.
    fn coalesce_block(state: &mut State, ptr: usize) -> usize {
        let mut start = ptr;
        // Predecessor: the block ending at our start.
        let prev_ptr = state.blocks.range(..ptr).next_back().map(|(&p, _)| p);
        if let Some(prev_ptr) = prev_ptr
            && !state.blocks[&prev_ptr].allocated
            && state.blocks[&prev_ptr].segment == state.blocks[&ptr].segment
        {
            let merged_size = state.blocks[&prev_ptr].size + state.blocks[&ptr].size;
            let (prev_size, small) = (
                state.blocks[&prev_ptr].size,
                state.blocks[&prev_ptr].pool_small,
            );
            state.pool_for(small).remove(&(prev_size, prev_ptr));
            state.blocks.remove(&ptr);
            state.blocks.get_mut(&prev_ptr).unwrap().size = merged_size;
            start = prev_ptr;
        }
        // Successor: the block starting at our end.
        let end = start + state.blocks[&start].size;
        if let Some(next) = state.blocks.get(&end)
            && !next.allocated
            && next.segment == state.blocks[&start].segment
        {
            let (next_size, small) = (next.size, next.pool_small);
            state.pool_for(small).remove(&(next_size, end));
            state.blocks.remove(&end);
            state.blocks.get_mut(&start).unwrap().size += next_size;
        }
        start
    }

    /// Return every wholly free segment to the device. Free neighbors always
    /// coalesce, so a segment is wholly free exactly when one free block
    /// spans it (c10: `release_cached_blocks`, MPS: `release_free_heaps`).
    fn release_free_segments(&self, state: &mut State) {
        let whole: Vec<usize> = state
            .blocks
            .values()
            .filter(|b| !b.allocated && b.size == b.segment_size)
            .map(|b| b.ptr)
            .collect();
        for ptr in whole {
            let block = state.blocks.remove(&ptr).unwrap();
            state.pool_for(block.pool_small).remove(&(block.size, ptr));
            state.stats.reserved_bytes -= block.size;
            state.stats.num_device_free += 1;
            // SAFETY: a whole segment from device_alloc; no block refers to
            // it any more.
            unsafe { self.backend.device_free(block.segment.as_ptr()) };
        }
    }
}

/// Flush the cache once the last reference goes away. Every `DataPtr` holds
/// a reference too, so by now every block is free.
impl<B: DeviceBackend, P: CachePolicy> Drop for Inner<B, P> {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        let mut state = std::mem::take(state);
        self.release_free_segments(&mut state);
    }
}

impl<B: DeviceBackend, P: CachePolicy> Allocator for CachingAllocator<B, P> {
    fn device(&self) -> Device {
        self.inner.device
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        CachingAllocator::allocate(self, nbytes)
    }
}
