//! A device-agnostic caching allocator, modeled on c10's
//! `CUDACachingAllocator` (`c10/cuda/CUDACachingAllocator.cpp`).
//!
//! Raw device memory operations are abstracted behind [`DeviceBackend`] so
//! the pooling logic is shared across device families (CUDA, MPS) and can be
//! tested with a mock. Tuning comes from [`CacheConfig`]; PyTorch likewise
//! shares this design between `CUDACachingAllocator` and `MPSAllocator`,
//! though with separate implementations and MPS-specific extras we do not
//! model (MTLHeap suballocation, watermark GC).
//!
//! Layout of the machinery:
//! - A *segment* is one big backend allocation.
//! - A *block* is a slice of a segment handed out (or cached) by `allocate`.
//! - Free blocks live in size-ordered pools; freed blocks coalesce with
//!   neighbors; `empty_cache` releases whole free segments.

use crate::allocator::{Allocator, DataPtr};
use crate::device::Device;
use std::alloc::Layout;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

// ---------------- tuning ----------------

/// All sizes are rounded up to a multiple of this (c10: kMinBlockSize).
pub const K_MIN_BLOCK_SIZE: usize = 512;
/// Largest "small" allocation; anything bigger goes to the large pool (c10: kSmallSize).
pub const K_SMALL_SIZE: usize = 1 << 20; // 1 MiB
/// Allocations at least this big bypass splitting (c10: kMinLargeAlloc).
pub const K_MIN_LARGE_ALLOC: usize = 10 << 20; // 10 MiB
/// Round-up granularity for large allocations (c10: kRoundLarge).
pub const K_ROUND_LARGE: usize = 2 << 20; // 2 MiB

/// Segment-size tuning for a [`CachingAllocator`]. The thresholds
/// (`K_SMALL_SIZE`, `K_MIN_LARGE_ALLOC`, `K_ROUND_LARGE`) are shared by CUDA
/// and MPS in PyTorch; only the segment sizes differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheConfig {
    /// Segment size carved up for small allocations (c10: kSmallBuffer /
    /// MPS: kSmallHeap).
    pub small_buffer: usize,
    /// Segment size for mid-size large allocations (c10: kLargeBuffer /
    /// MPS: kLargeHeap).
    pub large_buffer: usize,
}

impl CacheConfig {
    /// c10's CUDA values: 2 MiB / 20 MiB segments.
    pub const fn cuda() -> Self {
        Self {
            small_buffer: 2 << 20,
            large_buffer: 20 << 20,
        }
    }

    /// aten's MPS values: 8 MiB / 32 MiB heaps, mapped onto our segments.
    pub const fn mps() -> Self {
        Self {
            small_buffer: 8 << 20,
            large_buffer: 32 << 20,
        }
    }
}

/// Round up to a multiple of `K_MIN_BLOCK_SIZE`.
fn round_size(size: usize) -> usize {
    size.div_ceil(K_MIN_BLOCK_SIZE) * K_MIN_BLOCK_SIZE
}

/// How big a segment to grab from the device for a request of `size` bytes.
fn segment_size(size: usize, config: &CacheConfig) -> usize {
    if size <= K_SMALL_SIZE {
        config.small_buffer
    } else if size < K_MIN_LARGE_ALLOC {
        config.large_buffer
    } else {
        size.div_ceil(K_ROUND_LARGE) * K_ROUND_LARGE
    }
}

// ---------------- backend abstraction ----------------

/// Raw device memory operations. Implemented by the CUDA backend (`cuda.rs`),
/// the Metal backend (`mps.rs`), and by mocks in tests.
///
/// # Safety
/// Implementations must return pointers to at least `nbytes` of device
/// memory that remain valid until `device_free`.
pub trait DeviceBackend: Send + Sync + 'static {
    /// Allocate `nbytes` on the device; null on failure.
    ///
    /// # Safety
    /// The returned pointer must denote at least `nbytes` of device memory
    /// that stays valid until passed to `device_free`.
    unsafe fn device_alloc(&self, nbytes: usize) -> *mut u8;
    /// Free a pointer from `device_alloc`.
    ///
    /// # Safety
    /// `ptr` must come from `device_alloc` and not have been freed already.
    unsafe fn device_free(&self, ptr: *mut u8);
}

// ---------------- stats ----------------

/// A snapshot of the allocator's counters (subset of c10's `DeviceStats`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Bytes currently handed out to users.
    pub allocated_bytes: usize,
    /// Peak of `allocated_bytes` since the last `reset_peak_stats`.
    pub allocated_bytes_peak: usize,
    /// Bytes currently obtained from the device (cached + in use).
    pub reserved_bytes: usize,
    /// Peak of `reserved_bytes`.
    pub reserved_bytes_peak: usize,
    /// Number of user allocations served.
    pub num_alloc: usize,
    /// Number of user blocks freed back to the cache.
    pub num_free: usize,
    /// Number of backend allocations (segments).
    pub num_device_alloc: usize,
    /// Number of backend frees.
    pub num_device_free: usize,
}

// ---------------- internals ----------------

/// One contiguous slice of device memory tracked by the allocator.
#[derive(Debug)]
struct Block {
    /// Start address (as an integer; the backend treats it as opaque).
    ptr: usize,
    size: usize,
    /// True while handed out to a user.
    allocated: bool,
    /// Blocks from the same `device_alloc` call form a segment; a segment is
    /// released only when every one of its blocks is free.
    segment_start: usize,
    segment_size: usize,
    /// Which pool this block belongs to. Membership is a property of the
    /// segment the block was carved from, NOT of its current size: a small
    /// block that coalesces back into a whole 2 MiB segment stays in the
    /// small pool. (c10 stores a `pool` pointer on each Block for the same
    /// reason.)
    pool_small: bool,
}

/// Key for the pool sets: ordered by size, then address, matching c10's
/// `BlockPool` comparator (best-fit with lowest address wins).
type PoolKey = (usize, usize);

#[derive(Debug, Default)]
struct State {
    /// All blocks ever carved, keyed by address (enables neighbor lookups).
    blocks: BTreeMap<usize, Block>,
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

struct Inner<B: DeviceBackend> {
    device: Device,
    config: CacheConfig,
    backend: B,
    state: Mutex<State>,
}

/// A caching device allocator. Clone it freely: clones share one cache.
///
/// Construct via the device modules (`cuda::get`, `mps::get`) or directly
/// with a custom [`DeviceBackend`].
pub struct CachingAllocator<B: DeviceBackend> {
    inner: Arc<Inner<B>>,
}

impl<B: DeviceBackend> Clone for CachingAllocator<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<B: DeviceBackend> CachingAllocator<B> {
    pub fn new(device: Device, backend: B, config: CacheConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                device,
                config,
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
        let releasable: Vec<usize> = state
            .blocks
            .iter()
            .filter(|(_, b)| b.ptr == b.segment_start && !b.allocated)
            .filter(|(_, b)| {
                // The whole segment must be free: check successors by
                // walking forward through the address-ordered map.
                let mut covered = b.segment_size;
                let mut addr = b.ptr;
                while covered > 0 {
                    match state.blocks.get(&addr) {
                        Some(block) if !block.allocated => {
                            covered -= block.size;
                            addr += block.size;
                        }
                        _ => return false,
                    }
                }
                true
            })
            .map(|(ptr, _)| *ptr)
            .collect();

        for ptr in releasable {
            let mut addr = ptr;
            while let Some(block) = state.blocks.remove(&addr) {
                let is_segment_end =
                    block.ptr + block.size == block.segment_start + block.segment_size;
                state
                    .pool_for(block.pool_small)
                    .remove(&(block.size, block.ptr));
                state.stats.reserved_bytes -= block.size;
                state.stats.num_device_free += 1;
                addr = block.ptr + block.size;
                if is_segment_end {
                    break;
                }
            }
            unsafe { self.inner.backend.device_free(ptr as *mut u8) };
        }
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        if nbytes == 0 {
            return DataPtr::with_deleter(
                std::ptr::NonNull::dangling(),
                Layout::from_size_align(0, 1).unwrap(),
                |_| {},
            );
        }

        let size = round_size(nbytes);
        let mut state = self.inner.state.lock().unwrap();

        // 1. Best-fit search in the pool.
        if let Some(block) = Self::try_alloc(&mut state, size) {
            return self.make_data_ptr(&mut state, block);
        }

        // 2. Nothing cached: ask the device for a fresh segment.
        let seg_size = segment_size(size, &self.inner.config);
        match self.alloc_new_segment(&mut state, size, seg_size) {
            Ok(block) => self.make_data_ptr(&mut state, block),
            // 3. Device is out of memory: release cached segments and retry
            //    once (c10's OutOfMemoryError path).
            Err(_) => {
                drop(state);
                self.empty_cache();
                let mut state = self.inner.state.lock().unwrap();
                match self.alloc_new_segment(&mut state, size, seg_size) {
                    Ok(block) => self.make_data_ptr(&mut state, block),
                    Err(e) => {
                        // Release the lock before panicking: panicking while
                        // holding the guard would poison the cache's mutex,
                        // making the allocator unusable for whoever catches
                        // the panic (and panic again in our own Drop).
                        drop(state);
                        panic!("{e} (after empty_cache retry)");
                    }
                }
            }
        }
    }

    /// Find the smallest free block that fits, splitting it if it is much
    /// bigger than needed. Returns the allocated block's address.
    fn try_alloc(state: &mut State, size: usize) -> Option<usize> {
        let small = size <= K_SMALL_SIZE;
        let pool = if small {
            &state.small_pool
        } else {
            &state.large_pool
        };
        // First entry >= (size, 0) is the best fit.
        let key = *pool.range((size, 0)..).next()?;
        let (block_size, ptr) = key;

        let should_split = if size <= K_SMALL_SIZE {
            // c10: remaining >= kMinBlockSize (keep small blocks splittable)
            block_size - size >= K_MIN_BLOCK_SIZE
        } else {
            // c10: remaining > kSmallSize
            block_size - size > K_SMALL_SIZE
        };

        state.pool_for(small).remove(&key);
        let mut block = state.blocks.remove(&ptr).expect("pool/block mismatch");
        if should_split {
            let remaining = Block {
                ptr: ptr + size,
                size: block_size - size,
                allocated: false,
                segment_start: block.segment_start,
                segment_size: block.segment_size,
                pool_small: small,
            };
            block.size = size;
            state.blocks.insert(remaining.ptr, remaining);
            state
                .pool_for(small)
                .insert((block_size - size, ptr + size));
        }
        block.allocated = true;
        state.blocks.insert(ptr, block);
        state.stats.allocated_bytes += size;
        state.stats.allocated_bytes_peak = state
            .stats
            .allocated_bytes_peak
            .max(state.stats.allocated_bytes);
        state.stats.num_alloc += 1;
        Some(ptr)
    }

    /// Obtain a new segment from the backend and carve the request out of it.
    fn alloc_new_segment(
        &self,
        state: &mut State,
        size: usize,
        seg_size: usize,
    ) -> Result<usize, String> {
        let ptr = unsafe { self.inner.backend.device_alloc(seg_size) };
        if ptr.is_null() {
            return Err(format!(
                "failed to allocate {seg_size} bytes on {}",
                self.inner.device
            ));
        }
        let ptr = ptr as usize;
        let small = size <= K_SMALL_SIZE;
        state.stats.reserved_bytes += seg_size;
        state.stats.reserved_bytes_peak = state
            .stats
            .reserved_bytes_peak
            .max(state.stats.reserved_bytes);
        state.stats.num_device_alloc += 1;

        let mut block = Block {
            ptr,
            size: seg_size,
            allocated: false,
            segment_start: ptr,
            segment_size: seg_size,
            pool_small: small,
        };
        // Split the request off the front; cache the remainder.
        if seg_size > size {
            let remaining = Block {
                ptr: ptr + size,
                size: seg_size - size,
                allocated: false,
                segment_start: ptr,
                segment_size: seg_size,
                pool_small: small,
            };
            state.blocks.insert(remaining.ptr, remaining);
            state.pool_for(small).insert((seg_size - size, ptr + size));
        }
        block.size = size;
        block.allocated = true;
        state.blocks.insert(ptr, block);
        state.stats.allocated_bytes += size;
        state.stats.allocated_bytes_peak = state
            .stats
            .allocated_bytes_peak
            .max(state.stats.allocated_bytes);
        state.stats.num_alloc += 1;
        Ok(ptr)
    }

    /// Wrap an allocated block in a `DataPtr` whose deleter returns the block
    /// to the cache.
    fn make_data_ptr(&self, state: &mut State, ptr: usize) -> DataPtr {
        let block_size = state.blocks[&ptr].size;
        let inner = Arc::clone(&self.inner);
        DataPtr::with_deleter(
            std::ptr::NonNull::new(ptr as *mut u8).expect("block ptr null"),
            Layout::from_size_align(block_size, K_MIN_BLOCK_SIZE).unwrap(),
            move |ptr| inner.free_block(ptr.as_ptr() as usize, block_size),
        )
    }
}

impl<B: DeviceBackend> Inner<B> {
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
            && state.blocks[&prev_ptr].segment_start == state.blocks[&ptr].segment_start
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
            && next.segment_start == state.blocks[&start].segment_start
        {
            let (next_size, small) = (next.size, next.pool_small);
            state.pool_for(small).remove(&(next_size, end));
            state.blocks.remove(&end);
            state.blocks.get_mut(&start).unwrap().size += next_size;
        }
        start
    }
}

impl<B: DeviceBackend> Allocator for CachingAllocator<B> {
    fn device(&self) -> Device {
        self.inner.device
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        CachingAllocator::allocate(self, nbytes)
    }
}

/// Flush the cache when the last clone of an allocator goes away.
impl<B: DeviceBackend> Drop for CachingAllocator<B> {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.empty_cache();
        }
    }
}
