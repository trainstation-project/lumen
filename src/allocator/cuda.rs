//! CUDA device allocator, modeled on PyTorch's `CUDACachingAllocator`
//! (`c10/cuda/CUDACachingAllocator.cpp`).
//!
//! Why caching? `cudaMalloc`/`cudaFree` are expensive, synchronizing
//! driver calls. PyTorch therefore never frees device memory on tensor
//! death; it returns the block to a pool and reuses it. The design:
//!
//! * **Two pools per device**: requests ≤ 1 MiB come from the *small*
//!   pool (2 MiB `cudaMalloc` segments), larger ones from the *large*
//!   pool (20 MiB segments, or 2 MiB-rounded for ≥ 10 MiB requests).
//! * **Size rounding**: every request rounds up to a 512 B multiple, so
//!   cached blocks fit many similar request sizes.
//! * **Splitting**: a block much larger than the request is split; the
//!   remainder goes back into the pool.
//! * **Coalescing**: freed blocks merge with free neighbors so segments
//!   reassemble into whole `cudaMalloc`ed regions.
//! * **No `cudaFree` on drop**: the `DataPtr` deleter just returns the
//!   block to its pool. Memory goes back to the driver only via
//!   [`CachingAllocator::empty_cache`] or under allocation pressure.
//!
//! The raw device operations are abstracted behind [`DeviceBackend`]
//! (PyTorch calls `cudaMalloc`/`cudaFree` directly). The real CUDA
//! backend lives behind the `cuda` cargo feature; tests use a mock
//! backend, which lets the caching logic be exercised without a GPU.
//!
//! Not modeled (prototype scope): per-stream pools and `record_stream`
//! (PyTorch tracks which streams used a block to avoid reuse-before-
//! kernel-completion), expandable segments, memory pools for CUDA
//! graphs, and `max_split_size` configuration.

use std::alloc::Layout;
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};

use super::{Allocator, DataPtr};
use crate::device::Device;

// Configuration constants (c10/core/AllocatorConfig.h).
/// All request sizes round up to a multiple of this.
pub const K_MIN_BLOCK_SIZE: usize = 512;
/// Largest request served by the small pool.
pub const K_SMALL_SIZE: usize = 1024 * 1024; // 1 MiB
/// Segment size `cudaMalloc`ed for the small pool.
pub const K_SMALL_BUFFER: usize = 2 * 1024 * 1024; // 2 MiB
/// Requests above this get an exactly-sized (2 MiB-rounded) segment.
pub const K_MIN_LARGE_ALLOC: usize = 10 * 1024 * 1024; // 10 MiB
/// Segment size for large-pool requests below [`K_MIN_LARGE_ALLOC`].
pub const K_LARGE_BUFFER: usize = 20 * 1024 * 1024; // 20 MiB
/// Very large requests round up to a multiple of this.
pub const K_ROUND_LARGE: usize = 2 * 1024 * 1024; // 2 MiB

/// Round a request size up to a [`K_MIN_BLOCK_SIZE`] multiple
/// (`CUDACachingAllocator::round_size`).
fn round_size(size: usize) -> usize {
    size.max(K_MIN_BLOCK_SIZE).div_ceil(K_MIN_BLOCK_SIZE) * K_MIN_BLOCK_SIZE
}

/// How many bytes to `cudaMalloc` for a request of `size` bytes
/// (`CUDACachingAllocator::get_allocation_size`).
fn segment_size(size: usize) -> usize {
    if size <= K_SMALL_SIZE {
        K_SMALL_BUFFER
    } else if size < K_MIN_LARGE_ALLOC {
        K_LARGE_BUFFER
    } else {
        size.div_ceil(K_ROUND_LARGE) * K_ROUND_LARGE
    }
}

/// Raw device memory operations — the two calls PyTorch makes to the
/// CUDA driver (`cudaMalloc` / `cudaFree`). Abstracted so the caching
/// logic can be tested without a GPU.
pub trait DeviceBackend: Send + Sync {
    /// Allocate `nbytes` on the device; return null on failure (OOM).
    ///
    /// # Safety
    /// The returned pointer must be valid for `nbytes` and freed only
    /// via [`DeviceBackend::device_free`].
    unsafe fn device_alloc(&self, nbytes: usize) -> *mut u8;

    /// Free a pointer from [`DeviceBackend::device_alloc`].
    ///
    /// # Safety
    /// `ptr` must come from `device_alloc` and not already be freed.
    unsafe fn device_free(&self, ptr: *mut u8);
}

/// Allocation statistics (`CUDACachingAllocator::DeviceStats`, trimmed).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Bytes currently handed out to users (block sizes, not requests).
    pub allocated_bytes: usize,
    pub allocated_bytes_peak: usize,
    /// Bytes currently held from the device (`cudaMalloc`ed segments).
    pub reserved_bytes: usize,
    pub reserved_bytes_peak: usize,
    /// Calls to `allocate`.
    pub num_alloc: u64,
    /// Blocks returned to the cache (tensor deaths).
    pub num_free: u64,
    /// `cudaMalloc` calls.
    pub num_device_alloc: u64,
    /// `cudaFree` calls.
    pub num_device_free: u64,
}

/// One contiguous run of device memory, either a whole segment or a
/// piece of one (`c10::cuda::CUDACachingAllocator::Block`).
#[derive(Debug)]
struct Block {
    /// Total bytes in this block (after any splits).
    size: usize,
    /// Bytes the user asked for; 0 while the block sits free in a pool.
    requested: usize,
    /// Which pool this block belongs to when free.
    small_pool: bool,
    /// Whether the block is currently free (in a pool).
    free: bool,
    /// Base address of the `cudaMalloc`ed segment this block was carved
    /// from. Two segments can be address-adjacent, so address adjacency
    /// alone doesn't mean two blocks may merge; they must also share a
    /// segment (PyTorch's prev/next links never cross segments).
    segment: usize,
}

/// Mutable allocator state, guarded by a mutex (PyTorch uses
/// `std::recursive_mutex` per device the same way).
struct State {
    /// Every block ever carved, keyed by start address. Adjacency for
    /// coalescing is found by range queries (restricted to the same
    /// segment) instead of PyTorch's intrusive prev/next pointers.
    blocks: BTreeMap<usize, Block>,
    /// Free blocks, ordered by `(size, ptr)` for best-fit lookup
    /// (PyTorch's `BlockPool` is a `std::set` with the same ordering).
    small_pool: BTreeSet<(usize, usize)>,
    large_pool: BTreeSet<(usize, usize)>,
    stats: CacheStats,
}

impl State {
    fn pool(&mut self, small: bool) -> &mut BTreeSet<(usize, usize)> {
        if small {
            &mut self.small_pool
        } else {
            &mut self.large_pool
        }
    }

    fn insert_free(&mut self, ptr: usize) {
        let block = self.blocks.get_mut(&ptr).expect("block must exist");
        block.free = true;
        block.requested = 0;
        let (size, small) = (block.size, block.small_pool);
        self.pool(small).insert((size, ptr));
    }

    fn remove_free(&mut self, ptr: usize) -> (usize, bool) {
        let block = self.blocks.get_mut(&ptr).expect("block must exist");
        block.free = false;
        let (size, small) = (block.size, block.small_pool);
        let removed = self.pool(small).remove(&(size, ptr));
        debug_assert!(removed, "free block must be in its pool");
        (size, small)
    }

    /// Best-fit search: smallest free block with size >= `size` in the
    /// pool for `small` requests (`CUDACachingAllocator::get_free_block`).
    fn get_free_block(&mut self, size: usize, small: bool) -> Option<usize> {
        let &(_, ptr) = self.pool(small).range((size, 0)..).next()?;
        self.remove_free(ptr);
        Some(ptr)
    }

    /// Split `ptr`'s block down to `size` if the remainder is worth
    /// keeping (`CUDACachingAllocator::should_split` + `alloc_found_block`).
    fn maybe_split(&mut self, ptr: usize, size: usize) {
        let block = self.blocks.get(&ptr).expect("block must exist");
        let remaining = block.size - size;
        let should_split = if block.small_pool {
            remaining >= K_MIN_BLOCK_SIZE
        } else {
            remaining > K_SMALL_SIZE
        };
        if !should_split {
            return;
        }
        let rest_ptr = ptr + size;
        let (small, segment) = (block.small_pool, block.segment);
        self.blocks.get_mut(&ptr).expect("block must exist").size = size;
        self.blocks.insert(
            rest_ptr,
            Block {
                size: remaining,
                requested: 0,
                small_pool: small,
                free: false,
                segment,
            },
        );
        self.insert_free(rest_ptr);
    }

    /// The block ending right before `ptr` in the same segment, if any
    /// (PyTorch's `block->prev`).
    fn prev_in_segment(&self, ptr: usize) -> Option<usize> {
        let segment = self.blocks[&ptr].segment;
        let (&p, b) = self.blocks.range(..ptr).next_back()?;
        (p + b.size == ptr && b.segment == segment).then_some(p)
    }

    /// The block starting right after `ptr`'s block in the same segment,
    /// if any (PyTorch's `block->next`).
    fn next_in_segment(&self, ptr: usize) -> Option<usize> {
        let block = &self.blocks[&ptr];
        let end = ptr + block.size;
        let next = self.blocks.get(&end)?;
        (next.segment == block.segment).then_some(end)
    }

    /// Merge `ptr`'s (free) block with free neighbors in its segment
    /// (`CUDACachingAllocator::try_merge_blocks`, called on free).
    fn coalesce(&mut self, ptr: usize) {
        // Merge with the previous block if it is free.
        if let Some(prev_ptr) = self.prev_in_segment(ptr)
            && self.blocks[&prev_ptr].free
        {
            let size = self.blocks[&ptr].size;
            self.remove_free(prev_ptr);
            self.remove_free(ptr);
            self.blocks.get_mut(&prev_ptr).expect("prev exists").size += size;
            self.blocks.remove(&ptr);
            self.insert_free(prev_ptr);
            // The merged block may now be adjacent to the next one.
            return self.coalesce_next(prev_ptr);
        }
        self.coalesce_next(ptr);
    }

    fn coalesce_next(&mut self, ptr: usize) {
        let Some(next_ptr) = self.next_in_segment(ptr) else {
            return;
        };
        if !self.blocks[&next_ptr].free {
            return;
        }
        let next_size = self.blocks[&next_ptr].size;
        self.remove_free(ptr);
        self.remove_free(next_ptr);
        self.blocks.get_mut(&ptr).expect("block exists").size += next_size;
        self.blocks.remove(&next_ptr);
        self.insert_free(ptr);
    }

    /// Return all releasable free blocks to the device
    /// (`CUDACachingAllocator::release_cached_blocks` / `emptyCache`).
    ///
    /// Only *whole segments* can go back to the driver: a free block
    /// with a neighbor in its segment is a split piece of a segment
    /// whose other piece is still in use (free neighbors would have
    /// coalesced). PyTorch checks the same via `block->prev/next ==
    /// nullptr`.
    ///
    /// Returns the pointers to `device_free` (done outside the lock).
    fn drain_free_blocks(&mut self) -> Vec<(usize, usize)> {
        let free_ptrs: Vec<usize> = self
            .small_pool
            .iter()
            .chain(self.large_pool.iter())
            .map(|&(_, ptr)| ptr)
            .collect();
        let mut freed = Vec::new();
        for ptr in free_ptrs {
            if self.prev_in_segment(ptr).is_some() || self.next_in_segment(ptr).is_some() {
                continue; // split piece of a busy segment; keep it cached
            }
            let block = self.blocks.remove(&ptr).expect("free block exists");
            self.pool(block.small_pool).remove(&(block.size, ptr));
            freed.push((ptr, block.size));
            self.stats.reserved_bytes -= block.size;
            self.stats.num_device_free += 1;
        }
        freed
    }
}

struct Inner<B: DeviceBackend> {
    device_index: usize,
    backend: B,
    state: Mutex<State>,
}

impl<B: DeviceBackend> Inner<B> {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("caching allocator mutex poisoned")
    }

    /// A block's `DataPtr` dropped: return it to its pool and coalesce
    /// (`CUDACachingAllocator::free`). No `cudaFree` happens here.
    fn free_block(&self, ptr: usize, requested: usize) {
        let mut state = self.lock();
        state.stats.num_free += 1;
        state.stats.allocated_bytes -= requested;
        state.insert_free(ptr);
        state.coalesce(ptr);
    }
}

/// A caching device allocator (`c10::cuda::CUDACachingAllocator`).
///
/// Cloneable handle around shared state; `DataPtr` deleters hold their
/// own reference, so the cache outlives every pointer it handed out
/// (PyTorch uses a global per-device allocator for the same reason).
pub struct CachingAllocator<B: DeviceBackend> {
    inner: Arc<Inner<B>>,
}

impl<B: DeviceBackend> Clone for CachingAllocator<B> {
    fn clone(&self) -> Self {
        CachingAllocator {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<B: DeviceBackend> CachingAllocator<B> {
    pub fn new(device_index: usize, backend: B) -> Self {
        CachingAllocator {
            inner: Arc::new(Inner {
                device_index,
                backend,
                state: Mutex::new(State {
                    blocks: BTreeMap::new(),
                    small_pool: BTreeSet::new(),
                    large_pool: BTreeSet::new(),
                    stats: CacheStats::default(),
                }),
            }),
        }
    }

    /// Snapshot of the allocator's statistics.
    pub fn stats(&self) -> CacheStats {
        self.inner.lock().stats
    }

    /// Return all cached (unused) memory to the device
    /// (`torch.cuda.empty_cache()`).
    pub fn empty_cache(&self) {
        let freed = self.inner.lock().drain_free_blocks();
        for (ptr, _size) in freed {
            // SAFETY: these are whole segments from device_alloc, no
            // longer referenced by any block.
            unsafe { self.inner.backend.device_free(ptr as *mut u8) };
        }
    }
}

// 'static: DataPtr deleters capture Arc<Inner<B>> and must be 'static.
impl<B: DeviceBackend + 'static> Allocator for CachingAllocator<B> {
    fn device(&self) -> Device {
        Device::Cuda(self.inner.device_index)
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        if nbytes == 0 {
            let layout = Layout::from_size_align(0, K_MIN_BLOCK_SIZE).unwrap();
            let dangling =
                NonNull::new(std::ptr::without_provenance_mut(K_MIN_BLOCK_SIZE)).unwrap();
            return DataPtr::new(dangling, layout);
        }

        let size = round_size(nbytes);
        let small = size <= K_SMALL_SIZE;

        let mut state = self.inner.lock();
        state.stats.num_alloc += 1;

        // 1. Best-fit reuse from the pool.
        let ptr = match state.get_free_block(size, small) {
            Some(ptr) => ptr,
            None => {
                // 2. Carve a new segment out of the device.
                let seg_size = segment_size(size);
                // SAFETY: seg_size > 0; failure returns null, handled below.
                let mut raw = unsafe { self.inner.backend.device_alloc(seg_size) };
                if raw.is_null() {
                    // Under pressure: return cached blocks to the driver
                    // and retry once (CUDACachingAllocator releases
                    // cached blocks on OOM before raising).
                    let freed = state.drain_free_blocks();
                    drop(state); // don't hold the lock across driver calls
                    for (ptr, _) in freed {
                        unsafe { self.inner.backend.device_free(ptr as *mut u8) };
                    }
                    // SAFETY: as above.
                    raw = unsafe { self.inner.backend.device_alloc(seg_size) };
                    if raw.is_null() {
                        panic!(
                            "CUDA out of memory: failed to allocate {seg_size} bytes \
                             on device {}",
                            self.inner.device_index
                        );
                    }
                    state = self.inner.lock();
                }
                state.stats.num_device_alloc += 1;
                state.stats.reserved_bytes += seg_size;
                state.stats.reserved_bytes_peak = state
                    .stats
                    .reserved_bytes_peak
                    .max(state.stats.reserved_bytes);
                let ptr = raw as usize;
                state.blocks.insert(
                    ptr,
                    Block {
                        size: seg_size,
                        requested: 0,
                        small_pool: small,
                        free: false,
                        segment: ptr,
                    },
                );
                ptr
            }
        };

        state.maybe_split(ptr, size);
        {
            let block = state.blocks.get_mut(&ptr).expect("block exists");
            block.requested = nbytes;
        }
        let block_size = state.blocks[&ptr].size;
        state.stats.allocated_bytes += block_size;
        state.stats.allocated_bytes_peak = state
            .stats
            .allocated_bytes_peak
            .max(state.stats.allocated_bytes);
        drop(state);

        // The deleter returns the block to the cache instead of freeing
        // it (c10::DataPtr's DeleterFnPtr calling CUDACachingAllocator::free).
        let inner = Arc::clone(&self.inner);
        let layout = Layout::from_size_align(block_size, K_MIN_BLOCK_SIZE).unwrap();
        DataPtr::with_deleter(
            NonNull::new(ptr as *mut u8).expect("device pointer is non-null"),
            layout,
            move |ptr| inner.free_block(ptr.as_ptr() as usize, block_size),
        )
    }
}

// ---------------------------------------------------------------------
// Real CUDA backend: cudaMalloc / cudaFree via the CUDA runtime API —
// the same calls PyTorch makes. Compiled only when build.rs found a
// `cudart` to link (feature `cuda` *and* a toolkit on the machine), so
// `cargo test --all-features` works on GPU-less hosts too.
// ---------------------------------------------------------------------

#[cfg(lumen_cuda_linked)]
mod ffi {
    use std::ffi::c_int;

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaSetDevice(device: c_int) -> c_int;
        pub fn cudaMalloc(ptr: *mut *mut u8, size: usize) -> c_int;
        pub fn cudaFree(ptr: *mut u8) -> c_int;
    }
}

/// Backend calling the CUDA runtime API (`cudaMalloc` / `cudaFree`).
#[cfg(lumen_cuda_linked)]
pub struct CudaBackend {
    device_index: i32,
}

#[cfg(lumen_cuda_linked)]
impl CudaBackend {
    pub fn new(device_index: usize) -> Self {
        CudaBackend {
            device_index: device_index as i32,
        }
    }
}

#[cfg(lumen_cuda_linked)]
impl DeviceBackend for CudaBackend {
    unsafe fn device_alloc(&self, nbytes: usize) -> *mut u8 {
        unsafe {
            ffi::cudaSetDevice(self.device_index);
            let mut ptr = std::ptr::null_mut();
            match ffi::cudaMalloc(&mut ptr, nbytes) {
                0 => ptr,
                _ => std::ptr::null_mut(), // OOM: let the cache release + retry
            }
        }
    }

    unsafe fn device_free(&self, ptr: *mut u8) {
        unsafe {
            ffi::cudaSetDevice(self.device_index);
            ffi::cudaFree(ptr);
        }
    }
}

/// The caching CUDA allocator (`CUDACachingAllocator::get()`).
#[cfg(lumen_cuda_linked)]
pub type CudaAllocator = CachingAllocator<CudaBackend>;

/// Global per-device caching allocators, created on first use
/// (`CUDACachingAllocator::allocator` is a similar process-wide array).
#[cfg(lumen_cuda_linked)]
pub fn get(device_index: usize) -> CudaAllocator {
    use std::collections::HashMap;
    use std::sync::OnceLock;

    static ALLOCATORS: OnceLock<Mutex<HashMap<usize, CudaAllocator>>> = OnceLock::new();
    ALLOCATORS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("allocator registry mutex poisoned")
        .entry(device_index)
        .or_insert_with(|| CachingAllocator::new(device_index, CudaBackend::new(device_index)))
        .clone()
}
