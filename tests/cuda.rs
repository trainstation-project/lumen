//! Tests for the CUDA caching allocator.
//!
//! No GPU needed: a mock `DeviceBackend` allocates real host memory (so
//! pointers are valid) while counting `cudaMalloc`/`cudaFree` calls and
//! enforcing an optional memory limit. This is how we verify PyTorch's
//! caching semantics — reuse, rounding, splitting, coalescing, stats —
//! on a CPU-only machine.

use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use lumen::allocator::cuda::{K_MIN_BLOCK_SIZE, K_SMALL_SIZE};
use lumen::{Allocator, CachingAllocator, Device, DeviceBackend};

/// Shared observation state: the allocator owns the backend, so tests
/// keep a clone of this handle to inspect it.
#[derive(Default)]
struct MockState {
    mallocs: AtomicUsize,
    frees: AtomicUsize,
    /// Sizes passed to `device_alloc` (the segment sizes).
    malloc_sizes: Mutex<Vec<usize>>,
    /// Live allocations, for correct deallocation.
    live: Mutex<HashMap<usize, Layout>>,
}

/// Pretends to be the GPU: `device_alloc` is host `malloc` with
/// accounting, `device_free` is `free`. `byte_limit` simulates VRAM size.
struct MockBackend {
    state: Arc<MockState>,
    byte_limit: Option<usize>,
}

impl DeviceBackend for MockBackend {
    unsafe fn device_alloc(&self, nbytes: usize) -> *mut u8 {
        if let Some(limit) = self.byte_limit {
            let live_bytes: usize = self
                .state
                .live
                .lock()
                .unwrap()
                .values()
                .map(|l| l.size())
                .sum();
            if live_bytes + nbytes > limit {
                return std::ptr::null_mut(); // pretend the GPU is full
            }
        }
        self.state.mallocs.fetch_add(1, Ordering::SeqCst);
        self.state.malloc_sizes.lock().unwrap().push(nbytes);
        // 256-byte alignment, like cudaMalloc guarantees.
        let layout = Layout::from_size_align(nbytes, 256).unwrap();
        let ptr = unsafe { alloc::alloc(layout) };
        if !ptr.is_null() {
            self.state.live.lock().unwrap().insert(ptr as usize, layout);
        }
        ptr
    }

    unsafe fn device_free(&self, ptr: *mut u8) {
        self.state.frees.fetch_add(1, Ordering::SeqCst);
        let layout = self
            .state
            .live
            .lock()
            .unwrap()
            .remove(&(ptr as usize))
            .expect("double free or unknown pointer");
        unsafe { alloc::dealloc(ptr, layout) };
    }
}

fn allocator(byte_limit: Option<usize>) -> (CachingAllocator<MockBackend>, Arc<MockState>) {
    let state = Arc::new(MockState::default());
    let backend = MockBackend {
        state: Arc::clone(&state),
        byte_limit,
    };
    (CachingAllocator::new(0, backend), state)
}

#[test]
fn reports_cuda_device() {
    let (alloc, _) = allocator(None);
    assert_eq!(alloc.device(), Device::Cuda(0));
    let (alloc7, _) = {
        let state = Arc::new(MockState::default());
        (
            CachingAllocator::new(
                7,
                MockBackend {
                    state: Arc::clone(&state),
                    byte_limit: None,
                },
            ),
            state,
        )
    };
    assert_eq!(alloc7.device(), Device::Cuda(7));
}

#[test]
fn dropped_blocks_are_reused_not_freed() {
    let (alloc, mock) = allocator(None);
    let a = alloc.allocate(512);
    let a_ptr = a.as_ptr();
    drop(a);

    // Block went back to the pool, NOT to the device.
    assert_eq!(mock.mallocs.load(Ordering::SeqCst), 1);
    assert_eq!(mock.frees.load(Ordering::SeqCst), 0);

    // Same-size request reuses the same block.
    let b = alloc.allocate(512);
    assert_eq!(b.as_ptr(), a_ptr);
    assert_eq!(mock.mallocs.load(Ordering::SeqCst), 1);
}

#[test]
fn sizes_round_up_to_512() {
    let (alloc, _) = allocator(None);
    let data = alloc.allocate(300);
    let stats = alloc.stats();
    assert_eq!(stats.allocated_bytes, K_MIN_BLOCK_SIZE);

    // A 500-byte request fits the rounded 512-byte block.
    let ptr = data.as_ptr();
    drop(data);
    let again = alloc.allocate(500);
    assert_eq!(again.as_ptr(), ptr);
}

#[test]
fn small_and_large_pools_use_different_segment_sizes() {
    let (alloc, mock) = allocator(None);
    let small = alloc.allocate(K_SMALL_SIZE); // ≤ 1 MiB → small pool, 2 MiB segment
    let large = alloc.allocate(K_SMALL_SIZE * 2); // → large pool, 20 MiB segment
    let sizes = mock.malloc_sizes.lock().unwrap().clone();
    assert_eq!(sizes, vec![2 * K_SMALL_SIZE, 20 * K_SMALL_SIZE]);
    drop(small);
    drop(large);
}

#[test]
fn segments_are_split_and_remainders_reused() {
    let (alloc, mock) = allocator(None);
    // First request carves 512 B out of a fresh 2 MiB segment...
    let a = alloc.allocate(512);
    // ...the second 512 B request is served from the same segment's
    // remainder — no new cudaMalloc.
    let b = alloc.allocate(512);
    assert_eq!(mock.mallocs.load(Ordering::SeqCst), 1);
    assert_eq!(b.as_ptr(), unsafe { a.as_ptr().add(512) });
    assert_eq!(alloc.stats().reserved_bytes, 2 * K_SMALL_SIZE);
}

#[test]
fn freed_blocks_coalesce_back_into_whole_segments() {
    let (alloc, mock) = allocator(None);
    let a = alloc.allocate(512);
    let b = alloc.allocate(512);
    drop(a);
    drop(b);

    // The two 512 B blocks plus the segment remainder must have merged
    // back into one whole 2 MiB segment: empty_cache then frees it with
    // a single device_free, and reserved bytes drop to zero.
    alloc.empty_cache();
    assert_eq!(mock.frees.load(Ordering::SeqCst), 1);
    assert_eq!(alloc.stats().reserved_bytes, 0);
}

#[test]
fn stats_track_peaks_and_counts() {
    let (alloc, mock) = allocator(None);
    let a = alloc.allocate(1024);
    let b = alloc.allocate(1024);
    let peak = alloc.stats();
    assert_eq!(peak.allocated_bytes, 2048);
    assert_eq!(peak.allocated_bytes_peak, 2048);
    assert_eq!(peak.num_alloc, 2);
    assert_eq!(peak.num_device_alloc, 1); // one 2 MiB segment
    assert_eq!(peak.num_free, 0);

    drop(a);
    drop(b);
    let after = alloc.stats();
    assert_eq!(after.allocated_bytes, 0);
    assert_eq!(after.allocated_bytes_peak, 2048); // peak survives
    assert_eq!(after.num_free, 2);
    assert_eq!(after.num_device_free, 0); // nothing returned to device
    assert_eq!(mock.frees.load(Ordering::SeqCst), 0);
}

#[test]
fn oom_releases_cached_blocks_and_retries() {
    // "VRAM" holds exactly one 2 MiB segment.
    let (alloc, mock) = allocator(Some(2 * K_SMALL_SIZE));
    let a = alloc.allocate(1024);
    drop(a); // cached, but still reserved on the "device"

    // A 3 MiB request needs a fresh 20 MiB segment: device_alloc fails,
    // the allocator releases the cached 2 MiB segment... still not
    // enough, so it panics — but the release must have happened.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        alloc.allocate(3 * K_SMALL_SIZE)
    }));
    assert!(result.is_err());
    assert!(mock.frees.load(Ordering::SeqCst) >= 1);
    assert_eq!(alloc.stats().reserved_bytes, 0);
}

#[test]
fn empty_cache_only_releases_unused_memory() {
    let (alloc, mock) = allocator(None);
    let live = alloc.allocate(1024); // stays alive across empty_cache
    let temp = alloc.allocate(1024);
    drop(temp);

    alloc.empty_cache();
    // The live block's segment can't be freed (it coalesced with the
    // temp block, but the live 1024 B piece keeps the segment busy), so
    // nothing whole is free... except nothing: the segment is split.
    // Only wholly-free segments are returned.
    assert!(alloc.stats().reserved_bytes > 0);
    let freed_before = mock.frees.load(Ordering::SeqCst);

    drop(live);
    alloc.empty_cache();
    assert_eq!(alloc.stats().reserved_bytes, 0);
    assert!(mock.frees.load(Ordering::SeqCst) > freed_before);
}

#[test]
fn zero_byte_allocation_is_safe() {
    let (alloc, mock) = allocator(None);
    let data = alloc.allocate(0);
    assert!(!data.as_ptr().is_null());
    drop(data);
    assert_eq!(mock.mallocs.load(Ordering::SeqCst), 0);
}

#[test]
fn buffer_is_writable_and_readable() {
    let (alloc, _) = allocator(None);
    let data = alloc.allocate(4096);
    // SAFETY: 4096 bytes were allocated above; data is alive.
    unsafe {
        std::ptr::write_bytes(data.as_ptr(), 0xCD, 4096);
        let bytes = std::slice::from_raw_parts(data.as_ptr(), 4096);
        assert!(bytes.iter().all(|&b| b == 0xCD));
    }
}
