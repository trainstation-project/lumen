use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use lumen::allocator::cuda::K_MIN_BLOCK_SIZE;
use lumen::allocator::traits::K_SMALL_SIZE;
use lumen::{Allocator, CachingAllocator, CudaPolicy, DataPtr, Device};

/// Shared observation state: the allocator owns the backend, so tests
/// keep a clone of this handle to inspect it.
#[derive(Default)]
struct MockState {
    mallocs: AtomicUsize,
    frees: AtomicUsize,
    /// Sizes passed to `device_alloc` (the segment sizes).
    malloc_sizes: Mutex<Vec<usize>>,
    /// Live allocations by address, for correct deallocation.
    live: Mutex<HashMap<usize, LiveAlloc>>,
}

/// A live host allocation. Keeps the pointer itself (not just its
/// address) so it can be freed without an integer-to-pointer cast,
/// which Miri's strict-provenance mode rejects.
struct LiveAlloc {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: plain owned host memory; access is serialized by the `Mutex`.
unsafe impl Send for LiveAlloc {}

struct MockBackend {
    state: Arc<MockState>,
    byte_limit: Option<usize>,
    device_index: usize,
}

impl Allocator for MockBackend {
    fn device(&self) -> Device {
        Device::Cuda(self.device_index)
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        self.try_allocate(nbytes).expect("mock device OOM")
    }

    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        if let Some(limit) = self.byte_limit {
            let live_bytes: usize = self
                .state
                .live
                .lock()
                .unwrap()
                .values()
                .map(|l| l.layout.size())
                .sum();
            if live_bytes + nbytes > limit {
                return None; // pretend the GPU is full
            }
        }
        self.state.mallocs.fetch_add(1, Ordering::SeqCst);
        self.state.malloc_sizes.lock().unwrap().push(nbytes);
        // 256-byte alignment, like cudaMalloc guarantees.
        let layout = Layout::from_size_align(nbytes, 256).unwrap();
        let ptr = NonNull::new(unsafe { alloc::alloc(layout) })?;
        self.state.live.lock().unwrap().insert(
            ptr.as_ptr().addr(),
            LiveAlloc {
                ptr: ptr.as_ptr(),
                layout,
            },
        );
        let state = Arc::clone(&self.state);
        Some(DataPtr::with_deleter(ptr, layout, move |p| {
            state.frees.fetch_add(1, Ordering::SeqCst);
            let LiveAlloc { layout, .. } = state
                .live
                .lock()
                .unwrap()
                .remove(&p.as_ptr().addr())
                .expect("double free or unknown pointer");
            // SAFETY: allocated above with this layout.
            unsafe { alloc::dealloc(p.as_ptr(), layout) };
        }))
    }
}

impl Drop for MockBackend {
    // The cache never returns segments on its own (like PyTorch), so
    // free whatever is still reserved; otherwise Miri reports leaks.
    fn drop(&mut self) {
        for (_, LiveAlloc { ptr, layout }) in self.state.live.lock().unwrap().drain() {
            // SAFETY: allocated in device_alloc with this layout; the
            // allocator (and every DataPtr into it) is gone.
            unsafe { alloc::dealloc(ptr, layout) };
        }
    }
}

fn allocator(
    byte_limit: Option<usize>,
) -> (CachingAllocator<MockBackend, CudaPolicy>, Arc<MockState>) {
    let state = Arc::new(MockState::default());
    let backend = MockBackend {
        state: Arc::clone(&state),
        byte_limit,
        device_index: 0,
    };
    (CachingAllocator::new(backend, CudaPolicy), state)
}

#[test]
fn reports_cuda_device() {
    let (alloc, _) = allocator(None);
    assert_eq!(alloc.device(), Device::Cuda(0));
    let alloc7 = CachingAllocator::new(
        MockBackend {
            state: Arc::new(MockState::default()),
            byte_limit: None,
            device_index: 7,
        },
        CudaPolicy,
    );
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

/// `(ptr, size)` of every `device_free`, in call order.
type FreedLog = Arc<Mutex<Vec<(usize, usize)>>>;

/// Bump-allocates segments back to back out of one host arena, so
/// consecutive `try_allocate`s are address-adjacent (real `cudaMalloc`
/// can do this too). The `DataPtr` deleter only records the pointer.
struct ArenaBackend {
    base: *mut u8,
    layout: Layout,
    next: AtomicUsize,
    freed: FreedLog,
    sizes: Arc<Mutex<HashMap<usize, usize>>>,
}

// SAFETY: the arena pointer is only used for address arithmetic and is
// owned exclusively by this backend.
unsafe impl Send for ArenaBackend {}
unsafe impl Sync for ArenaBackend {}

impl ArenaBackend {
    fn new(nbytes: usize) -> (Self, FreedLog) {
        let layout = Layout::from_size_align(nbytes, 512).unwrap();
        // SAFETY: nonzero size.
        let base = unsafe { alloc::alloc(layout) };
        assert!(!base.is_null());
        let freed = Arc::new(Mutex::new(Vec::new()));
        let backend = ArenaBackend {
            base,
            layout,
            next: AtomicUsize::new(0),
            freed: Arc::clone(&freed),
            sizes: Arc::new(Mutex::new(HashMap::new())),
        };
        (backend, freed)
    }
}

impl Drop for ArenaBackend {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { alloc::dealloc(self.base, self.layout) };
    }
}

impl Allocator for ArenaBackend {
    fn device(&self) -> Device {
        Device::Cuda(0)
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        self.try_allocate(nbytes).expect("arena exhausted")
    }

    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        let offset = self.next.fetch_add(nbytes, Ordering::SeqCst);
        if offset + nbytes > self.layout.size() {
            return None;
        }
        // SAFETY: offset + nbytes <= arena size; base is 512-aligned.
        let ptr = NonNull::new(unsafe { self.base.add(offset) })?;
        self.sizes
            .lock()
            .unwrap()
            .insert(ptr.as_ptr().addr(), nbytes);
        let sizes = Arc::clone(&self.sizes);
        let freed = Arc::clone(&self.freed);
        Some(DataPtr::with_deleter(
            ptr,
            Layout::from_size_align(nbytes, 512).unwrap(),
            move |p| {
                let size = sizes
                    .lock()
                    .unwrap()
                    .remove(&p.as_ptr().addr())
                    .expect("double free or unknown pointer");
                freed.lock().unwrap().push((p.as_ptr().addr(), size));
            },
        ))
    }
}

#[test]
fn adjacent_segments_never_coalesce() {
    let (backend, freed) = ArenaBackend::new(8 * K_SMALL_SIZE);
    let base = backend.base.addr();
    let alloc = CachingAllocator::new(backend, CudaPolicy);

    // a and b fill segment 1 exactly; c starts segment 2, which the
    // arena places right after segment 1.
    let a = alloc.allocate(K_SMALL_SIZE);
    let b = alloc.allocate(K_SMALL_SIZE);
    let c = alloc.allocate(K_SMALL_SIZE);
    assert_eq!(c.as_ptr().addr(), base + 2 * K_SMALL_SIZE);
    assert_eq!(alloc.stats().reserved_bytes, 4 * K_SMALL_SIZE);

    // b (end of segment 1) and c (start of segment 2) are free and
    // address-adjacent, but must not merge across the boundary.
    drop(b);
    drop(c);
    drop(a);

    alloc.empty_cache();
    let mut freed = freed.lock().unwrap().clone();
    freed.sort();
    assert_eq!(
        freed,
        vec![
            (base, 2 * K_SMALL_SIZE),
            (base + 2 * K_SMALL_SIZE, 2 * K_SMALL_SIZE)
        ]
    );
    assert_eq!(alloc.stats().reserved_bytes, 0);
    assert_eq!(alloc.stats().num_device_free, 2);
}

#[test]
fn whole_segment_next_to_busy_segment_is_released() {
    let (backend, freed) = ArenaBackend::new(8 * K_SMALL_SIZE);
    let base = backend.base.addr();
    let alloc = CachingAllocator::new(backend, CudaPolicy);

    // a and b keep segment 1 busy; c's segment 2 sits right after it.
    let a = alloc.allocate(K_SMALL_SIZE);
    let b = alloc.allocate(K_SMALL_SIZE);
    let c = alloc.allocate(K_SMALL_SIZE);
    drop(c);

    // Segment 2 is wholly free; its address neighbor b belongs to another
    // segment, so it must not count as a split piece.
    alloc.empty_cache();
    assert_eq!(
        freed.lock().unwrap().clone(),
        vec![(base + 2 * K_SMALL_SIZE, 2 * K_SMALL_SIZE)]
    );
    assert_eq!(alloc.stats().reserved_bytes, 2 * K_SMALL_SIZE);
    drop(a);
    drop(b);
}
