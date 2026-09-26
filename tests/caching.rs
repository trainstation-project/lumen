use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use lumen::allocator::mps::{K_SMALL_HEAP, K_XLARGE_HEAP};
use lumen::{Allocator, CachePolicy, CachingAllocator, CudaPolicy, DataPtr, Device, MpsPolicy};

const MIB: usize = 1 << 20;

#[derive(Default)]
struct Log {
    /// Sizes passed to `device_alloc`, in call order.
    allocs: Vec<usize>,
    /// Addresses passed to `device_free`, in call order.
    frees: Vec<usize>,
    next_addr: usize,
}

/// Returns fake addresses with a gap between segments; never touches memory.
struct FakeBackend(Arc<Mutex<Log>>);

impl Allocator for FakeBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        self.try_allocate(nbytes).expect("fake backend OOM")
    }

    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        let addr = {
            let mut log = self.0.lock().unwrap();
            if log.next_addr == 0 {
                log.next_addr = 1 << 32;
            }
            let addr = log.next_addr;
            log.next_addr += nbytes + MIB;
            log.allocs.push(nbytes);
            addr
        };
        let ptr = NonNull::new(std::ptr::without_provenance_mut(addr))?;
        let handle = Arc::clone(&self.0);
        Some(DataPtr::with_deleter(
            ptr,
            Layout::from_size_align(nbytes, 1).unwrap(),
            move |p| {
                handle.lock().unwrap().frees.push(p.as_ptr().addr());
            },
        ))
    }
}

type Fake<P> = CachingAllocator<FakeBackend, P>;

fn allocator<P: CachePolicy>(policy: P) -> (Fake<P>, Arc<Mutex<Log>>) {
    let log = Arc::new(Mutex::new(Log::default()));
    let backend = FakeBackend(Arc::clone(&log));
    (CachingAllocator::new(backend, policy), log)
}

/// Apple Silicon's values (256 B alignment, 16 KiB pages), no memory pressure.
const MPS: MpsPolicy = MpsPolicy {
    alignment: 256,
    page_size: 16 << 10,
    max_buffer_size: 1 << 40,
    low_watermark_limit: usize::MAX,
};

fn mps() -> (Fake<MpsPolicy>, Arc<Mutex<Log>>) {
    allocator(MPS)
}

fn allocs(log: &Mutex<Log>) -> Vec<usize> {
    log.lock().unwrap().allocs.clone()
}

/// Bytes the allocator reports for a single live allocation of `nbytes`.
fn block_size<P: CachePolicy>(alloc: &Fake<P>, nbytes: usize) -> usize {
    let before = alloc.stats().allocated_bytes;
    let data: DataPtr = alloc.allocate(nbytes);
    let size = alloc.stats().allocated_bytes - before;
    drop(data);
    size
}

// ---------------- shared bookkeeping ----------------

#[test]
fn unsplit_reused_block_is_counted_whole() {
    let (alloc, log) = allocator(CudaPolicy);
    drop(alloc.allocate(3 * MIB)); // one 20 MiB segment, coalesced free again
    // Leaves 0.5 MiB, below the large-pool split threshold: the whole 20 MiB
    // block is handed out, and must be counted (and uncounted) as such.
    let big = alloc.allocate(19 * MIB + MIB / 2);
    assert_eq!(alloc.stats().allocated_bytes, 20 * MIB);
    drop(big);
    assert_eq!(alloc.stats().allocated_bytes, 0);
    assert_eq!(allocs(&log), vec![20 * MIB]);
}

#[test]
fn allocator_dropped_before_its_pointers_still_frees_segments() {
    let (alloc, log) = allocator(CudaPolicy);
    let data = alloc.allocate(512);
    drop(alloc);
    assert!(log.lock().unwrap().frees.is_empty(), "segment still in use");
    drop(data); // last reference: the cache flushes
    assert_eq!(log.lock().unwrap().frees.len(), 1);
}

#[test]
fn last_clone_dropped_frees_cached_segments() {
    let (alloc, log) = allocator(CudaPolicy);
    let other = alloc.clone();
    drop(alloc.allocate(512));
    drop(alloc);
    assert!(log.lock().unwrap().frees.is_empty());
    drop(other);
    assert_eq!(log.lock().unwrap().frees.len(), 1);
}

// ---------------- CUDA ----------------

#[test]
fn cuda_new_segment_remainder_below_threshold_is_not_split() {
    let (alloc, log) = allocator(CudaPolicy);
    // 11.5 MiB gets a 12 MiB segment; the 0.5 MiB left is not > 1 MiB, so
    // c10 hands out the whole segment instead of stranding a sliver.
    let data = alloc.allocate(11 * MIB + MIB / 2);
    assert_eq!(allocs(&log), vec![12 * MIB]);
    assert_eq!(alloc.stats().allocated_bytes, 12 * MIB);
    drop(data);
}

#[test]
fn cuda_large_remainder_of_exactly_1_mib_is_not_split() {
    let (alloc, _) = allocator(CudaPolicy);
    // 11 MiB request, 12 MiB segment: remainder 1 MiB is not > kSmallSize.
    assert_eq!(block_size(&alloc, 11 * MIB), 12 * MIB);
}

// ---------------- MPS ----------------

#[test]
fn mps_small_requests_round_to_metal_alignment() {
    let (alloc, log) = mps();
    assert_eq!(block_size(&alloc, 1), 256);
    assert_eq!(block_size(&alloc, 300), 512);
    assert_eq!(allocs(&log), vec![K_SMALL_HEAP]);
}

#[test]
fn mps_small_blocks_split_at_alignment() {
    let (alloc, _) = mps();
    let a = alloc.allocate(256);
    let b = alloc.allocate(256);
    assert_eq!(b.as_ptr().addr(), a.as_ptr().addr() + 256);
}

#[test]
fn mps_large_requests_round_into_buckets() {
    let (alloc, _) = mps();
    // 3 MiB + 1 aligns to 3 MiB + 256; the granule for [2, 4) MiB is
    // 2 MiB / 32 = 64 KiB.
    assert_eq!(block_size(&alloc, 3 * MIB + 1), 3 * MIB + (64 << 10));
    // [1, 2) MiB: granule 1 MiB / 32 = 32 KiB.
    assert_eq!(block_size(&alloc, MIB + 1), MIB + (32 << 10));
}

#[test]
fn mps_bucket_rounding_never_crosses_a_heap_tier() {
    let (alloc, log) = mps();
    // Bucketing 10 MiB - 4 KiB would give exactly 10 MiB, the XLARGE tier,
    // so the aligned size is kept and it stays in a 32 MiB heap.
    assert_eq!(block_size(&alloc, 10 * MIB - 4096), 10 * MIB - 4096);
    assert_eq!(allocs(&log), vec![32 * MIB]);
}

/// Segment size a fresh MPS allocator requests for `nbytes`.
fn mps_heap_for(nbytes: usize) -> usize {
    let (alloc, log) = mps();
    drop(alloc.allocate(nbytes));
    allocs(&log)[0]
}

#[test]
fn mps_heap_tiers() {
    assert_eq!(mps_heap_for(MIB), K_SMALL_HEAP); // SMALL
    assert_eq!(mps_heap_for(MIB + 1), 32 * MIB); // LARGE
    assert_eq!(mps_heap_for(10 * MIB), K_XLARGE_HEAP); // XLARGE
    assert_eq!(mps_heap_for(512 * MIB - MIB), K_XLARGE_HEAP);
    // OVERSIZE from 512 MiB: bucketed (16 MiB granule), then 2 MiB-rounded.
    assert_eq!(mps_heap_for(512 * MIB), 512 * MIB);
    assert_eq!(mps_heap_for(600 * MIB), 608 * MIB);
}

#[test]
fn mps_memory_pressure_skips_the_xlarge_tier() {
    let (alloc, log) = allocator(MpsPolicy {
        low_watermark_limit: 0,
        ..MPS
    });
    let _small = alloc.allocate(512); // small pool ignores pressure
    let _xl = alloc.allocate(10 * MIB);
    assert_eq!(allocs(&log), vec![K_SMALL_HEAP, 10 * MIB]);
}

#[test]
fn mps_pressure_starts_within_1_mib_of_the_low_watermark() {
    // 8 MiB already reserved; limit 9 MiB: 8 + 1 > 9 is false -> no pressure.
    let (alloc, log) = allocator(MpsPolicy {
        low_watermark_limit: 9 * MIB,
        ..MPS
    });
    let _small = alloc.allocate(512);
    let _xl = alloc.allocate(10 * MIB);
    assert_eq!(allocs(&log), vec![K_SMALL_HEAP, K_XLARGE_HEAP]);

    // Limit 9 MiB - 1: 8 MiB + 1 MiB > limit -> pressure.
    let (alloc, log) = allocator(MpsPolicy {
        low_watermark_limit: 9 * MIB - 1,
        ..MPS
    });
    let _small = alloc.allocate(512);
    let _xl = alloc.allocate(10 * MIB);
    assert_eq!(allocs(&log), vec![K_SMALL_HEAP, 10 * MIB]);
}

#[test]
fn mps_large_remainder_of_exactly_1_mib_is_split() {
    let (alloc, log) = mps();
    drop(alloc.allocate(2 * MIB)); // one 32 MiB heap, free again
    // Remainder 1 MiB >= min_split (kMaxSmallAlloc): split, unlike CUDA.
    let data = alloc.allocate(31 * MIB);
    assert_eq!(alloc.stats().allocated_bytes, 31 * MIB);
    assert_eq!(allocs(&log), vec![32 * MIB]);
    drop(data);
}

#[test]
fn mps_grown_request_reuses_its_bucket() {
    let (alloc, _) = mps();
    // Both land in the same 64 KiB bucket, so a block freed by the first
    // fits the second exactly.
    assert_eq!(
        block_size(&alloc, 3 * MIB + 1),
        block_size(&alloc, 3 * MIB + 5000)
    );
}

#[test]
#[should_panic(expected = "Invalid buffer size")]
fn mps_rejects_requests_at_max_buffer_length() {
    let (alloc, _) = allocator(MpsPolicy {
        max_buffer_size: 64 * MIB,
        ..MPS
    });
    alloc.allocate(64 * MIB);
}
