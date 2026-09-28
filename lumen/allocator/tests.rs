//! Unit tests for the allocators, one module per former test file. They
//! live in the crate (not tests/) so they can reach private items.
//!
//! The `cuda` and `mps` modules hold the per-device tests the Makefile's
//! `test-cuda` / `test-mps` select by path.

mod cpu {
    use crate::{Allocator, CpuAllocator, DataPtr, Device};

    #[test]
    fn allocator_reports_cpu_device() {
        assert_eq!(CpuAllocator.device(), Device::Cpu);
        assert_eq!(CpuAllocator::get().device(), Device::Cpu);
    }

    #[test]
    fn get_returns_the_same_global_instance() {
        let a = CpuAllocator::get();
        let b = CpuAllocator::get();
        // Both references point at the one static CPU_ALLOCATOR. Compare data
        // addresses only: vtable pointers of `&dyn` aren't guaranteed unique.
        assert!(std::ptr::addr_eq(a, b));
    }

    #[test]
    fn allocation_is_non_null_and_64_byte_aligned() {
        let data = CpuAllocator::get().allocate(256);
        let ptr = data.as_ptr();
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 64, 0, "pointer must be 64-byte aligned");
    }

    #[test]
    fn allocation_initialization() {
        let data = CpuAllocator::get().allocate(256);
        unsafe {
            std::ptr::write_bytes(data.as_ptr(), 0xAB, 256);
            let bytes = std::slice::from_raw_parts(data.as_ptr(), 256);
            assert!(bytes.iter().all(|&b| b == 0xAB));
        }
    }

    #[test]
    fn can_write_and_read_back_through_the_pointer() {
        let data = CpuAllocator::get().allocate(64);
        // SAFETY: 64 bytes were allocated above; we stay in bounds and
        // `data` outlives the slice.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(data.as_ptr(), 64);
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = i as u8;
            }
            assert_eq!(bytes[0], 0);
            assert_eq!(bytes[63], 63);
        }
    }

    #[test]
    fn zero_byte_allocation_is_safe() {
        let data = CpuAllocator::get().allocate(0);
        // Dangling but aligned; must not be dereferenced, must drop cleanly.
        assert_eq!(data.as_ptr() as usize % 64, 0);
        drop(data);
    }

    #[test]
    fn repeated_alloc_and_drop_cycles_do_not_crash() {
        // Exercises the Drop impl (dealloc with the matching layout).
        for i in 0..100 {
            let data = CpuAllocator::get().allocate(1 << (i % 12));
            assert!(!data.as_ptr().is_null());
            drop(data);
        }
    }

    #[test]
    fn data_ptr_is_send_and_sync() {
        // Compile-time proof of the unsafe Send/Sync impls.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DataPtr>();

        // And a runtime check: a DataPtr can cross a thread boundary.
        let data = CpuAllocator::get().allocate(16);
        std::thread::spawn(move || {
            assert!(!data.as_ptr().is_null());
        })
        .join()
        .unwrap();
    }
}

mod caching {
    use std::alloc::Layout;
    use std::ptr::NonNull;
    use std::sync::{Arc, Mutex};

    use crate::allocator::mps::{K_SMALL_HEAP, K_XLARGE_HEAP};
    use crate::{Allocator, CachePolicy, CachingAllocator, CudaPolicy, DataPtr, Device, MpsPolicy};

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
}

mod config {
    //! Switching `config::memory_caching` off and on. Flipped with the per-thread
    //! test override: the global flag is shared by every test in this process
    //! (its setter is exercised from Python, tests/test_config.py).

    use std::alloc::Layout;
    use std::ptr::NonNull;
    use std::sync::{Arc, Mutex};

    use crate::allocator::config;
    use crate::{Allocator, CachingAllocator, CudaPolicy, DataPtr, Device};

    #[derive(Default)]
    struct Log {
        allocs: Vec<usize>,
        frees: usize,
    }

    /// Returns fake, never-dereferenced addresses and logs every call.
    struct FakeBackend(Arc<Mutex<Log>>);

    impl Allocator for FakeBackend {
        fn device(&self) -> Device {
            Device::Cuda(0)
        }

        fn allocate(&self, nbytes: usize) -> DataPtr {
            let addr = {
                let mut log = self.0.lock().unwrap();
                log.allocs.push(nbytes);
                (1 << 32) + log.allocs.len() * (1 << 30)
            };
            let log = Arc::clone(&self.0);
            DataPtr::with_deleter(
                NonNull::new(std::ptr::without_provenance_mut(addr)).unwrap(),
                Layout::from_size_align(nbytes, 1).unwrap(),
                move |_| log.lock().unwrap().frees += 1,
            )
        }

        fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
            Some(self.allocate(nbytes))
        }
    }

    #[test]
    fn memory_caching_can_be_switched_off_and_on() {
        let log = Arc::new(Mutex::new(Log::default()));
        let alloc = CachingAllocator::new(FakeBackend(Arc::clone(&log)), CudaPolicy);
        assert!(config::memory_caching(), "caching is on by default");

        // Cached: one 2 MiB segment, rounded blocks, reuse, nothing freed.
        let cached = alloc.allocate(100);
        drop(alloc.allocate(100));
        assert_eq!(log.lock().unwrap().allocs, vec![2 << 20]);
        assert_eq!(alloc.stats().allocated_bytes, 512);

        // Off: exact sizes straight to the backend, freed on drop, not counted.
        config::with_memory_caching(false, || {
            let a = alloc.allocate(100);
            let b = alloc.allocate(3 << 20);
            assert_eq!(log.lock().unwrap().allocs, vec![2 << 20, 100, 3 << 20]);
            assert_eq!(alloc.stats().allocated_bytes, 512, "only the cached block");
            drop(a);
            drop(b);
            assert_eq!(log.lock().unwrap().frees, 2);
            assert_eq!(alloc.allocate(0).as_ptr(), NonNull::dangling().as_ptr());
        });

        // A block allocated while caching was on still returns to the cache.
        drop(cached);
        assert_eq!(alloc.stats().allocated_bytes, 0);
        assert_eq!(log.lock().unwrap().frees, 2, "the segment stays cached");

        // Back on: the cached segment is reused.
        assert!(config::memory_caching());
        drop(alloc.allocate(100));
        assert_eq!(log.lock().unwrap().allocs.len(), 3);

        drop(alloc); // last reference: the cached segment goes back to the device
        assert_eq!(log.lock().unwrap().frees, 3);
    }
}

mod devices {
    //! Device selection for tensors and storage, on any machine.

    use crate::allocator::{allocator_for, cuda, mps};
    use crate::{Device, Storage, Tensor};

    #[test]
    fn cpu_is_always_available() {
        assert_eq!(allocator_for(Device::Cpu).unwrap().device(), Device::Cpu);
        let t = Tensor::zeros(&[2], Device::Cpu);
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(Storage::new(8, Device::Cpu).device(), Device::Cpu);
    }

    #[test]
    fn unavailable_devices_are_reported() {
        if !cuda::is_available() {
            let err = allocator_for(Device::Cuda(0)).err().unwrap();
            assert!(err.contains("CUDA device 0 is not available"), "{err}");
        }
        let past_last = Device::Cuda(cuda::device_count());
        assert!(allocator_for(past_last).is_err());
        if !mps::is_available() {
            assert!(allocator_for(Device::Mps).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "is not available")]
    fn tensor_on_unavailable_device_panics() {
        Tensor::zeros(&[1], Device::Cuda(cuda::device_count()));
    }
}

mod cuda {
    use std::alloc::{self, Layout};
    use std::collections::HashMap;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::allocator::cuda::K_MIN_BLOCK_SIZE;
    use crate::allocator::traits::K_SMALL_SIZE;
    use crate::{Allocator, CachingAllocator, CudaPolicy, DataPtr, Device};

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

    #[cfg(lumen_cuda_linked)] // the real backend exists only when cudart is linked
    mod device {
        //! The real CUDA backend on a GPU (the tests above use mocks). Each
        //! test skips when no device is visible. Tests that assert on stats
        //! use a fresh allocator, not the global `cuda::get`, so parallel
        //! tests can't perturb each other's counters.

        use crate::allocator::allocator_for;
        use crate::allocator::cuda::{self, CudaBackend};
        use crate::{Allocator, CachingAllocator, CudaPolicy, Device};

        /// Skip guard: returns early from a test when there is no GPU.
        macro_rules! require_cuda {
            () => {
                if !cuda::is_available() {
                    eprintln!("no CUDA device, skipping");
                    return;
                }
            };
        }

        /// An allocator on device 0 with its own private cache.
        fn fresh() -> cuda::CudaAllocator {
            CachingAllocator::new(CudaBackend::new(0), CudaPolicy)
        }

        #[test]
        fn availability_matches_the_device_count() {
            assert_eq!(cuda::is_available(), cuda::device_count() > 0);
            assert!(allocator_for(Device::Cuda(cuda::device_count())).is_err());
        }

        #[test]
        fn reports_cuda_devices() {
            require_cuda!();
            assert_eq!(fresh().device(), Device::Cuda(0));
            let last = cuda::device_count() - 1;
            assert_eq!(cuda::get(last).device(), Device::Cuda(last));
            assert_eq!(
                allocator_for(Device::Cuda(last)).unwrap().device(),
                Device::Cuda(last)
            );
        }

        #[test]
        fn freed_blocks_are_reused_without_another_cuda_malloc() {
            require_cuda!();
            let alloc = fresh();
            {
                let _a = alloc.allocate(2048);
            }
            let _b = alloc.allocate(2048);
            assert_eq!(alloc.stats().num_device_alloc, 1);
        }

        #[test]
        fn empty_cache_returns_segments_with_cuda_free() {
            require_cuda!();
            let alloc = fresh();
            drop(alloc.allocate(1 << 20));
            assert!(alloc.stats().reserved_bytes > 0);
            alloc.empty_cache();
            let stats = alloc.stats();
            assert_eq!(stats.reserved_bytes, 0);
            assert_eq!(stats.num_device_free, stats.num_device_alloc);
        }

        #[test]
        fn get_returns_the_same_global_instance() {
            require_cuda!();
            let (a, b) = (cuda::get(0), cuda::get(0));
            // Other tests put small tensors on the global allocator in
            // parallel; a large-pool size keeps them from taking the block.
            let size = 5 << 20;
            let p = a.allocate(size);
            let ptr = p.as_ptr();
            drop(p);
            assert_eq!(
                b.allocate(size).as_ptr(),
                ptr,
                "expected block reuse from the shared cache"
            );
        }
    }
}

#[cfg(lumen_mps_linked)] // shim-backed types exist only when Metal is linked
mod mps {
    //! Tests for the MPS (Apple Silicon / Metal) caching allocator.
    //!
    //! Unlike the CUDA tests (which use a mock backend — CI has no NVIDIA GPU),
    //! these run against the *real* Metal device: this backend compiles and runs
    //! on any Apple Silicon Mac. Every test skips silently when Metal is
    //! unavailable (non-macOS build, or `mps` feature off).
    //!
    //! Tests that assert on stats use fresh allocator instances rather than the
    //! global `mps::get()`, so parallel test threads can't perturb each other's
    //! counters.

    use crate::allocator::mps;
    use crate::{Allocator, CachingAllocator, Device, MpsPolicy};

    /// Skip guard: returns early from a test when Metal is unavailable.
    macro_rules! require_mps {
        () => {
            if !mps::is_available() {
                eprintln!("Metal unavailable, skipping");
                return;
            }
        };
    }

    /// An allocator with its own private cache (not the global one).
    fn fresh() -> mps::MpsAllocator {
        CachingAllocator::new(mps::MpsBackend, MpsPolicy::from_device())
    }

    #[test]
    fn reports_mps_device() {
        require_mps!();
        assert_eq!(mps::get().device(), Device::Mps);
    }

    #[test]
    fn get_returns_the_same_global_instance() {
        require_mps!();
        let a = mps::get();
        let b = mps::get();
        // Clones share one cache: an allocation freed through `a` is reusable by `b`.
        // Other tests put tensors on the global allocator in parallel, but only
        // small ones; a large-pool size keeps them from taking the freed block.
        let size = 5 << 20;
        let p = a.allocate(size);
        let ptr = p.as_ptr();
        drop(p);
        let q = b.allocate(size);
        assert_eq!(
            q.as_ptr(),
            ptr,
            "expected block reuse from the shared cache"
        );
    }

    #[test]
    fn unified_memory_is_host_writable() {
        require_mps!();
        // Shared-mode MTLBuffers are ordinary CPU memory on Apple Silicon:
        // write through the pointer and read it back.
        let data = fresh().allocate(1024);
        let slice = unsafe { std::slice::from_raw_parts_mut(data.as_ptr(), 1024) };
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        assert!(slice.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8));
    }

    #[test]
    fn small_alloc_uses_mps_segment_size() {
        require_mps!();
        let alloc = fresh();
        let _data = alloc.allocate(512);
        // MPS tuning: small requests carve an 8 MiB segment (kSmallHeap),
        // not CUDA's 2 MiB.
        assert_eq!(alloc.stats().reserved_bytes, 8 << 20);
    }

    #[test]
    fn freed_block_is_reused_without_new_device_alloc() {
        require_mps!();
        let alloc = fresh();
        {
            let _a = alloc.allocate(2048);
            let _b = alloc.allocate(2048); // served from the same segment's remainder
        }
        assert_eq!(alloc.stats().num_device_alloc, 1);
    }

    #[test]
    fn empty_cache_releases_segments_to_metal() {
        require_mps!();
        let alloc = fresh();
        {
            let _data = alloc.allocate(1 << 20);
        }
        assert!(alloc.stats().reserved_bytes > 0);
        alloc.empty_cache();
        let stats = alloc.stats();
        assert_eq!(stats.reserved_bytes, 0);
        assert_eq!(stats.num_device_free, stats.num_device_alloc);
    }

    #[test]
    fn stats_track_allocated_and_peak() {
        require_mps!();
        let alloc = fresh();
        let a = alloc.allocate(1000); // rounds to 1024
        let b = alloc.allocate(1000);
        assert_eq!(alloc.stats().allocated_bytes, 2048);
        drop(a);
        assert_eq!(alloc.stats().allocated_bytes, 1024);
        assert_eq!(alloc.stats().allocated_bytes_peak, 2048);
        drop(b);
    }

    #[test]
    fn limits_come_from_metal() {
        require_mps!();
        let limits = MpsPolicy::from_device();
        assert!(limits.alignment.is_power_of_two());
        assert!(limits.page_size.is_power_of_two());
        assert!(limits.max_buffer_size > 1 << 30);
        assert!(limits.low_watermark_limit > limits.max_buffer_size / 2);
        eprintln!("{limits:?}");
    }

    #[test]
    fn small_alloc_rounds_to_metal_alignment() {
        require_mps!();
        let alloc = fresh();
        let _data = alloc.allocate(1);
        assert_eq!(
            alloc.stats().allocated_bytes,
            MpsPolicy::from_device().alignment
        );
    }
}
