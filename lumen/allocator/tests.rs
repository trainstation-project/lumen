//! Unit tests for the allocators, one module per former test file. They
//! live in the crate (not tests/) so they can reach private items.
//!
//! The `cuda` and `mps` modules hold the per-device tests the Makefile's
//! `test-cuda` / `test-mps` select by path.

mod cpu {
    use crate::{Allocator, CpuAllocator};

    #[test]
    fn get_returns_the_same_global_instance() {
        let a = CpuAllocator::get();
        let b = CpuAllocator::get();
        // Both references point at the one static CPU_ALLOCATOR. Compare data
        // addresses only: vtable pointers of `&dyn` aren't guaranteed unique.
        assert!(std::ptr::addr_eq(a, b));
    }

    #[test]
    fn allocation_is_64_byte_aligned() {
        let data = CpuAllocator.allocate(256);
        assert_eq!(
            data.as_ptr() as usize % 64,
            0,
            "pointer must be 64-byte aligned"
        );
        unsafe { CpuAllocator.deallocate(data, 256) };
    }

    #[test]
    fn can_write_and_read_back_through_the_pointer() {
        let data = CpuAllocator.allocate(64);
        // SAFETY: 64 bytes were allocated above; we stay in bounds and free
        // them after.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(data.as_ptr(), 64);
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = i as u8;
            }
            assert_eq!(bytes[0], 0);
            assert_eq!(bytes[63], 63);
            CpuAllocator.deallocate(data, 64);
        }
    }

    #[test]
    fn zero_byte_allocation_is_safe() {
        let data = CpuAllocator.allocate(0);
        // Dangling but aligned; must not be dereferenced, must free cleanly.
        assert_eq!(data.as_ptr() as usize % 64, 0);
        unsafe { CpuAllocator.deallocate(data, 0) };
    }

    #[test]
    fn repeated_alloc_and_free_cycles_do_not_crash() {
        // Exercises deallocate (dealloc with the matching layout).
        for i in 0..100 {
            let nbytes = 1 << (i % 12);
            let data = CpuAllocator.allocate(nbytes);
            unsafe { CpuAllocator.deallocate(data, nbytes) };
        }
    }
}

mod static_allocator {
    //! The static allocator, over host memory posing as a device.

    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::allocator::static_allocator::StaticAllocator;
    use crate::{Allocator, CpuAllocator, DataPtr, Device};

    /// Host memory posing as a CUDA device, counting reservations and
    /// releases.
    struct Host {
        reservations: Arc<AtomicUsize>,
        releases: Arc<AtomicUsize>,
    }

    impl Allocator for Host {
        fn allocate(&self, nbytes: usize) -> NonNull<u8> {
            self.reservations.fetch_add(1, Ordering::Relaxed);
            CpuAllocator.allocate(nbytes)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
            self.releases.fetch_add(1, Ordering::Relaxed);
            unsafe { CpuAllocator.deallocate(ptr, nbytes) }
        }
    }

    fn allocator(capacity: usize) -> StaticAllocator<Host> {
        counted(capacity).0
    }

    /// An allocator, and how often it reserved and released its region.
    fn counted(capacity: usize) -> (StaticAllocator<Host>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let (reservations, releases) =
            (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let host = Host {
            reservations: Arc::clone(&reservations),
            releases: Arc::clone(&releases),
        };
        (
            StaticAllocator::new(host, Device::Cuda(0), 256, capacity),
            reservations,
            releases,
        )
    }

    /// Free `ptr`, an allocation of `nbytes` from `allocator`.
    fn free(allocator: &StaticAllocator<Host>, ptr: NonNull<u8>, nbytes: usize) {
        unsafe { allocator.deallocate(ptr, nbytes) }
    }

    #[test]
    fn the_region_is_reserved_once_and_released_with_the_allocator() {
        let (allocator, reservations, releases) = counted(1 << 20);
        assert_eq!(reservations.load(Ordering::Relaxed), 0);
        let (a, b) = (allocator.allocate(100), allocator.allocate(100));
        assert_eq!(reservations.load(Ordering::Relaxed), 1);
        assert_eq!(allocator.capacity(), 1 << 20);
        free(&allocator, a, 100);
        free(&allocator, b, 100);
        assert_eq!(releases.load(Ordering::Relaxed), 0);
        drop(allocator);
        assert_eq!(releases.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_data_ptr_gives_its_allocation_back_on_drop() {
        let allocator = allocator(4096);
        let data = DataPtr::allocate(Arc::new(allocator.clone()), 100);
        assert_eq!((data.nbytes(), allocator.live_bytes()), (100, 100));
        drop(data);
        assert_eq!(allocator.live_bytes(), 0);
    }

    #[test]
    fn allocations_are_aligned_bumps() {
        let allocator = allocator(1 << 20);
        let a = allocator.allocate(100);
        let b = allocator.allocate(10);
        let c = allocator.allocate(300);
        let base = a.as_ptr().addr();
        assert_eq!(b.as_ptr().addr() - base, 256);
        assert_eq!(c.as_ptr().addr() - base, 512);
        assert_eq!(allocator.used(), 512 + 300);
        assert_eq!(allocator.live_bytes(), 410);
    }

    #[test]
    fn freed_memory_is_reused_while_others_are_live() {
        let allocator = allocator(1 << 20);
        let a = allocator.allocate(100);
        let b = allocator.allocate(100);
        free(&allocator, a, 100);
        assert_eq!(allocator.used(), 256 + 100);
        assert_eq!(allocator.live_bytes(), 100);
        // a's block is free again although b is live: first fit takes it.
        let c = allocator.allocate(200);
        assert_eq!(c, a);
        free(&allocator, b, 100);
        free(&allocator, c, 200);
        assert_eq!((allocator.used(), allocator.live_bytes()), (0, 0));
    }

    #[test]
    fn a_loop_with_live_inputs_does_not_grow() {
        // A compiled function called in a loop: its inputs stay live while
        // each call's workspace and output come and go.
        let allocator = allocator(4096);
        let _inputs = (allocator.allocate(1000), allocator.allocate(1000));
        for _ in 0..100 {
            let workspace = allocator.allocate(700);
            let output = allocator.allocate(300);
            free(&allocator, workspace, 700);
            free(&allocator, output, 300);
        }
        assert_eq!(allocator.used(), 1024 + 1000);
    }

    #[test]
    fn freed_neighbours_merge() {
        let allocator = allocator(1024);
        let blocks: Vec<_> = (0..4).map(|_| allocator.allocate(256)).collect();
        // Freed out of order, the four blocks merge back into one that holds
        // the whole region.
        for i in [2, 0, 3, 1] {
            free(&allocator, blocks[i], 256);
        }
        assert_eq!(allocator.allocate(1024), blocks[0]);
    }

    #[test]
    fn a_full_allocator_refuses_allocations() {
        let allocator = allocator(1024);
        let _a = allocator.allocate(1000);
        assert!(
            allocator.try_allocate(1).is_none(),
            "the next offset is past the end"
        );
        assert!(allocator.try_allocate(1 << 20).is_none());
    }

    #[test]
    #[should_panic(expected = "static_allocator_bytes")]
    fn allocating_past_the_end_names_the_setting() {
        allocator(1024).allocate(2048);
    }

    #[test]
    fn zero_byte_allocations_are_fine() {
        let allocator = allocator(1024);
        let a = allocator.allocate(0);
        let b = allocator.allocate(8);
        assert_eq!(a, b, "an empty allocation takes no room");
        free(&allocator, a, 0);
        free(&allocator, b, 8);
        assert_eq!(allocator.used(), 0);
    }

    #[test]
    fn the_memory_is_usable() {
        let allocator = allocator(4096);
        let data = allocator.allocate(1024);
        let slice = unsafe { std::slice::from_raw_parts_mut(data.as_ptr(), 1024) };
        slice.fill(0xAB);
        assert!(slice.iter().all(|&b| b == 0xAB));
    }
}

mod devices {
    //! Device selection for tensors and storage, on any machine.

    use crate::allocator::allocator_for;
    use crate::{Device, Storage, Tensor};

    #[test]
    fn cpu_is_always_available() {
        assert!(allocator_for(Device::Cpu).is_ok());
        let t = Tensor::zeros(&[2], Device::Cpu);
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(Storage::new(8, Device::Cpu).device(), Device::Cpu);
    }

    #[test]
    fn storage_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Storage>();
        let storage = Storage::new(16, Device::Cpu);
        std::thread::spawn(move || assert!(!storage.data_ptr().is_null()))
            .join()
            .unwrap();
    }

    #[test]
    fn unavailable_devices_are_reported() {
        if !crate::device::cuda::is_available() {
            let err = allocator_for(Device::Cuda(0)).err().unwrap();
            assert!(err.contains("CUDA device 0 is not available"), "{err}");
        }
        let past_last = Device::Cuda(crate::device::cuda::device_count());
        assert!(allocator_for(past_last).is_err());
        if !crate::device::mps::is_available() {
            assert!(allocator_for(Device::Mps).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "is not available")]
    fn tensor_on_unavailable_device_panics() {
        Tensor::zeros(&[1], Device::Cuda(crate::device::cuda::device_count()));
    }
}

mod cuda {
    //! The CUDA static allocator on a GPU. Each test skips when no device is visible.

    use crate::Device;
    use crate::allocator::allocator_for;

    #[test]
    fn availability_matches_the_device_count() {
        assert_eq!(
            crate::device::cuda::is_available(),
            crate::device::cuda::device_count() > 0
        );
        assert!(allocator_for(Device::Cuda(crate::device::cuda::device_count())).is_err());
    }

    #[cfg(lumen_cuda_linked)]
    mod device {
        use crate::allocator::cuda::{self, CudaBackend};
        use crate::allocator::static_allocator::StaticAllocator;
        use crate::{Allocator, Device, Storage};

        /// Skip guard: returns early from a test when there is no GPU.
        macro_rules! require_cuda {
            () => {
                if !crate::device::cuda::is_available() {
                    eprintln!("no CUDA device, skipping");
                    return;
                }
            };
        }

        #[test]
        fn reports_cuda_devices() {
            require_cuda!();
            let last = crate::device::cuda::device_count() - 1;
            assert_eq!(
                Storage::new(8, Device::Cuda(last)).device(),
                Device::Cuda(last)
            );
        }

        #[test]
        fn the_allocator_reserves_device_memory_and_rewinds() {
            require_cuda!();
            let allocator = StaticAllocator::new(
                CudaBackend::new(0),
                Device::Cuda(0),
                cuda::ALIGNMENT,
                1 << 20,
            );
            let a = allocator.allocate(1000);
            let b = allocator.allocate(1000);
            assert_eq!(a.as_ptr().addr() % cuda::ALIGNMENT, 0);
            assert_eq!(b.as_ptr().addr() - a.as_ptr().addr(), 1024);
            unsafe {
                allocator.deallocate(a, 1000);
                allocator.deallocate(b, 1000);
            }
            assert_eq!(allocator.allocate(8), a);
        }

        #[test]
        fn get_returns_the_same_global_instance() {
            require_cuda!();
            let (a, b) = (cuda::get(0), cuda::get(0));
            let size = 5 << 20;
            let p = a.allocate(size);
            // Other tests allocate on the global static allocator in parallel; they only
            // add to what is live.
            assert!(b.live_bytes() >= size);
            unsafe { a.deallocate(p, size) };
        }
    }
}

#[cfg(lumen_mps_linked)] // shim-backed types exist only when Metal is linked
mod mps {
    //! The MPS allocator, on the real Metal device. Every test skips when Metal
    //! is unavailable.

    use crate::allocator::mps;
    use crate::allocator::static_allocator::StaticAllocator;
    use crate::{Allocator, Device, Storage};

    /// Skip guard: returns early from a test when Metal is unavailable.
    macro_rules! require_mps {
        () => {
            if !crate::device::mps::is_available() {
                eprintln!("Metal unavailable, skipping");
                return;
            }
        };
    }

    fn fresh() -> mps::MpsAllocator {
        StaticAllocator::new(mps::MpsBackend, Device::Mps, mps::ALIGNMENT, 1 << 20)
    }

    #[test]
    fn reports_mps_device() {
        require_mps!();
        assert_eq!(Storage::new(8, Device::Mps).device(), Device::Mps);
    }

    #[test]
    fn get_returns_the_same_global_instance() {
        require_mps!();
        let (a, b) = (mps::get(), mps::get());
        let size = 5 << 20;
        let p = a.allocate(size);
        // Other tests allocate on the global static allocator in parallel; they only add
        // to what is live.
        assert!(b.live_bytes() >= size);
        unsafe { a.deallocate(p, size) };
    }

    #[test]
    fn unified_memory_is_host_writable() {
        require_mps!();
        // A shared-mode MTLBuffer is ordinary CPU memory on Apple Silicon:
        // write through the pointer and read it back.
        let allocator = fresh();
        let data = allocator.allocate(1024);
        let slice = unsafe { std::slice::from_raw_parts_mut(data.as_ptr(), 1024) };
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        assert!(slice.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8));
    }

    #[test]
    fn the_allocator_is_one_metal_buffer() {
        require_mps!();
        let allocator = fresh();
        let a = allocator.allocate(1000);
        let b = allocator.allocate(1000);
        assert_eq!(a.as_ptr().addr() % mps::ALIGNMENT, 0);
        assert_eq!(b.as_ptr().addr() - a.as_ptr().addr(), 1024);
    }
}
