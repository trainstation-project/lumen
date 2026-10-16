mod cpu {
    use crate::{Allocator, CpuAllocator};

    #[test]
    fn get_returns_the_same_global_instance() {
        let a = CpuAllocator::get();
        let b = CpuAllocator::get();
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
        assert_eq!(data.as_ptr() as usize % 64, 0);
        unsafe { CpuAllocator.deallocate(data, 0) };
    }

    #[test]
    fn repeated_alloc_and_free_cycles_do_not_crash() {
        for i in 0..100 {
            let nbytes = 1 << (i % 12);
            let data = CpuAllocator.allocate(nbytes);
            unsafe { CpuAllocator.deallocate(data, nbytes) };
        }
    }
}

mod region {
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::allocator::Region;
    use crate::{Allocator, CpuAllocator, DataPtr, Device};

    struct Host {
        region: Region,
        reservations: Arc<AtomicUsize>,
        releases: Arc<AtomicUsize>,
    }

    impl Allocator for Host {
        fn allocate(&self, nbytes: usize) -> NonNull<u8> {
            let reserve = |n| {
                self.reservations.fetch_add(1, Ordering::Relaxed);
                CpuAllocator.allocate(n)
            };
            self.region.take(nbytes, reserve, || {})
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
            self.region.give(ptr, nbytes)
        }
    }

    impl Drop for Host {
        fn drop(&mut self) {
            if let Some(base) = self.region.base() {
                self.releases.fetch_add(1, Ordering::Relaxed);
                unsafe { CpuAllocator.deallocate(base, self.region.capacity()) }
            }
        }
    }

    impl std::ops::Deref for Host {
        type Target = Region;

        fn deref(&self) -> &Region {
            &self.region
        }
    }

    fn counted(capacity: usize) -> (Host, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let (reservations, releases) =
            (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let host = Host {
            region: Region::new(Device::Cpu(0), capacity),
            reservations: Arc::clone(&reservations),
            releases: Arc::clone(&releases),
        };
        (host, reservations, releases)
    }

    fn allocator(capacity: usize) -> Host {
        counted(capacity).0
    }

    fn free(allocator: &Host, ptr: NonNull<u8>, nbytes: usize) {
        unsafe { Allocator::deallocate(allocator, ptr, nbytes) }
    }

    #[test]
    fn the_region_is_reserved_once_and_released_with_the_allocator() {
        let (allocator, reservations, releases) = counted(1 << 20);
        assert_eq!(reservations.load(Ordering::Relaxed), 0);
        let (a, b) = (
            Allocator::allocate(&allocator, 100),
            Allocator::allocate(&allocator, 100),
        );
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
        let allocator = Arc::new(allocator(4096));
        let data = DataPtr::allocate(Arc::clone(&allocator) as Arc<dyn Allocator>, 100);
        assert_eq!((data.nbytes(), allocator.live_bytes()), (100, 100));
        drop(data);
        assert_eq!(allocator.live_bytes(), 0);
    }

    #[test]
    fn allocations_are_aligned_bumps() {
        let allocator = allocator(1 << 20);
        let a = Allocator::allocate(&allocator, 100);
        let b = Allocator::allocate(&allocator, 10);
        let c = Allocator::allocate(&allocator, 300);
        let base = a.as_ptr().addr();
        assert_eq!(b.as_ptr().addr() - base, 256);
        assert_eq!(c.as_ptr().addr() - base, 512);
        assert_eq!(allocator.used(), 512 + 300);
        assert_eq!(allocator.live_bytes(), 410);
    }

    #[test]
    fn freed_memory_is_reused_while_others_are_live() {
        let allocator = allocator(1 << 20);
        let a = Allocator::allocate(&allocator, 100);
        let b = Allocator::allocate(&allocator, 100);
        free(&allocator, a, 100);
        assert_eq!(allocator.used(), 256 + 100);
        assert_eq!(allocator.live_bytes(), 100);
        let c = Allocator::allocate(&allocator, 200);
        assert_eq!(c, a);
        free(&allocator, b, 100);
        free(&allocator, c, 200);
        assert_eq!((allocator.used(), allocator.live_bytes()), (0, 0));
    }

    #[test]
    fn a_loop_with_live_inputs_does_not_grow() {
        let allocator = allocator(4096);
        let _inputs = (
            Allocator::allocate(&allocator, 1000),
            Allocator::allocate(&allocator, 1000),
        );
        for _ in 0..100 {
            let workspace = Allocator::allocate(&allocator, 700);
            let output = Allocator::allocate(&allocator, 300);
            free(&allocator, workspace, 700);
            free(&allocator, output, 300);
        }
        assert_eq!(allocator.used(), 1024 + 1000);
    }

    #[test]
    fn freed_neighbours_merge() {
        let allocator = allocator(1024);
        let blocks: Vec<_> = (0..4)
            .map(|_| Allocator::allocate(&allocator, 256))
            .collect();
        for i in [2, 0, 3, 1] {
            free(&allocator, blocks[i], 256);
        }
        assert_eq!(Allocator::allocate(&allocator, 1024), blocks[0]);
    }

    #[test]
    fn a_full_allocator_refuses_allocations() {
        let allocator = allocator(1024);
        let _a = Allocator::allocate(&allocator, 1000);
        assert!(
            allocator
                .try_take(1, |n| CpuAllocator.allocate(n))
                .is_none(),
            "the next offset is past the end"
        );
        assert!(
            allocator
                .try_take(1 << 20, |n| CpuAllocator.allocate(n))
                .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "static_allocator_bytes")]
    fn allocating_past_the_end_names_the_setting() {
        Allocator::allocate(&allocator(1024), 2048);
    }

    #[test]
    fn zero_byte_allocations_are_fine() {
        let allocator = allocator(1024);
        let a = Allocator::allocate(&allocator, 0);
        let b = Allocator::allocate(&allocator, 8);
        assert_eq!(a, b, "an empty allocation takes no room");
        free(&allocator, a, 0);
        free(&allocator, b, 8);
        assert_eq!(allocator.used(), 0);
    }

    #[test]
    fn the_memory_is_usable() {
        let allocator = allocator(4096);
        let data = Allocator::allocate(&allocator, 1024);
        let slice = unsafe { std::slice::from_raw_parts_mut(data.as_ptr(), 1024) };
        slice.fill(0xAB);
        assert!(slice.iter().all(|&b| b == 0xAB));
    }
}

mod devices {
    use crate::allocator::allocator_for;
    use crate::{Device, Storage, Tensor};

    #[test]
    fn cpu_is_always_available() {
        assert!(allocator_for(Device::Cpu(0)).is_ok());
        let t = Tensor::zeros(&[2], Device::Cpu(0));
        assert_eq!(t.device(), Device::Cpu(0));
        assert_eq!(Storage::new(8, Device::Cpu(0)).device(), Device::Cpu(0));
    }

    #[test]
    fn every_device_has_one_shared_allocator() {
        use std::sync::Arc;
        let devices = [
            Device::Cpu(0),
            Device::Meta(0),
            Device::Mps(0),
            Device::Cuda(0),
        ];
        for device in devices.into_iter().filter(|d| d.check().is_ok()) {
            let (a, b) = (
                allocator_for(device).unwrap(),
                allocator_for(device).unwrap(),
            );
            assert!(Arc::ptr_eq(&a, &b), "{device}");
            let ptr = a.allocate(64);
            unsafe { a.deallocate(ptr, 64) };
        }
    }

    #[test]
    fn storage_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Storage>();
        let storage = Storage::new(16, Device::Cpu(0));
        std::thread::spawn(move || assert!(!storage.data_ptr().is_null()))
            .join()
            .unwrap();
    }

    #[test]
    fn unavailable_devices_are_reported() {
        if !crate::device::cuda::CUDA.is_available() {
            let err = allocator_for(Device::Cuda(0)).err().unwrap();
            assert!(err.contains("CUDA device 0 is not available"), "{err}");
        }
        let past_last = Device::Cuda(crate::device::cuda::CUDA.device_count());
        assert!(allocator_for(past_last).is_err());
        if !crate::device::mps::MPS.is_available() {
            assert!(allocator_for(Device::Mps(0)).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "is not available")]
    fn tensor_on_unavailable_device_panics() {
        Tensor::zeros(&[1], Device::Cuda(crate::device::cuda::CUDA.device_count()));
    }
}

mod cuda {
    use crate::Device;
    use crate::allocator::allocator_for;

    #[test]
    fn availability_matches_the_device_count() {
        assert_eq!(
            crate::device::cuda::CUDA.is_available(),
            crate::device::cuda::CUDA.device_count() > 0
        );
        assert!(allocator_for(Device::Cuda(crate::device::cuda::CUDA.device_count())).is_err());
    }

    #[cfg(lumen_cuda_linked)]
    mod device {
        use crate::allocator::allocator_for;
        use crate::allocator::{ALIGNMENT, CudaAllocator};
        use crate::{Allocator, Device, Storage};

        macro_rules! require_cuda {
            () => {
                if !crate::device::cuda::CUDA.is_available() {
                    eprintln!("no CUDA device, skipping");
                    return;
                }
            };
        }

        #[test]
        fn reports_cuda_devices() {
            require_cuda!();
            let last = crate::device::cuda::CUDA.device_count() - 1;
            assert_eq!(
                Storage::new(8, Device::Cuda(last)).device(),
                Device::Cuda(last)
            );
        }

        #[test]
        fn the_allocator_reserves_device_memory_and_rewinds() {
            require_cuda!();
            let allocator = CudaAllocator::new(0, 1 << 20);
            let a = allocator.allocate(1000);
            let b = allocator.allocate(1000);
            assert_eq!(a.as_ptr().addr() % ALIGNMENT, 0);
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
            let (a, b) = (
                allocator_for(Device::Cuda(0)).unwrap(),
                allocator_for(Device::Cuda(0)).unwrap(),
            );
            assert!(std::sync::Arc::ptr_eq(&a, &b));
        }
    }
}

#[cfg(lumen_mps_linked)]
mod mps {
    use crate::allocator::allocator_for;
    use crate::allocator::{ALIGNMENT, MpsAllocator};
    use crate::{Allocator, Device, Storage};

    macro_rules! require_mps {
        () => {
            if !crate::device::mps::MPS.is_available() {
                eprintln!("Metal unavailable, skipping");
                return;
            }
        };
    }

    fn fresh() -> MpsAllocator {
        MpsAllocator::new(1 << 20)
    }

    #[test]
    fn reports_mps_device() {
        require_mps!();
        assert_eq!(Storage::new(8, Device::Mps(0)).device(), Device::Mps(0));
    }

    #[test]
    fn get_returns_the_same_global_instance() {
        require_mps!();
        let (a, b) = (
            allocator_for(Device::Mps(0)).unwrap(),
            allocator_for(Device::Mps(0)).unwrap(),
        );
        assert!(std::sync::Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn unified_memory_is_host_writable() {
        require_mps!();
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
        assert_eq!(a.as_ptr().addr() % ALIGNMENT, 0);
        assert_eq!(b.as_ptr().addr() - a.as_ptr().addr(), 1024);
    }
}
