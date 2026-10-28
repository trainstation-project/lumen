//! Unit tests for the CPU allocator, the `Region` sub-allocator and each device backend's allocator.

/// Tests of the host heap allocator.
mod cpu {
    use crate::{Allocator, CpuAllocator};

    /// `CpuAllocator::get` hands out one static instance.
    #[test]
    fn get_returns_the_same_global_instance() {
        let a = CpuAllocator::get();
        let b = CpuAllocator::get();
        assert!(std::ptr::addr_eq(a, b));
    }

    /// Host allocations start on a 64-byte boundary.
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

    /// Allocated host bytes can be written and read back.
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

    /// A zero-byte allocation yields an aligned pointer that can be freed.
    #[test]
    fn zero_byte_allocation_is_safe() {
        let data = CpuAllocator.allocate(0);
        assert_eq!(data.as_ptr() as usize % 64, 0);
        unsafe { CpuAllocator.deallocate(data, 0) };
    }

    /// Allocating and freeing sizes up to 2 KiB many times stays sound.
    #[test]
    fn repeated_alloc_and_free_cycles_do_not_crash() {
        for i in 0..100 {
            let nbytes = 1 << (i % 12);
            let data = CpuAllocator.allocate(nbytes);
            unsafe { CpuAllocator.deallocate(data, nbytes) };
        }
    }
}

/// Tests of `Region` backed by host memory.
mod region {
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::allocator::Region;
    use crate::{Allocator, CpuAllocator, DataPtr, Device};

    /// `Region` over CPU memory that counts how often its range is reserved and released.
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

    /// Releases the reserved range back to the CPU allocator, counting the release.
    impl Drop for Host {
        fn drop(&mut self) {
            if let Some(base) = self.region.base() {
                self.releases.fetch_add(1, Ordering::Relaxed);
                unsafe { CpuAllocator.deallocate(base, self.region.capacity()) }
            }
        }
    }

    /// Exposes the region's test accessors on `Host`.
    impl std::ops::Deref for Host {
        type Target = Region;

        fn deref(&self) -> &Region {
            &self.region
        }
    }

    /// Host-backed region of `capacity` bytes plus its reservation and release counters.
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

    /// Host-backed region of `capacity` bytes.
    fn allocator(capacity: usize) -> Host {
        counted(capacity).0
    }

    /// Gives `nbytes` at `ptr` back to `allocator`.
    fn free(allocator: &Host, ptr: NonNull<u8>, nbytes: usize) {
        unsafe { Allocator::deallocate(allocator, ptr, nbytes) }
    }

    /// The range is reserved on the first allocation and freed only when the allocator drops.
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

    /// Dropping a `DataPtr` returns its bytes to the region.
    #[test]
    fn a_data_ptr_gives_its_allocation_back_on_drop() {
        let allocator = Arc::new(allocator(4096));
        let data = DataPtr::allocate(Arc::clone(&allocator) as Arc<dyn Allocator>, 100);
        assert_eq!((data.nbytes(), allocator.live_bytes()), (100, 100));
        drop(data);
        assert_eq!(allocator.live_bytes(), 0);
    }

    /// Consecutive allocations land at increasing `ALIGNMENT` multiples.
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

    /// A freed block is reused by a later allocation that fits it.
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

    /// Repeated alloc/free of temporaries next to live inputs keeps the high-water mark fixed.
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

    /// Freeing neighbours in any order merges them back into one block.
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

    /// `try_take` returns `None` when no free block fits.
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

    /// An over-aligned block skips ahead and the skipped gap stays allocatable.
    #[test]
    fn aligned_allocations_start_at_their_alignment_and_leave_the_gap_free() {
        let allocator = allocator(1 << 16);
        let a = Allocator::allocate(&allocator, 300);
        let base = a.as_ptr().addr();
        let b = allocator
            .try_take_aligned(4096, 4096, |n| CpuAllocator.allocate(n))
            .unwrap();
        assert_eq!(b.as_ptr().addr() - base, 4096);
        let c = Allocator::allocate(&allocator, 1000);
        assert_eq!(
            c.as_ptr().addr() - base,
            512,
            "the gap before an aligned block is free"
        );
        free(&allocator, b, 4096);
        free(&allocator, c, 1000);
        free(&allocator, a, 300);
        assert_eq!(allocator.used(), 0);
        let whole = Allocator::allocate(&allocator, 1 << 16);
        assert_eq!(whole.as_ptr().addr(), base, "freed blocks merge back");
    }

    /// The out-of-memory panic points at `static_allocator_bytes`.
    #[test]
    #[should_panic(expected = "static_allocator_bytes")]
    fn allocating_past_the_end_names_the_setting() {
        Allocator::allocate(&allocator(1024), 2048);
    }

    /// Zero-byte allocations take no space.
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

    /// Region memory can be written and read through the returned pointer.
    #[test]
    fn the_memory_is_usable() {
        let allocator = allocator(4096);
        let data = Allocator::allocate(&allocator, 1024);
        let slice = unsafe { std::slice::from_raw_parts_mut(data.as_ptr(), 1024) };
        slice.fill(0xAB);
        assert!(slice.iter().all(|&b| b == 0xAB));
    }
}

/// Tests of `allocator_for` across devices.
mod devices {
    use crate::allocator::allocator_for;
    use crate::{Device, Storage, Tensor};

    /// CPU tensors and storage always allocate.
    #[test]
    fn cpu_is_always_available() {
        assert!(allocator_for(Device::Cpu(0)).is_ok());
        let t = Tensor::zeros(&[2], Device::Cpu(0));
        assert_eq!(t.device(), Device::Cpu(0));
        assert_eq!(Storage::new(8, Device::Cpu(0)).device(), Device::Cpu(0));
    }

    /// `allocator_for` returns the same allocator for repeated calls on every available device.
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

    /// `Storage` can move to another thread.
    #[test]
    fn storage_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Storage>();
        let storage = Storage::new(16, Device::Cpu(0));
        std::thread::spawn(move || assert!(!storage.data_ptr().is_null()))
            .join()
            .unwrap();
    }

    /// Missing CUDA, MPS or out-of-range devices yield an error instead of an allocator.
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

    /// Creating a tensor on a missing device panics.
    #[test]
    #[should_panic(expected = "is not available")]
    fn tensor_on_unavailable_device_panics() {
        Tensor::zeros(&[1], Device::Cuda(crate::device::cuda::CUDA.device_count()));
    }
}

/// CUDA allocator tests; the device ones run only with CUDA linked and a GPU present.
mod cuda {
    use crate::Device;
    use crate::allocator::allocator_for;

    /// CUDA reports available exactly when it has devices, and one past the last is rejected.
    #[test]
    fn availability_matches_the_device_count() {
        assert_eq!(
            crate::device::cuda::CUDA.is_available(),
            crate::device::cuda::CUDA.device_count() > 0
        );
        assert!(allocator_for(Device::Cuda(crate::device::cuda::CUDA.device_count())).is_err());
    }

    /// Tests that need a CUDA device.
    #[cfg(lumen_cuda_linked)]
    mod device {
        use crate::allocator::allocator_for;
        use crate::allocator::{ALIGNMENT, CudaAllocator};
        use crate::{Allocator, Device, Storage};

        /// Returns from the test early when no CUDA device is present.
        macro_rules! require_cuda {
            () => {
                if !crate::device::cuda::CUDA.is_available() {
                    eprintln!("no CUDA device, skipping");
                    return;
                }
            };
        }

        /// Storage on the last CUDA device reports that device.
        #[test]
        fn reports_cuda_devices() {
            require_cuda!();
            let last = crate::device::cuda::CUDA.device_count() - 1;
            assert_eq!(
                Storage::new(8, Device::Cuda(last)).device(),
                Device::Cuda(last)
            );
        }

        /// Blocks are `ALIGNMENT`-spaced and freed space is reused from the start.
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

        /// `allocator_for` shares one allocator for `cuda:0`.
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

/// Metal allocator tests, run only when a Metal device is present.
#[cfg(lumen_mps_linked)]
mod mps {
    use crate::allocator::allocator_for;
    use crate::allocator::{ALIGNMENT, MpsAllocator};
    use crate::{Allocator, Device, Storage};

    /// Returns from the test early when Metal is unavailable.
    macro_rules! require_mps {
        () => {
            if !crate::device::mps::MPS.is_available() {
                eprintln!("Metal unavailable, skipping");
                return;
            }
        };
    }

    /// Fresh 1 MiB MPS allocator separate from the shared one.
    fn fresh() -> MpsAllocator {
        MpsAllocator::new(1 << 20)
    }

    /// Storage on MPS reports the MPS device.
    #[test]
    fn reports_mps_device() {
        require_mps!();
        assert_eq!(Storage::new(8, Device::Mps(0)).device(), Device::Mps(0));
    }

    /// `allocator_for` shares one allocator for `mps:0`.
    #[test]
    fn get_returns_the_same_global_instance() {
        require_mps!();
        let (a, b) = (
            allocator_for(Device::Mps(0)).unwrap(),
            allocator_for(Device::Mps(0)).unwrap(),
        );
        assert!(std::sync::Arc::ptr_eq(&a, &b));
    }

    /// MPS allocations live in unified memory the host can write and read.
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

    /// Allocations are `ALIGNMENT`-spaced offsets into one Metal buffer.
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

/// TPU allocator tests; the device ones run only with the TPU runtime linked and a TPU present.
mod tpu {
    use crate::Device;
    use crate::allocator::allocator_for;
    use crate::device::DeviceBackend;

    /// TPU reports available exactly when it has devices, and one past the last is rejected.
    #[test]
    fn availability_matches_the_device_count() {
        let tpu: &dyn DeviceBackend = crate::device::tpu::TPU;
        assert_eq!(tpu.is_available(), tpu.device_count() > 0);
        let missing = allocator_for(Device::Tpu(tpu.device_count())).err();
        assert!(missing.is_some_and(|e| e.starts_with("TPU device")));
    }

    /// Tests that need a TPU device.
    #[cfg(lumen_tpu_linked)]
    mod device {
        use std::ptr::NonNull;

        use crate::allocator::tpu::copy_to_host;
        use crate::allocator::{ALIGNMENT, TpuAllocator, allocator_for};
        use crate::device::tpu::{ONE_AT_A_TIME, TPU, memory};
        use crate::{Allocator, DType, Device, Storage, Tensor, TensorOptions};

        /// Serializes TPU tests and returns early when no TPU is present.
        macro_rules! require_tpu {
            () => {
                let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
                if !TPU.is_available() {
                    eprintln!("no TPU device, skipping");
                    return;
                }
            };
        }

        /// Copies `nbytes` at `ptr` on `tpu:0` to the host.
        fn read(ptr: NonNull<u8>, nbytes: usize) -> Vec<u8> {
            let mut bytes = vec![0; nbytes];
            copy_to_host(0, ptr, &mut bytes).unwrap();
            bytes
        }

        /// Storage on the last TPU reports that device, and TPU devices parse and print.
        #[test]
        fn reports_tpu_devices() {
            require_tpu!();
            let last = TPU.device_count() - 1;
            assert_eq!(
                Storage::new(8, Device::Tpu(last)).device(),
                Device::Tpu(last)
            );
            assert_eq!(Device::of_type("tpu", 1), Some(Device::Tpu(1)));
            assert_eq!(Device::Tpu(1).to_string(), "tpu:1");
        }

        /// Reserving takes device memory, freed blocks are reused, and dropping returns the memory.
        #[test]
        fn the_allocator_reserves_device_memory_and_rewinds() {
            require_tpu!();
            let (before, limit) = memory(0).unwrap();
            let capacity = 64 << 20;
            assert!(limit > capacity);
            let allocator = TpuAllocator::new(0, capacity);
            let a = allocator.allocate(1000);
            let b = allocator.allocate(1000);
            assert_eq!(b.as_ptr().addr() - a.as_ptr().addr(), 1024);
            assert!(memory(0).unwrap().0 >= before + capacity);
            unsafe {
                allocator.deallocate(b, 1000);
                allocator.deallocate(a, 1000);
            }
            assert_eq!(allocator.allocate(ALIGNMENT), a);
            drop(allocator);
            assert!(memory(0).unwrap().0 < before + capacity);
        }

        /// Reads of a whole block, a slice inside it and another block return the right lengths and bytes.
        #[test]
        fn allocations_read_back_at_their_offsets() {
            require_tpu!();
            let allocator = allocator_for(Device::Tpu(0)).unwrap();
            let a = allocator.allocate(4096);
            let b = allocator.allocate(300);
            let whole = read(a, 4096);
            assert_eq!(read(a, 4096), whole);
            let inside = NonNull::new(a.as_ptr().wrapping_add(100)).unwrap();
            assert_eq!(read(inside, 200), whole[100..300]);
            assert_eq!(read(b, 300).len(), 300);
            unsafe {
                allocator.deallocate(b, 300);
                allocator.deallocate(a, 4096);
            }
        }

        /// Reading beyond the reserved buffer is an error.
        #[test]
        fn reads_past_the_region_fail() {
            require_tpu!();
            let allocator = allocator_for(Device::Tpu(0)).unwrap();
            let a = allocator.allocate(256);
            let past = NonNull::new(a.as_ptr().wrapping_add(4 << 30)).unwrap();
            assert!(copy_to_host(0, past, &mut [0; 16]).is_err());
            unsafe { allocator.deallocate(a, 256) };
        }

        /// Copying a TPU tensor to the host matches its raw device bytes.
        #[test]
        fn a_tpu_tensor_copies_to_the_host() {
            require_tpu!();
            let options = TensorOptions::new()
                .dtype(DType::F32)
                .device(Device::Tpu(0));
            let t = unsafe { Tensor::empty(&[64, 3], options) };
            let host = t.to(Device::Cpu(0));
            assert_eq!(host.shape(), &[64, 3]);
            let ptr = NonNull::new(t.data_ptr()).unwrap();
            let bytes: Vec<u8> = host
                .to_vec::<f32>()
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect();
            assert_eq!(bytes, read(ptr, 64 * 3 * 4));
        }
    }
}

/// Tests of the Trainium allocator against a fake Neuron runtime.
mod trainium {
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::{Arc, Mutex};

    use crate::Allocator;
    use crate::allocator::trainium::{Handle, Runtime, TrainiumAllocator};

    /// What a fake runtime tensor is: the root of `nbytes`, or a slice of it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Tensor {
        Root(usize),
        Slice { offset: usize, nbytes: usize },
    }

    /// The fake runtime's record: tensors made, in order, and those freed, in order.
    #[derive(Default)]
    struct Log {
        made: Vec<Tensor>,
        freed: Vec<Tensor>,
    }

    /// A runtime whose tensors are boxed descriptions and whose device memory is host memory.
    #[derive(Clone, Default)]
    struct Fake {
        log: Arc<Mutex<Log>>,
        /// Fails every root allocation when set.
        full: bool,
    }

    impl Fake {
        /// A handle to `tensor`, recorded as made.
        fn handle(&self, tensor: Tensor) -> Handle {
            self.log.lock().unwrap().made.push(tensor);
            Handle(NonNull::from(Box::leak(Box::new(tensor))).cast::<c_void>())
        }

        /// Tensors made so far.
        fn made(&self) -> Vec<Tensor> {
            self.log.lock().unwrap().made.clone()
        }

        /// Tensors freed so far.
        fn freed(&self) -> Vec<Tensor> {
            self.log.lock().unwrap().freed.clone()
        }
    }

    impl Runtime for Fake {
        /// A root handle over `nbytes` of leaked host memory standing in for device memory.
        fn allocate(&self, _core: usize, nbytes: usize) -> Result<(Handle, NonNull<u8>), String> {
            if self.full {
                return Err("NRT_RESOURCE".into());
            }
            let memory = Box::leak(vec![0u8; nbytes].into_boxed_slice());
            Ok((
                self.handle(Tensor::Root(nbytes)),
                NonNull::from(memory).cast(),
            ))
        }

        /// A slice handle recording where it starts and how long it is.
        fn slice(&self, _root: Handle, offset: usize, nbytes: usize) -> Result<Handle, String> {
            Ok(self.handle(Tensor::Slice { offset, nbytes }))
        }

        /// Records the tensor as freed and drops its description.
        fn free(&self, tensor: Handle) {
            let tensor = unsafe { Box::from_raw(tensor.0.cast::<Tensor>().as_ptr()) };
            self.log.lock().unwrap().freed.push(*tensor);
        }
    }

    /// A Trainium allocator of `capacity` bytes over `runtime`.
    fn allocator(capacity: usize, runtime: &Fake) -> TrainiumAllocator<Fake> {
        TrainiumAllocator::new(0, capacity, runtime.clone())
    }

    /// The first allocation reserves one root tensor of the whole capacity; every block is a slice
    /// of it at the block's offset and size; dropping the allocator frees the live slices, then the root.
    #[test]
    fn blocks_are_slices_of_one_root_tensor() {
        let runtime = Fake::default();
        let trainium = allocator(1 << 20, &runtime);
        assert!(
            runtime.made().is_empty(),
            "nothing is reserved before the first allocation"
        );
        let a = trainium.allocate(100);
        let b = trainium.allocate(1000);
        let root = Tensor::Root(1 << 20);
        let slice = |offset, nbytes| Tensor::Slice { offset, nbytes };
        assert_eq!(runtime.made(), [root, slice(0, 100), slice(256, 1000)]);
        assert_eq!(b.as_ptr().addr() - a.as_ptr().addr(), 256);
        unsafe { trainium.deallocate(a, 100) };
        assert_eq!(runtime.freed(), [slice(0, 100)]);
        drop(trainium);
        assert_eq!(runtime.freed(), [slice(0, 100), slice(256, 1000), root]);
    }

    /// A freed block's place is reused, with a new slice of the same offset.
    #[test]
    fn a_freed_block_is_sliced_again_when_reused() {
        let runtime = Fake::default();
        let trainium = allocator(1 << 20, &runtime);
        let a = trainium.allocate(512);
        unsafe { trainium.deallocate(a, 512) };
        let b = trainium.allocate(300);
        assert_eq!(a, b);
        assert_eq!(
            runtime.made().last(),
            Some(&Tensor::Slice {
                offset: 0,
                nbytes: 300
            })
        );
        unsafe { trainium.deallocate(b, 300) };
    }

    /// A pointer at or inside a live block (a view of a tensor) finds the block's slice and its
    /// offset there; a freed block's pointer finds none.
    #[test]
    fn pointers_find_their_blocks_slice() {
        let runtime = Fake::default();
        let trainium = allocator(1 << 20, &runtime);
        let a = trainium.allocate(100);
        let b = trainium.allocate(1000);
        let at = |ptr: NonNull<u8>, bytes: usize| {
            let (slice, offset) =
                trainium.tensor(NonNull::new(ptr.as_ptr().wrapping_add(bytes)).unwrap())?;
            Some((unsafe { *slice.0.cast::<Tensor>().as_ptr() }, offset))
        };
        assert_eq!(
            at(a, 0),
            Some((
                Tensor::Slice {
                    offset: 0,
                    nbytes: 100
                },
                0
            ))
        );
        assert_eq!(
            at(b, 999),
            Some((
                Tensor::Slice {
                    offset: 256,
                    nbytes: 1000
                },
                999
            ))
        );
        assert_eq!(at(a, 100), None, "the padding after a block is no block's");
        unsafe { trainium.deallocate(a, 100) };
        assert_eq!(at(a, 0), None);
        unsafe { trainium.deallocate(b, 1000) };
    }

    /// A zero-byte allocation makes no slice, and freeing it frees none.
    #[test]
    fn zero_byte_allocations_have_no_slice() {
        let runtime = Fake::default();
        let trainium = allocator(1 << 20, &runtime);
        let empty = trainium.allocate(0);
        assert_eq!(runtime.made(), [Tensor::Root(1 << 20)]);
        unsafe { trainium.deallocate(empty, 0) };
        assert!(runtime.freed().is_empty());
    }

    /// A root tensor the runtime can't allocate is an out-of-memory panic naming the device.
    #[test]
    #[should_panic(expected = "Trainium out of memory: failed to reserve")]
    fn a_failed_reservation_panics() {
        let runtime = Fake {
            full: true,
            ..Fake::default()
        };
        allocator(1 << 20, &runtime).allocate(64);
    }
}
