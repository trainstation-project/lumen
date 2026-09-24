use lumen::{Allocator, CpuAllocator, DataPtr, Device};

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
