//! `lumen::config::set_memory_caching`. The flag is process-wide, so this
//! file holds a single test: it runs in its own test binary, where flipping
//! the flag cannot race the caching tests in other files.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use lumen::{Allocator, CachingAllocator, CudaPolicy, DataPtr, Device, config};

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
    config::set_memory_caching(false);
    let a = alloc.allocate(100);
    let b = alloc.allocate(3 << 20);
    assert_eq!(log.lock().unwrap().allocs, vec![2 << 20, 100, 3 << 20]);
    assert_eq!(alloc.stats().allocated_bytes, 512, "only the cached block");
    drop(a);
    drop(b);
    assert_eq!(log.lock().unwrap().frees, 2);
    assert_eq!(alloc.allocate(0).as_ptr(), NonNull::dangling().as_ptr());

    // A block allocated while caching was on still returns to the cache.
    drop(cached);
    assert_eq!(alloc.stats().allocated_bytes, 0);
    assert_eq!(log.lock().unwrap().frees, 2, "the segment stays cached");

    // Back on: the cached segment is reused.
    config::set_memory_caching(true);
    drop(alloc.allocate(100));
    assert_eq!(log.lock().unwrap().allocs.len(), 3);

    drop(alloc); // last reference: the cached segment goes back to the device
    assert_eq!(log.lock().unwrap().frees, 3);
}
