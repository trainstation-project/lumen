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
#![cfg(lumen_mps_linked)] // file references shim-backed types; empty on other builds

use lumen::allocator::mps;
use lumen::{Allocator, CachingAllocator, Device, MpsPolicy};

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
    let p = a.allocate(512);
    let ptr = p.as_ptr();
    drop(p);
    let q = b.allocate(512);
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
