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
use lumen::{Allocator, CachingAllocator, DType, Device, MpsPolicy, TensorOptions};

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

// ---------------- tensors on MPS ----------------

#[test]
fn tensor_on_mps_roundtrips() {
    require_mps!();
    let t = lumen::Tensor::arange(
        6,
        TensorOptions::new().dtype(DType::F32).device(Device::Mps),
    )
    .reshape(&[2, 3]);
    assert_eq!(t.device(), Device::Mps);
    assert_eq!(t.get::<f32>(&[1, 2]), 5.0);
    t.set(&[0, 1], 9.0f32);
    assert_eq!(t.to_vec::<f32>(), vec![0.0, 9.0, 2.0, 3.0, 4.0, 5.0]);
    // Views share the Metal buffer.
    let col = t.select(1, 1);
    assert!(col.shares_storage_with(&t));
    assert_eq!(col.to_vec::<f32>(), vec![9.0, 4.0]);
}

#[test]
fn tensor_moves_between_cpu_and_mps() {
    require_mps!();
    let cpu = lumen::Tensor::from_slice(&[1i64, 2, 3, 4], Device::Cpu).reshape(&[2, 2]);
    let on_mps = cpu.transpose(0, 1).to(Device::Mps);
    assert_eq!(on_mps.device(), Device::Mps);
    assert!(!on_mps.is_contiguous(), "layout is kept");
    assert_eq!(on_mps.to_vec::<i64>(), vec![1, 3, 2, 4]);
    let contiguous = on_mps.contiguous::<i64>();
    assert_eq!(contiguous.device(), Device::Mps);
    let back = contiguous.to(Device::Cpu);
    assert_eq!(back.device(), Device::Cpu);
    assert_eq!(back.to_vec::<i64>(), vec![1, 3, 2, 4]);
}

#[test]
fn zeros_on_mps_are_zero() {
    require_mps!();
    let t = lumen::Tensor::zeros(
        &[3, 5],
        TensorOptions::new().dtype(DType::Bool).device(Device::Mps),
    );
    assert!(t.to_vec::<bool>().iter().all(|&b| !b));
}
