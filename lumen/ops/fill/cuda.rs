//! The CUDA `fill_` kernel: element memsets with the driver API (the
//! runtime's `cudaMemset` only sets bytes), timed for the profiler with CUDA
//! events. Compiled only when cudart is linked (cfg `lumen_cuda_linked`).

use crate::Tensor;
use crate::device::Device;
use crate::tensor::dtype::Element;
use crate::tensor::storage::as_bytes;

/// CUDA `fill_`: the element's bytes written on the device with the driver
/// API's memsets, so no host buffer is copied over. Strided views take the
/// host read-modify-write of [`super::cpu::fill`].
pub(super) fn fill<T: Element>(t: &Tensor, value: T) {
    let numel = t.numel();
    if numel == 0 {
        return;
    }

    if !t.is_contiguous() {
        return super::cpu::fill(t, value);
    }

    let Device::Cuda(index) = t.device() else {
        unreachable!("the CUDA kernel runs on CUDA tensors")
    };

    let dst = t
        .storage()
        .data_ptr()
        .wrapping_add(t.storage_offset() * size_of::<T>());
    let pattern = as_bytes(std::slice::from_ref(&value));
    memset_elements(index, dst, pattern, numel);
}

// The runtime API (device and event calls) comes from the allocator's
// bindings; the driver API's memsets are only used here.
use crate::allocator::cuda::ffi as cudart;

mod ffi {
    // `CUdeviceptr` is a u64; the exported symbols are the `_v2` ones
    // `cuda.h` maps the names to.
    #[link(name = "cuda")]
    unsafe extern "C" {
        pub fn cuMemsetD8_v2(dst: u64, value: u8, count: usize) -> i32;
        pub fn cuMemsetD16_v2(dst: u64, value: u16, count: usize) -> i32;
        pub fn cuMemsetD32_v2(dst: u64, value: u32, count: usize) -> i32;
        pub fn cuMemsetD2D32_v2(
            dst: u64,
            pitch: usize,
            value: u32,
            width: usize,
            height: usize,
        ) -> i32;
    }
}

/// Set `count` consecutive elements at `dst` on CUDA device `device_index`
/// to `pattern`, one element's bytes (1, 2, 4 or 8), on the device: no host
/// buffer is copied (PyTorch launches a fill kernel).
fn memset_elements(device_index: usize, dst: *mut u8, pattern: &[u8], count: usize) {
    let dst = dst.addr() as u64;
    let mut err = 0;
    timed(device_index, "Memset", || {
        err = unsafe {
            match *pattern {
                [b] => ffi::cuMemsetD8_v2(dst, b, count),
                [a, b] => ffi::cuMemsetD16_v2(dst, u16::from_ne_bytes([a, b]), count),
                [a, b, c, d] => ffi::cuMemsetD32_v2(dst, u32::from_ne_bytes([a, b, c, d]), count),
                // 8 bytes: the low and the high 32-bit words, each as a 2D
                // memset of one word per 8-byte row.
                [a, b, c, d, e, f, g, h] => {
                    let lo = u32::from_ne_bytes([a, b, c, d]);
                    let hi = u32::from_ne_bytes([e, f, g, h]);
                    match ffi::cuMemsetD2D32_v2(dst, 8, lo, 1, count) {
                        0 => ffi::cuMemsetD2D32_v2(dst + 4, 8, hi, 1, count),
                        err => err,
                    }
                }
                _ => panic!("no memset for {}-byte elements", pattern.len()),
            }
        };
    });
    assert_eq!(
        err,
        0,
        "CUDA driver memset of {count} {}-byte elements on cuda:{device_index} failed (error {err})",
        pattern.len()
    );
}

/// Run `work` on device `device_index` (on the legacy default stream).
/// When the profiler times CUDA, bracket it with events and record it as
/// `name`: its GPU duration, ending when the host saw it finish.
fn timed(device_index: usize, name: &'static str, work: impl FnOnce()) {
    unsafe { cudart::cudaSetDevice(device_index as i32) };
    let device = Device::Cuda(device_index);
    if !crate::profiler::device_enabled(device) {
        return work();
    }
    let (mut start, mut stop) = (std::ptr::null_mut(), std::ptr::null_mut());
    let created = unsafe {
        cudart::cudaEventCreate(&mut start) == 0 && cudart::cudaEventCreate(&mut stop) == 0
    };
    if !created {
        return work();
    }
    let null_stream = std::ptr::null_mut();
    unsafe { cudart::cudaEventRecord(start, null_stream) };
    work();
    let mut ms = 0.0f32;
    let finished = unsafe {
        cudart::cudaEventRecord(stop, null_stream) == 0
            && cudart::cudaEventSynchronize(stop) == 0
            && cudart::cudaEventElapsedTime(&mut ms, start, stop) == 0
    };
    let end_ns = crate::profiler::now_ns();
    unsafe {
        cudart::cudaEventDestroy(start);
        cudart::cudaEventDestroy(stop);
    }
    if finished {
        let duration_ns = (f64::from(ms) * 1e6).max(0.0) as u64;
        crate::profiler::record_gpu(name, device, end_ns.saturating_sub(duration_ns), end_ns);
    }
}
