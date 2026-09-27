//! The MPS `fill_` kernel. A byte pattern uses Metal's blit `fillBuffer`
//! (the allocator's memset); a 2-, 4- or 8-byte element is written on the GPU
//! by a compute shader (`mps_fill.mm`), as in PyTorch. Strided views, and
//! memory that is not mapped (a custom allocator's), take the CPU kernel:
//! MPS memory is unified.

use crate::Tensor;
use crate::device::Device;
use crate::tensor::dtype::Element;
use crate::tensor::storage::as_bytes;

unsafe extern "C" {
    // In mps_fill.mm.
    fn lumen_mps_fill(
        ptr: *mut u8,
        pattern: *const u8,
        elem_size: usize,
        count: usize,
        gpu_start: *mut f64,
        gpu_end: *mut f64,
    ) -> i32;
}

pub(super) fn fill<T: Element>(t: &Tensor, value: T) {
    let numel = t.numel();
    let pattern = as_bytes(std::slice::from_ref(&value));
    let one_byte = pattern.iter().all(|&b| b == pattern[0]);
    if numel == 0 || one_byte || !t.is_contiguous() {
        return super::cpu::fill(t, value);
    }

    let dst = t
        .storage()
        .data_ptr()
        .wrapping_add(t.storage_offset() * pattern.len());

    // GPU start/end times, asked for only while profiling MPS.
    let mut times = crate::profiler::device_enabled(Device::Mps).then_some((0.0, 0.0));
    let (start_out, end_out): (*mut f64, *mut f64) = match &mut times {
        Some((start, end)) => (start, end),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    };

    let status = unsafe {
        lumen_mps_fill(
            dst,
            pattern.as_ptr(),
            pattern.len(),
            numel,
            start_out,
            end_out,
        )
    };

    match status {
        0 => {}
        -1 => return super::cpu::fill(t, value), // not mapped memory
        err => panic!(
            "Metal fill of {numel} {}-byte elements failed ({err})",
            pattern.len()
        ),
    }

    if let Some((start, end)) = times {
        // The GPU duration, ending when the host saw it finish.
        let end_ns = crate::profiler::now_ns();
        let duration_ns = ((end - start) * 1e9).max(0.0) as u64;
        crate::profiler::record_gpu(
            "Fill",
            Device::Mps,
            end_ns.saturating_sub(duration_ns),
            end_ns,
        );
    }
}
