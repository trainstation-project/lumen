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

mod ffi {
    use std::ffi::c_void;

    // `CUdeviceptr` is a u64, and a `CUstream` a `cudaStream_t`.
    #[link(name = "cuda")]
    unsafe extern "C" {
        pub fn cuMemsetD8Async(dst: u64, value: u8, count: usize, stream: *mut c_void) -> i32;
        pub fn cuMemsetD16Async(dst: u64, value: u16, count: usize, stream: *mut c_void) -> i32;
        pub fn cuMemsetD32Async(dst: u64, value: u32, count: usize, stream: *mut c_void) -> i32;
        pub fn cuMemsetD2D32Async(
            dst: u64,
            pitch: usize,
            value: u32,
            width: usize,
            height: usize,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Set `count` consecutive elements at `dst` on CUDA device `device_index`
/// to `pattern`, one element's bytes (1, 2, 4 or 8), on the device: no host
/// buffer is copied (PyTorch launches a fill kernel). Launched on the CUDA
/// stream without waiting.
fn memset_elements(device_index: usize, dst: *mut u8, pattern: &[u8], count: usize) {
    let dst = dst.addr() as u64;
    let mut err = 0;
    crate::stream::cuda::launch(device_index, "Memset", |stream| {
        err = unsafe {
            match *pattern {
                [b] => ffi::cuMemsetD8Async(dst, b, count, stream),
                [a, b] => ffi::cuMemsetD16Async(dst, u16::from_ne_bytes([a, b]), count, stream),
                [a, b, c, d] => {
                    ffi::cuMemsetD32Async(dst, u32::from_ne_bytes([a, b, c, d]), count, stream)
                }
                // 8 bytes: the low and the high 32-bit words, each as a 2D
                // memset of one word per 8-byte row.
                [a, b, c, d, e, f, g, h] => {
                    let lo = u32::from_ne_bytes([a, b, c, d]);
                    let hi = u32::from_ne_bytes([e, f, g, h]);
                    match ffi::cuMemsetD2D32Async(dst, 8, lo, 1, count, stream) {
                        0 => ffi::cuMemsetD2D32Async(dst + 4, 8, hi, 1, count, stream),
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
