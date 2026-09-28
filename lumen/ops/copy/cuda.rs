use std::ffi::c_void;

use super::{data_ptr, nbytes};
use crate::Tensor;
use crate::device::Device;

const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    copy(dst.device(), dst, src, HOST_TO_DEVICE);
}

pub(super) fn copy_d2h(dst: &Tensor, src: &Tensor) {
    copy(src.device(), dst, src, DEVICE_TO_HOST);
}

fn copy(device: Device, dst: &Tensor, src: &Tensor, kind: i32) {
    let Device::Cuda(index) = device else {
        unreachable!("the CUDA copy kernel runs on CUDA tensors")
    };
    let nbytes = nbytes(dst);
    let mut err = 0;
    crate::profiler::cupti::correlated(Device::Cuda(index), || {
        err = unsafe { cudaMemcpy(data_ptr(dst).cast(), data_ptr(src).cast(), nbytes, kind) };
    });
    assert_eq!(
        err, 0,
        "cudaMemcpy of {nbytes} bytes (kind {kind}) on cuda:{index} failed (error {err})"
    );
}
