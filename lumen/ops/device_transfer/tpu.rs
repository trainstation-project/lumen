//! TPU-to-host copies, through the TPU allocator's reads of its region
//! (`allocator::tpu::copy_to_host`); host-to-TPU copies, through a Pallas
//! kernel (`tpu.py`) that copies the host's bytes (passed to it, whole
//! rows) to the destination's rows.

use std::ptr::NonNull;

use crate::Tensor;
use crate::allocator::tpu::ROW;
use crate::device::Device;
use crate::ops::tpu::{Launch, launch};

pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    let Device::Tpu(index) = dst.device() else {
        unreachable!("the TPU copy kernel runs on TPU tensors")
    };
    let nbytes = dst.nbytes();
    // Whole rows: a TPU allocation's, its padding the destination's own.
    let mut host = vec![0u8; nbytes.next_multiple_of(ROW)];
    // SAFETY: `src` is a host tensor of `nbytes` contiguous bytes.
    host[..nbytes].copy_from_slice(unsafe { std::slice::from_raw_parts(src.data_ptr(), nbytes) });
    let launched = Launch {
        module: "lumen.ops.device_transfer.tpu",
        spec: format!("\"host_rows\": {}", host.len() / ROW),
        rows: &[dst.data_ptr().cast_const()],
        scalars: &[],
        host: &host,
    };
    launch(index, launched)
        .unwrap_or_else(|e| panic!("copying {nbytes} bytes to tpu:{index} failed: {e}"));
}

pub(super) fn copy_d2h(dst: &Tensor, src: &Tensor) {
    let Device::Tpu(index) = src.device() else {
        unreachable!("the TPU copy kernel runs on TPU tensors")
    };
    let nbytes = dst.nbytes();
    let Some(ptr) = NonNull::new(src.data_ptr()) else {
        return;
    };
    // SAFETY: `dst` is a host tensor of `nbytes` contiguous bytes, which
    // this copy alone writes (`copy_to`'s caller ensures it).
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.data_ptr(), nbytes) };
    crate::allocator::tpu::copy_to_host(index, ptr, dst)
        .unwrap_or_else(|e| panic!("copying {nbytes} bytes from tpu:{index} failed: {e}"));
}
