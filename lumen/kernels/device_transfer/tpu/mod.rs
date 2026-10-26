//! TPU host/device copies; device memory holds tensors in the TPU tiled layout.

use std::ptr::NonNull;

use crate::Tensor;
use crate::allocator::tpu::ROW;
use crate::device::Device;
use crate::device::tpu::layout::Layout;
use crate::kernels::tpu::{Launch, flush, launch};

/// The TPU tiled layout of `t`'s dtype and shape.
fn layout(t: &Tensor) -> Layout {
    Layout::new(t.dtype(), t.shape()).unwrap_or_else(|e| panic!("{e}"))
}

/// Tiles host tensor `src` and launches a DMA kernel writing it into TPU tensor `dst`.
pub(super) fn copy_h2d(dst: &Tensor, src: &Tensor) {
    let Device::Tpu(index) = dst.device() else {
        unreachable!("the TPU copy kernel runs on TPU tensors")
    };
    let nbytes = dst.nbytes();
    let bytes = unsafe { std::slice::from_raw_parts(src.data_ptr(), nbytes) };
    let host = layout(dst).tile(bytes);
    debug_assert!(host.len().is_multiple_of(ROW));
    let launched = Launch {
        module: "lumen.kernels.device_transfer.tpu",
        kernel: "copy_h2d".to_owned(),
        spec: String::new(),
        rows: &[dst.data_ptr().cast_const()],
        scalars: &[],
        host: &host,
        sparse: None,
    };
    launch(index, launched)
        .unwrap_or_else(|e| panic!("copying {nbytes} bytes to tpu:{index} failed: {e}"));
}

/// Runs `src`'s device's pending launches, reads its tiles back and untiles them into host tensor `dst`.
pub(super) fn copy_d2h(dst: &Tensor, src: &Tensor) {
    let Device::Tpu(index) = src.device() else {
        unreachable!("the TPU copy kernel runs on TPU tensors")
    };
    let nbytes = dst.nbytes();
    // A null pointer is an empty tensor with nothing to copy.
    let Some(ptr) = NonNull::new(src.data_ptr()) else {
        return;
    };
    flush(index)
        .unwrap_or_else(|e| panic!("running tpu:{index}'s launches before a copy failed: {e}"));
    let layout = layout(src);
    let mut memory = vec![0u8; layout.bytes()];
    crate::allocator::tpu::copy_to_host(index, ptr, &mut memory)
        .unwrap_or_else(|e| panic!("copying {nbytes} bytes from tpu:{index} failed: {e}"));
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.data_ptr(), nbytes) };
    layout.untile(&memory, dst);
}
