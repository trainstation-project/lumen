//! Host/device copy ops, dispatched to each backend's copy kernel.

mod cpu;
#[cfg(lumen_cuda_linked)]
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;
#[cfg(lumen_tpu_linked)]
mod tpu;

use crate::Tensor;
use crate::device::Device;
use crate::kernels::{DispatchKey, Op};

/// A copy kernel: writes all of `src`'s bytes into `dst`.
pub type CopyKernel = fn(dst: &Tensor, src: &Tensor);

/// Host-to-device copy op.
pub static COPY_H2D: Op<CopyKernel> = Op::new(op_name!("copy_h2d"), h2d_kernels);

/// Device-to-host copy op.
pub static COPY_D2H: Op<CopyKernel> = Op::new(op_name!("copy_d2h"), d2h_kernels);

/// The host-to-device kernel for a dispatch key, if the backend is built in.
fn h2d_kernels(key: DispatchKey) -> Option<CopyKernel> {
    match key {
        DispatchKey::Cpu => Some(cpu::memcpy),
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(mps::copy_h2d),
        #[cfg(not(lumen_mps_linked))]
        DispatchKey::Mps => None,
        #[cfg(lumen_cuda_linked)]
        DispatchKey::Cuda => Some(cuda::copy_h2d),
        #[cfg(not(lumen_cuda_linked))]
        DispatchKey::Cuda => None,
        #[cfg(lumen_tpu_linked)]
        DispatchKey::Tpu => Some(tpu::copy_h2d),
        #[cfg(not(lumen_tpu_linked))]
        DispatchKey::Tpu => None,
        DispatchKey::Meta => None,
    }
}

/// The device-to-host kernel for a dispatch key, if the backend is built in.
fn d2h_kernels(key: DispatchKey) -> Option<CopyKernel> {
    match key {
        DispatchKey::Cpu => Some(cpu::memcpy),
        // MPS memory is host-visible, so reading it back is a plain memcpy.
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(cpu::memcpy),
        #[cfg(not(lumen_mps_linked))]
        DispatchKey::Mps => None,
        #[cfg(lumen_cuda_linked)]
        DispatchKey::Cuda => Some(cuda::copy_d2h),
        #[cfg(not(lumen_cuda_linked))]
        DispatchKey::Cuda => None,
        #[cfg(lumen_tpu_linked)]
        DispatchKey::Tpu => Some(tpu::copy_d2h),
        #[cfg(not(lumen_tpu_linked))]
        DispatchKey::Tpu => None,
        DispatchKey::Meta => None,
    }
}

/// Copies CPU tensor `src` into device tensor `dst` of the same dtype and size.
pub fn copy_h2d(dst: &Tensor, src: &Tensor) {
    check(dst, src, src);

    if dst.numel() == 0 {
        return;
    }

    let _op = transfer(&COPY_H2D, dst, src, dst.device());
    COPY_H2D.dispatch(dst.device())(dst, src);
}

/// Copies device tensor `src` into CPU tensor `dst`, waiting for `src`'s pending writes first.
pub fn copy_d2h(dst: &Tensor, src: &Tensor) {
    check(dst, src, dst);

    if src.numel() == 0 {
        return;
    }

    let _op = transfer(&COPY_D2H, dst, src, src.device());
    src.wait_();
    COPY_D2H.dispatch(src.device())(dst, src);
}

/// A profiler record for a copy on `device`; none for CPU-to-CPU copies.
fn transfer(
    op: &Op<CopyKernel>,
    dst: &Tensor,
    src: &Tensor,
    device: Device,
) -> Option<crate::profiler::RecordGuard> {
    if device == Device::Cpu(0) {
        return None;
    }

    let mut guard = crate::profiler::record_op(op.name(), || vec![src.ty()]);
    guard.outputs(|| vec![dst.ty()]);
    Some(guard)
}

/// Panics unless `host` is on CPU and the two tensors are distinct, contiguous and alike in dtype and size.
fn check(dst: &Tensor, src: &Tensor, host: &Tensor) {
    assert_eq!(
        host.device(),
        Device::Cpu(0),
        "the host side of a copy must be a CPU tensor"
    );

    assert_eq!(dst.dtype(), src.dtype(), "copy between dtypes");
    assert_eq!(dst.numel(), src.numel(), "copy between sizes");

    assert!(
        dst.is_contiguous() && src.is_contiguous(),
        "copy of a non-contiguous tensor"
    );

    assert!(!dst.shares_storage_with(src), "copy within one storage");
}
