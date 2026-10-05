mod cpu;
#[cfg(lumen_cuda_linked)]
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;

use crate::Tensor;
use crate::device::Device;
use crate::ops::{DispatchKey, Op};

pub type CopyKernel = fn(dst: &Tensor, src: &Tensor);

/// Host-to-device copy, dispatched on the destination's device.
pub static COPY_H2D: Op<CopyKernel> = Op::new(op_name!("copy_h2d"), h2d_kernels);

/// Device-to-host copy, dispatched on the source's device.
pub static COPY_D2H: Op<CopyKernel> = Op::new(op_name!("copy_d2h"), d2h_kernels);

/// `copy_h2d`'s static registry.
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
        // Meta tensors have no data to copy (`Tensor::copy_to`).
        DispatchKey::Meta => None,
    }
}

/// `copy_d2h`'s static registry.
fn d2h_kernels(key: DispatchKey) -> Option<CopyKernel> {
    match key {
        DispatchKey::Cpu => Some(cpu::memcpy),
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(cpu::memcpy),
        #[cfg(not(lumen_mps_linked))]
        DispatchKey::Mps => None,
        #[cfg(lumen_cuda_linked)]
        DispatchKey::Cuda => Some(cuda::copy_d2h),
        #[cfg(not(lumen_cuda_linked))]
        DispatchKey::Cuda => None,
        DispatchKey::Meta => None,
    }
}

/// Copy the host tensor `src` into `dst`, on any device.
///
/// Like [`Tensor::set`], this writes shared data through a shared
/// reference; the caller must ensure no data races.
///
/// # Panics
/// If `src` is not on the CPU, the two differ in dtype or size, either is
/// not contiguous, or they share storage.
///
/// Into MPS memory, the copy is in stream order: after the work submitted
/// so far, with no wait ([`mps::copy_h2d`]). Onto a device, profiled as
/// `copy_h2d` (`src`'s type in, `dst`'s out).
pub fn copy_h2d(dst: &Tensor, src: &Tensor) {
    check(dst, src, src);
    if dst.numel() == 0 {
        return;
    }
    let _op = transfer(&COPY_H2D, dst, src, dst.device());
    COPY_H2D.dispatch(dst.device())(dst, src);
}

/// Copy `src`, on any device, into the host tensor `dst`.
///
/// From a device, profiled as `copy_d2h`, the wait for `src`'s value (a
/// `wait`) inside it.
///
/// # Panics
/// As [`copy_h2d`], with `dst` the one that must be on the CPU.
pub fn copy_d2h(dst: &Tensor, src: &Tensor) {
    check(dst, src, dst);
    if src.numel() == 0 {
        return;
    }
    let _op = transfer(&COPY_D2H, dst, src, src.device());
    src.wait_();
    COPY_D2H.dispatch(src.device())(dst, src);
}

/// A transfer's profiler range, `op`'s name, if it moves data between the
/// host and `device` (none for a copy between host buffers).
fn transfer(
    op: &Op<CopyKernel>,
    dst: &Tensor,
    src: &Tensor,
    device: Device,
) -> Option<crate::profiler::RecordGuard> {
    if device == Device::Cpu {
        return None;
    }
    let mut guard = crate::profiler::record_op(op.name(), || vec![src.ty()]);
    guard.outputs(|| vec![dst.ty()]);
    Some(guard)
}

/// What every kernel relies on: `host` is on the CPU, and `dst` and `src`
/// are distinct contiguous buffers of the same dtype and size, so a kernel
/// copies `numel * size_of` bytes between their data pointers.
fn check(dst: &Tensor, src: &Tensor, host: &Tensor) {
    assert_eq!(
        host.device(),
        Device::Cpu,
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
