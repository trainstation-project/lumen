mod cpu;
#[cfg(lumen_cuda_linked)]
mod cuda;

use crate::Tensor;
use crate::device::Device;
use crate::ops::{DispatchKey, Op};

pub type CopyKernel = fn(dst: &Tensor, src: &Tensor);

/// Host-to-device copy, dispatched on the destination's device.
pub static COPY_H2D: Op<CopyKernel> = Op::new("lumen::copy_h2d", h2d_kernels);

/// Device-to-host copy, dispatched on the source's device.
pub static COPY_D2H: Op<CopyKernel> = Op::new("lumen::copy_d2h", d2h_kernels);

/// `copy_h2d`'s static registry.
fn h2d_kernels(key: DispatchKey) -> Option<CopyKernel> {
    match key {
        DispatchKey::Cpu => Some(cpu::memcpy),
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(cpu::memcpy),
        #[cfg(not(lumen_mps_linked))]
        DispatchKey::Mps => None,
        #[cfg(lumen_cuda_linked)]
        DispatchKey::Cuda => Some(cuda::copy_h2d),
        #[cfg(not(lumen_cuda_linked))]
        DispatchKey::Cuda => None,
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
    }
}

/// Copy `nbytes` from host memory at `src` into `allocator`'s memory on
/// `device` at `dst`.
///
/// # Safety
/// `dst` must be memory from `allocator` on `device` valid for writing
/// `nbytes`, `src` must be valid for reading `nbytes`, and the two must not
/// overlap. The caller also waits for device work in flight
/// (`Storage::synchronize`) first.
pub unsafe fn copy_h2d(
    allocator: &dyn Allocator,
    device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    if nbytes == 0 {
        return;
    }
    // SAFETY: the caller's contract.
    unsafe { COPY_H2D.dispatch(device)(allocator, device, dst, src, nbytes) }
}

/// Copy `nbytes` from `allocator`'s memory on `device` at `src` into host
/// memory at `dst`.
///
/// # Safety
/// As [`copy_h2d`], with the roles of source and destination swapped.
pub unsafe fn copy_d2h(
    allocator: &dyn Allocator,
    device: Device,
    dst: *mut u8,
    src: *const u8,
    nbytes: usize,
) {
    if nbytes == 0 {
        return;
    }
    // SAFETY: the caller's contract.
    unsafe { COPY_D2H.dispatch(device)(allocator, device, dst, src, nbytes) }
}
