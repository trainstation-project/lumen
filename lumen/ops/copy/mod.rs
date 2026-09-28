//! Host<->device copies, dispatched to a kernel per backend (PyTorch:
//! `aten::copy_`): `copy_h2d` moves host bytes into device memory and
//! `copy_d2h` moves device bytes out.
//!
//! These are the ops `Storage` calls whenever it has to touch a buffer the
//! host may not be able to address. Each kernel gets the tensor's allocator,
//! because the device alone does not say how its memory is reached: a CUDA
//! kernel needs `cudaMemcpy`, a CPU kernel is a plain `memcpy`, and a device
//! whose memory *happens* to be host-addressable (MPS's unified buffers)
//! still has to go through the allocator, since a custom allocator claiming
//! that device may not hand out addressable memory at all.

mod cpu;
#[cfg(lumen_cuda_linked)]
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;

use crate::allocator::Allocator;
use crate::device::Device;
use crate::ops::{DispatchKey, Op};

/// A host-to-device copy: `nbytes` from the host at `src` to device memory
/// at `dst` on `device`, whose buffers belong to `allocator`.
pub type CopyH2dKernel =
    unsafe fn(alloc: &dyn Allocator, device: Device, dst: *mut u8, src: *const u8, nbytes: usize);

/// A device-to-host copy: `nbytes` from device memory at `src` on `device`
/// to the host at `dst`, whose buffers belong to `allocator`.
pub type CopyD2hKernel =
    unsafe fn(alloc: &dyn Allocator, device: Device, dst: *mut u8, src: *const u8, nbytes: usize);

/// Host-to-device copy, dispatched on the destination's device.
pub static COPY_H2D: Op<CopyH2dKernel> = Op::new("lumen::copy_h2d", h2d_kernels);

/// Device-to-host copy, dispatched on the source's device.
pub static COPY_D2H: Op<CopyD2hKernel> = Op::new("lumen::copy_d2h", d2h_kernels);

/// `copy_h2d`'s static registry.
fn h2d_kernels(key: DispatchKey) -> Option<CopyH2dKernel> {
    match key {
        DispatchKey::Cpu => Some(cpu::copy_h2d),
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(mps::copy_h2d),
        #[cfg(not(lumen_mps_linked))]
        DispatchKey::Mps => None,
        #[cfg(lumen_cuda_linked)]
        DispatchKey::Cuda => Some(cuda::copy_h2d),
        #[cfg(not(lumen_cuda_linked))]
        DispatchKey::Cuda => None,
    }
}

/// `copy_d2h`'s static registry.
fn d2h_kernels(key: DispatchKey) -> Option<CopyD2hKernel> {
    match key {
        DispatchKey::Cpu => Some(cpu::copy_d2h),
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(mps::copy_d2h),
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
