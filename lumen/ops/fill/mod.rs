//! `fill_`, dispatched to a kernel per backend (PyTorch: `aten::fill_`):
//! `cpu.rs`, `mps.rs` and `cuda.rs`.

mod cpu;
#[cfg(lumen_cuda_linked)]
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;

use crate::Tensor;
use crate::ops::{DispatchKey, Op};
use crate::tensor::dtype::{Element, dispatch_dtype};
use crate::tensor::scalar::Scalar;

/// A `fill_` kernel: the tensor and the value, not yet converted to its dtype.
pub type FillKernel = fn(&Tensor, Scalar);

/// `fill_`, dispatched on the tensor's device.
pub static FILL: Op<FillKernel> = Op::new("lumen::fill_", fill_kernels);

/// Python `fill_` kernels, by device key (see [`crate::ops::python`]).
#[cfg(feature = "python")]
pub static FILL_PY: Op<crate::ops::python::KernelHandle> =
    Op::new("lumen::fill___python", |_| None);

/// `fill_`'s static registry.
fn fill_kernels(key: DispatchKey) -> Option<FillKernel> {
    match key {
        DispatchKey::Cpu => Some(
            |t, value: Scalar| dispatch_dtype!(t.dtype(), T => cpu::fill(t, T::from_scalar(value))),
        ),
        #[cfg(lumen_mps_linked)]
        DispatchKey::Mps => Some(
            |t, value: Scalar| dispatch_dtype!(t.dtype(), T => mps::fill(t, T::from_scalar(value))),
        ),
        #[cfg(not(lumen_mps_linked))]
        DispatchKey::Mps => None,
        #[cfg(lumen_cuda_linked)]
        DispatchKey::Cuda => {
            Some(|t, value| dispatch_dtype!(t.dtype(), T => cuda::fill(t, T::from_scalar(value))))
        }
        #[cfg(not(lumen_cuda_linked))]
        DispatchKey::Cuda => None,
    }
}

/// Set every element of `t` to `value` with the kernel for its device.
pub fn fill_op(t: &Tensor, value: Scalar) {
    #[cfg(feature = "python")]
    if let Some(handle) =
        crate::ops::python::handle_for("lumen::fill_", DispatchKey::of(t.device()))
        && handle.call_fill("lumen::fill_", t, value)
    {
        return;
    }
    FILL.dispatch(t.device())(t, value);
}
