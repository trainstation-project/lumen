//! `fill_`, dispatched to a kernel per backend (PyTorch: `aten::fill_`):
//! `cpu.rs` and `mps/` in Rust; CUDA's is `cuda.py`, a CuTe DSL kernel
//! registered from Python (see [`crate::ops::python`]).

mod cpu;
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
        DispatchKey::Cuda => None,
    }
}

/// Set every element of `t` to `value` with the kernel for its device.
pub fn fill_op(t: &Tensor, value: Scalar) {
    if t.numel() == 0 {
        return;
    }

    #[cfg(feature = "python")]
    if let Some(handle) =
        crate::ops::python::handle_for("lumen::fill_", DispatchKey::of(t.device()))
    {
        let (shape, strides) = fill_layout(t.shape(), t.strides());
        let address = t.data_ptr() as usize;
        if handle.launch(
            "lumen::fill_",
            t.device(),
            t.dtype(),
            &shape,
            &strides,
            address,
            value,
        ) {
            return;
        }
    }

    FILL.dispatch(t.device())(t, value);
}

/// A view's layout as a Python `fill_` kernel is handed it: a fill's order
/// does not matter, so the dimensions are sorted by stride, smallest first
/// (a kernel walking its first dimension fastest then touches memory in
/// order), size-1 dimensions are dropped, and dimensions contiguous with
/// each other are merged. A contiguous tensor becomes one dimension of
/// stride 1, the layout a kernel vectorizes best; a 0-d tensor is `[1]`.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn fill_layout(shape: &[usize], strides: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut dims: Vec<(usize, usize)> = shape
        .iter()
        .zip(strides)
        .filter(|&(&n, _)| n != 1)
        .map(|(&n, &s)| (n, s))
        .collect();
    dims.sort_by_key(|&(_, s)| s);
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(dims.len());
    for (n, s) in dims {
        match merged.last_mut() {
            Some((last_n, last_s)) if *last_s * *last_n == s => *last_n *= n,
            _ => merged.push((n, s)),
        }
    }
    if merged.is_empty() {
        return (vec![1], vec![1]);
    }
    merged.into_iter().unzip()
}
