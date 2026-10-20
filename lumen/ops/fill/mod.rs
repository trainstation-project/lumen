mod cpu;
#[cfg(lumen_mps_linked)]
mod mps;

use crate::Tensor;
use crate::ops::{DispatchKey, Op};
use crate::tensor::dtype::{Element, dispatch_dtype};
use crate::tensor::scalar::Scalar;

pub type FillKernel = fn(&Tensor, Scalar);

pub static FILL: Op<FillKernel> = Op::new(op_name!("fill_"), fill_kernels);

#[cfg(feature = "python")]
pub static FILL_PY: Op<crate::ops::python::KernelHandle> =
    Op::new(op_name!("fill___python"), |_| None);

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
        DispatchKey::Cuda | DispatchKey::Tpu => None,
        DispatchKey::Meta => Some(|_, _| {}),
    }
}

pub fn fill_op(t: &Tensor, value: Scalar) {
    if t.numel() == 0 {
        return;
    }

    #[cfg(feature = "python")]
    if let Some(handle) = crate::ops::python::handle_for(FILL.name(), DispatchKey::of(t.device())) {
        let (shape, strides) = fill_layout(t.shape(), t.strides());
        let address = t.data_ptr() as usize;
        if handle.launch(
            FILL.name(),
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
