//! `lumen::dummy_op`: an op with no built-in kernels, for testing Python
//! kernel registration (`lumen.ops.register`) without replacing a real op's
//! kernel, such as CUDA's `fill_`. Tests call it as
//! `lumen._C._dummy_op(tensor, value)`.

use crate::Tensor;
use crate::ops::Op;
use crate::tensor::scalar::Scalar;

/// A `dummy_op` kernel: the tensor and the value, like `fill_`'s.
pub type DummyOpKernel = fn(&Tensor, Scalar);

/// `lumen::dummy_op`, dispatched on the tensor's device. It has no built-in
/// kernels, so it only runs a registered one.
pub static DUMMY_OP: Op<DummyOpKernel> = Op::new(op_name!("dummy_op"), |_| None);

/// Python `dummy_op` kernels, by device key (see [`crate::ops::python`]).
#[cfg(feature = "python")]
pub static DUMMY_OP_PY: Op<crate::ops::python::KernelHandle> =
    Op::new(op_name!("dummy_op__python"), |_| None);

/// Run the Python kernel registered for `t`'s device.
///
/// # Panics
/// If none is registered.
pub fn dummy_op(t: &Tensor, value: Scalar) {
    #[cfg(feature = "python")]
    if let Some(handle) =
        crate::ops::python::handle_for(DUMMY_OP.name(), crate::ops::DispatchKey::of(t.device()))
        && handle.launch(
            DUMMY_OP.name(),
            t.device(),
            t.dtype(),
            t.shape(),
            t.strides(),
            t.data_ptr() as usize,
            value,
        )
    {
        return;
    }
    DUMMY_OP.dispatch(t.device())(t, value);
}
