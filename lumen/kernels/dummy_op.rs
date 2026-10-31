//! `dummy_op`, a fill-like test op with no built-in kernels that exercises Python kernel registration.
use crate::Tensor;
use crate::runtime::dispatch::Op;
use crate::tensor::scalar::Scalar;

/// Native kernel signature: writes `Scalar` into the tensor in place.
pub type DummyOpKernel = fn(&Tensor, Scalar);

/// Native kernel table for `dummy_op`; empty until a kernel is registered.
pub static DUMMY_OP: Op<DummyOpKernel> = Op::new(op_name!("dummy_op"), |_| None);

/// Python kernel handles registered for `dummy_op`, kept apart from native kernels since their type differs.
#[cfg(feature = "python")]
pub static DUMMY_OP_PY: Op<crate::kernels::python::KernelHandle> =
    Op::new(op_name!("dummy_op__python"), |_| None);

/// Runs `dummy_op` on `t` with `value`, preferring a registered Python kernel and falling back to a native one.
pub fn dummy_op(t: &Tensor, value: Scalar) {
    #[cfg(feature = "python")]
    if let Some(handle) = crate::kernels::python::handle_for(
        DUMMY_OP.name(),
        crate::runtime::dispatch::DispatchKey::of(t.device()),
    ) && handle.launch(
        DUMMY_OP.name(),
        t.device(),
        t.dtype(),
        t.shape(),
        t.strides(),
        t.data_ptr() as usize,
        value,
    ) {
        return;
    }

    DUMMY_OP.dispatch(t.device())(t, value);
}
